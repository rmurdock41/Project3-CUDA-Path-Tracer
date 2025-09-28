#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "../stream_compaction/efficient.h"  

#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <climits>   // for INT_MAX
#include <cfloat>    // for FLT_MAX
using StreamCompaction::Efficient::scanDevice;

#define ENABLE_MATERIAL_SORT 1
#define ENABLE_STREAM_COMPACTION 1

static bool gEnableMaterialSortRuntime = true;  
static bool gEnableStreamCompaction = true;   
void SetStreamCompactionEnabled(bool v) { gEnableStreamCompaction = v; }
bool GetStreamCompactionEnabled() { return gEnableStreamCompaction; }


static int* dev_matKeys = nullptr;
static int* dev_indices = nullptr;
static PathSegment* dev_paths_sorted = nullptr;
static ShadeableIntersection* dev_intersections_sorted = nullptr;
static int* dev_aliveFlags = nullptr;
static int* dev_scanIndices = nullptr;
static PathSegment* dev_paths_compacted = nullptr;
static ShadeableIntersection* dev_intersections_compacted = nullptr; 


#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}


__global__ void buildMaterialKeys(
    int num_paths,
    const ShadeableIntersection* __restrict__ inters,
    int* __restrict__ keys)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_paths) return;

    const ShadeableIntersection& isect = inters[i];
    keys[i] = (isect.t > 0.0f) ? isect.materialId : INT_MAX;
}



__global__ void kernFlagAlive(int n, PathSegment* paths, int* flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (paths[i].remainingBounces > 0) {
        flags[i] = 1;
    }
    else {
        flags[i] = 0;
        paths[i].color = glm::vec3(0.0f);
    }
}

__global__ void kernScatterPaths(
    int n, const PathSegment* inPaths,
    const int* flags, const int* indices,
    PathSegment* outPaths)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (flags[i]) {
        int dst = indices[i];
        outPaths[dst] = inPaths[i];
    }
}


__global__ void kernScatterIntersections(
    int n, const ShadeableIntersection* inIsect,
    const int* flags, const int* indices,
    ShadeableIntersection* outIsect)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (flags[i]) {
        int dst = indices[i];
        outIsect[dst] = inIsect[i];
    }
}

__global__ void accumulateTerminated(int n, const PathSegment* paths, glm::vec3* image) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (paths[i].remainingBounces <= 0) {
        int p = paths[i].pixelIndex;
        // atomic add into RGB to avoid races
        atomicAdd(&image[p].x, paths[i].color.x);
        atomicAdd(&image[p].y, paths[i].color.y);
        atomicAdd(&image[p].z, paths[i].color.z);
    }
}



//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need

	//material sort
#if ENABLE_MATERIAL_SORT
    cudaMalloc(&dev_matKeys, pixelcount * sizeof(int));
    cudaMalloc(&dev_indices, pixelcount * sizeof(int));
    cudaMalloc(&dev_paths_sorted, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_intersections_sorted, pixelcount * sizeof(ShadeableIntersection));
#endif


	//stream compaction
#if ENABLE_STREAM_COMPACTION
    cudaMalloc(&dev_aliveFlags, pixelcount * sizeof(int));
    cudaMalloc(&dev_scanIndices, pixelcount * sizeof(int));
    cudaMalloc(&dev_paths_compacted, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_intersections_compacted, pixelcount * sizeof(ShadeableIntersection));
#endif


    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);

	//material sort
#if ENABLE_MATERIAL_SORT
    cudaFree(dev_matKeys);
    cudaFree(dev_indices);
    cudaFree(dev_paths_sorted);
    cudaFree(dev_intersections_sorted);
#endif

	//stream compaction
#if ENABLE_STREAM_COMPACTION
    cudaFree(dev_aliveFlags);
    cudaFree(dev_scanIndices);
    cudaFree(dev_paths_compacted);
    cudaFree(dev_intersections_compacted);
#endif

    // TODO: clean up any extra device memory you created

    checkCUDAError("pathtraceFree");
}

// ===== dof helper function =====
// Concentric disk sampling (returns a point on unit disk)
__device__ __forceinline__ glm::vec2 concentricSampleDisk(float u1, float u2) {
    // map [0,1)^2 -> [-1,1]^2
    float sx = 2.0f * u1 - 1.0f;
    float sy = 2.0f * u2 - 1.0f;

    if (sx == 0.0f && sy == 0.0f) return glm::vec2(0.0f);

    float r, theta;
    if (fabsf(sx) > fabsf(sy)) {
        r = sx;
        theta = (PI * 0.25f) * (sy / fmaxf(fabsf(sx), 1e-8f));
    }
    else {
        r = sy;
        theta = (PI * 0.5f) - (PI * 0.25f) * (sx / fmaxf(fabsf(sy), 1e-8f));
    }
    return r * glm::vec2(cosf(theta), sinf(theta));
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // SSAA jitter (thrust RNG stable per pixel/iter)
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float jx = dist(rng);
        float jy = dist(rng);

        // Subpixel coordinates with jitter
        float sx = (float)x + jx - (float)cam.resolution.x * 0.5f;
        float sy = (float)y + jy - (float)cam.resolution.y * 0.5f;

        glm::vec3 baseDir = glm::normalize(
            cam.view
            - cam.right * cam.pixelLength.x * sx
            - cam.up * cam.pixelLength.y * sy
        );

        // Thin-lens DoF
        if (cam.apertureRadius > 0.0f && cam.focalDistance > 0.0f) {
            // Intersect with focus plane at distance 'focalDistance' along camera view
            float cosToView = glm::dot(baseDir, cam.view);
            float tFocus = cam.focalDistance / fmaxf(cosToView, 1e-6f);
            glm::vec3 pFocus = cam.position + baseDir * tFocus;

            // Sample lens disk (radius = apertureRadius)
            glm::vec2 d = concentricSampleDisk(dist(rng), dist(rng)) * cam.apertureRadius;
            glm::vec3 lensOffset = cam.right * d.x + cam.up * d.y;

            segment.ray.origin = cam.position + lensOffset;
            segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);
        }
        else {
            // Fallback
            segment.ray.origin = cam.position;
            segment.ray.direction = baseDir;
        }

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}



__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    PathSegment& pathSegment = pathSegments[idx];

    // Do not process already-terminated paths (prevents repeated light hits)
    if (pathSegment.remainingBounces <= 0) return;

    ShadeableIntersection intersection = shadeableIntersections[idx];

    if (intersection.t > 0.0f) {
        Material material = materials[intersection.materialId];

        // Emissive: accumulate once, then terminate and return
        if (material.emittance > 0.0f) {
            pathSegment.color *= (material.color * material.emittance);
            pathSegment.remainingBounces = 0;
            return; 
        }


        // Common data: hit point, normal, and RNG bound to pixelIndex so it's stable under sorting/compaction
        glm::vec3 p = pathSegment.ray.origin + intersection.t * pathSegment.ray.direction;
        glm::vec3 n = glm::normalize(intersection.surfaceNormal);
        thrust::default_random_engine rng = makeSeededRandomEngine(
            iter, pathSegment.pixelIndex, pathSegment.remainingBounces);
        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

        // === Refractive (glass) branch with Schlick Fresnel ===
        if (material.hasRefractive) {
            // wo points into the scene
            const glm::vec3 wo = pathSegment.ray.direction;

            // Determine whether we are entering or exiting
            bool entering = glm::dot(wo, n) < 0.0f;
            glm::vec3 N = entering ? n : -n;

            // Indices of refraction
            float etaI = entering ? 1.0f : material.indexOfRefraction;
            float etaT = entering ? material.indexOfRefraction : 1.0f;
            float eta = etaI / etaT;

            // cos(theta_i) in [0,1]
            float cosI = fminf(1.0f, fmaxf(0.0f, -glm::dot(wo, N)));

            // Try to refract; total internal reflection if glm::refract returns (0,0,0)
            glm::vec3 tdir = glm::refract(wo, N, eta);
            bool tir = (tdir.x == 0.0f && tdir.y == 0.0f && tdir.z == 0.0f);

            // Schlick approximation for reflection probability
            float r0 = (etaI - etaT) / (etaI + etaT);
            r0 = r0 * r0;
            float R = r0 + (1.0f - r0) * powf(1.0f - cosI, 5.0f);

            float xi = u01(rng);

            // Small offset to avoid self-intersection
            const float EPS = 1e-3f;

            if (tir || xi < R) {
                // Reflect
                glm::vec3 rdir = glm::reflect(wo, N);
                pathSegment.ray.origin = p + N * EPS;
                pathSegment.ray.direction = glm::normalize(rdir);
            }
            else {
                // Refract
                pathSegment.ray.origin = p - N * EPS;
                pathSegment.ray.direction = glm::normalize(tdir);
            }

            pathSegment.remainingBounces--;
            return; 
        }


        // Still has bounces left: scatter
        if (pathSegment.remainingBounces > 0) {
            scatterRay(pathSegment, p, n, material, rng);
        }
        else {
            pathSegment.color = glm::vec3(0.0f);
            pathSegment.remainingBounces = 0;
        }
    }
    else {
        // Miss: no environment light then go black and terminate
        pathSegment.color = glm::vec3(0.0f);
        pathSegment.remainingBounces = 0;
    }
}


// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

#if ENABLE_MATERIAL_SORT
        if (gEnableMaterialSortRuntime) {
            if (num_paths > 0) {
                const int blocks = (num_paths + blockSize1d - 1) / blockSize1d;
                if (blocks > 0) {
                    buildMaterialKeys << <blocks, blockSize1d >> > (num_paths, dev_intersections, dev_matKeys);
                    checkCUDAError("buildMaterialKeys");

                    thrust::sequence(thrust::device, dev_indices, dev_indices + num_paths, 0);
                    thrust::stable_sort_by_key(
                        thrust::device,
                        dev_matKeys, dev_matKeys + num_paths,
                        dev_indices
                    );

                    thrust::gather(
                        thrust::device, dev_indices, dev_indices + num_paths,
                        dev_paths, dev_paths_sorted
                    );
                    thrust::gather(
                        thrust::device, dev_indices, dev_indices + num_paths,
                        dev_intersections, dev_intersections_sorted
                    );

                    PathSegment* tmpP = dev_paths; dev_paths = dev_paths_sorted; dev_paths_sorted = tmpP;
                    ShadeableIntersection* tmpI = dev_intersections; dev_intersections = dev_intersections_sorted; dev_intersections_sorted = tmpI;
                }
            }
        }
#endif

        shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
            );
        checkCUDAError("shadeMaterial");

#if ENABLE_STREAM_COMPACTION
        if (gEnableStreamCompaction) {
            const int block = 128;
            const int blocks = (num_paths + block - 1) / block;
            if (num_paths > 0 && blocks > 0) {
                accumulateTerminated << <blocks, block >> > (num_paths, dev_paths, dev_image);
                checkCUDAError("accumulateTerminated");

                kernFlagAlive << <blocks, block >> > (num_paths, dev_paths, dev_aliveFlags);
                checkCUDAError("kernFlagAlive");

                scanDevice(num_paths, dev_scanIndices, dev_aliveFlags);

                int lastFlag = 0, lastIndex = 0;
                cudaMemcpy(&lastFlag, dev_aliveFlags + (num_paths - 1), sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&lastIndex, dev_scanIndices + (num_paths - 1), sizeof(int), cudaMemcpyDeviceToHost);
                int newCount = lastIndex + lastFlag;

                if (newCount > 0) {
                    kernScatterPaths << <blocks, block >> > (
                        num_paths, dev_paths, dev_aliveFlags, dev_scanIndices, dev_paths_compacted);
                    checkCUDAError("scatter paths");

                    PathSegment* tmpP = dev_paths; dev_paths = dev_paths_compacted; dev_paths_compacted = tmpP;
                    num_paths = newCount;
                }
                else {
                    num_paths = 0;
                }

                iterationComplete = iterationComplete || (num_paths == 0);
            }
        }
#endif

        iterationComplete = iterationComplete || (depth >= traceDepth);

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
