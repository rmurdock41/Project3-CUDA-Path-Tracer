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

#include <vector>
#include <algorithm>
#include "mesh_loader.h"

#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <climits>   // for INT_MAX
#include <cfloat>    // for FLT_MAX
using StreamCompaction::Efficient::scanDevice;

#define ENABLE_MATERIAL_SORT 1
#define ENABLE_STREAM_COMPACTION 1

// ==== BVH feature toggle ====
#define ENABLE_BVH 1

static bool gEnableBVH = true; 
void SetBVHEnabled(bool v) { gEnableBVH = v; }
bool GetBVHEnabled() { return gEnableBVH; }

// ===== Russian Roulette (RR) =====
#define ENABLE_RR 1

static bool gEnableRR = false;  
static int  gRRMinDepth = 3;     

void SetRREnabled(bool v) { gEnableRR = v; }
bool GetRREnabled() { return gEnableRR; }
void SetRRMinDepth(int d) { gRRMinDepth = d; }
int  GetRRMinDepth() { return gRRMinDepth; }



// ===== GPU Tris =====
struct Tri {
    glm::vec3 v0, v1, v2;
    int       materialId;
};

static Tri* dev_tris = nullptr;
static int  g_numTris = 0;
static bool gEnableMeshCull = true;
void SetMeshCullEnabled(bool v) { gEnableMeshCull = v; }
bool GetMeshCullEnabled() { return gEnableMeshCull; }
static std::vector<TriCPU> h_allTris;

static bool gEnableMaterialSortRuntime = true;  
static bool gEnableStreamCompaction = true;   
void SetStreamCompactionEnabled(bool v) { gEnableStreamCompaction = v; }
bool GetStreamCompactionEnabled() { return gEnableStreamCompaction; }
void SetMaterialSortEnabled(bool v) { gEnableMaterialSortRuntime = v; }
bool GetMaterialSortEnabled() { return gEnableMaterialSortRuntime; }

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

#if ENABLE_BVH
struct BVHNode;
static struct BVHNode* dev_bvhNodes = nullptr;
static int* dev_primIndices = nullptr;
#endif


// ===== AABB / BVH structs & helpers ======
struct AABB {
    glm::vec3 minB;
    glm::vec3 maxB;
};

struct BVHNode {
    AABB box;
    int left;       
    int right;      
    int firstPrim;  
    int primCount;  
};

struct TriBVHNode {
    AABB box;
    int  left;
    int  right;
    int  firstTri;  
    int  triCount;  
};



static TriBVHNode* dev_triBVHNodes = nullptr;
static int* dev_triPrimIdx = nullptr;

static std::vector<TriBVHNode> h_triBVHNodes;
static std::vector<int>        h_triPrimIdx;

static bool gEnableTriBVH = true;
void SetTriBVHEnabled(bool v) { gEnableTriBVH = v; }
bool GetTriBVHEnabled() { return gEnableTriBVH; }



__host__ __device__ inline AABB makeEmptyAABB() {
    AABB b;
    b.minB = glm::vec3(FLT_MAX);
    b.maxB = glm::vec3(-FLT_MAX);
    return b;
}

__host__ __device__ inline void expandAABB(AABB& b, const AABB& c) {
    b.minB = glm::min(b.minB, c.minB);
    b.maxB = glm::max(b.maxB, c.maxB);
}

// slab 
__host__ __device__ inline bool intersectAABB(const AABB& box, const Ray& r, float tMax) {
    const float kEps = 1e-8f;
    glm::vec3 invD = glm::vec3(
        1.0f / ((fabsf(r.direction.x) > kEps) ? r.direction.x : (r.direction.x >= 0 ? kEps : -kEps)),
        1.0f / ((fabsf(r.direction.y) > kEps) ? r.direction.y : (r.direction.y >= 0 ? kEps : -kEps)),
        1.0f / ((fabsf(r.direction.z) > kEps) ? r.direction.z : (r.direction.z >= 0 ? kEps : -kEps)));

    glm::vec3 t0 = (box.minB - r.origin) * invD;
    glm::vec3 t1 = (box.maxB - r.origin) * invD;
    glm::vec3 tmin3 = glm::min(t0, t1);
    glm::vec3 tmax3 = glm::max(t0, t1);

    float tmin = fmaxf(fmaxf(tmin3.x, tmin3.y), tmin3.z);
    float tmax = fminf(fminf(tmax3.x, tmax3.y), fminf(tMax, tmax3.z));
    return tmax >= fmaxf(tmin, 0.0f);
}




__device__ __forceinline__ float intersectTriangleMT(const Ray& r, const Tri& tr) {
    const float EPS = 1e-7f;
    glm::vec3 e1 = tr.v1 - tr.v0;
    glm::vec3 e2 = tr.v2 - tr.v0;
    glm::vec3 p = glm::cross(r.direction, e2);
    float det = glm::dot(e1, p);
    if (fabsf(det) < EPS) return -1.f;
    float invDet = 1.f / det;
    glm::vec3 tvec = r.origin - tr.v0;
    float u = glm::dot(tvec, p) * invDet; if (u < 0.f || u > 1.f) return -1.f;
    glm::vec3 q = glm::cross(tvec, e1);
    float v = glm::dot(r.direction, q) * invDet; if (v < 0.f || u + v > 1.f) return -1.f;
    float t = glm::dot(e2, q) * invDet; if (t <= 0.f) return -1.f;
    return t;
}


// ===== CPU-side BVH build  =====
struct BuildPrim {
    AABB      box;
    glm::vec3 centroid;
    int       primId;   
};

static std::vector<BVHNode> h_bvhNodes;
static std::vector<int>     h_primIndices;

static int buildBVHRecursive(std::vector<BVHNode>& outNodes,
    std::vector<int>& outPrimIdx,
    std::vector<BuildPrim>& bp,
    int begin, int end)
{
    int nodeIdx = (int)outNodes.size();
    outNodes.push_back(BVHNode{}); 

    AABB bbox = makeEmptyAABB();
    AABB cbox = makeEmptyAABB();
    for (int i = begin; i < end; ++i) {
        expandAABB(bbox, bp[i].box);
        AABB cc; cc.minB = cc.maxB = bp[i].centroid;
        expandAABB(cbox, cc);
    }

    int count = end - begin;
    if (count <= 4) {
        BVHNode leaf;
        leaf.box = bbox;
        leaf.left = -1;
        leaf.right = -1;
        leaf.firstPrim = (int)outPrimIdx.size();
        leaf.primCount = count;
        for (int i = begin; i < end; ++i) outPrimIdx.push_back(bp[i].primId);
        outNodes[nodeIdx] = leaf;
        return nodeIdx;
    }

    glm::vec3 diag = cbox.maxB - cbox.minB;
    int axis = (diag.x > diag.y && diag.x > diag.z) ? 0 : (diag.y > diag.z ? 1 : 2);

    int mid = (begin + end) / 2;
    std::nth_element(bp.begin() + begin, bp.begin() + mid, bp.begin() + end,
        [axis](const BuildPrim& a, const BuildPrim& b) {
            return a.centroid[axis] < b.centroid[axis];
        });

    int L = buildBVHRecursive(outNodes, outPrimIdx, bp, begin, mid);
    int R = buildBVHRecursive(outNodes, outPrimIdx, bp, mid, end);

    BVHNode inner;
    inner.box = bbox;
    inner.left = L;
    inner.right = R;
    inner.firstPrim = -1;
    inner.primCount = 0;
    outNodes[nodeIdx] = inner;
    return nodeIdx;
}


static inline void getGeomAABBAndCentroid(const Geom& g, AABB& outBox, glm::vec3& outCentroid)
{
    const glm::mat4 M = g.transform;

    if (g.type == CUBE) {
        const float h = 0.5f; 
        glm::vec3 corners[8] = {
            {-h,-h,-h},{ h,-h,-h},{-h, h,-h},{ h, h,-h},
            {-h,-h, h},{ h,-h, h},{-h, h, h},{ h, h, h}
        };
        AABB b = makeEmptyAABB();
        glm::vec3 sum(0.f);
        for (int i = 0; i < 8; ++i) {
            glm::vec3 pw = glm::vec3(M * glm::vec4(corners[i], 1.f));
            b.minB = glm::min(b.minB, pw);
            b.maxB = glm::max(b.maxB, pw);
            sum += pw;
        }
        outBox = b;
        outCentroid = sum / 8.f;
    }
    else if (g.type == SPHERE) {
        const float r = 0.5f;
        glm::vec3 c = glm::vec3(M * glm::vec4(0, 0, 0, 1));
        float ex = r * (fabs(M[0][0]) + fabs(M[1][0]) + fabs(M[2][0]));
        float ey = r * (fabs(M[0][1]) + fabs(M[1][1]) + fabs(M[2][1]));
        float ez = r * (fabs(M[0][2]) + fabs(M[1][2]) + fabs(M[2][2]));
        glm::vec3 ext(ex, ey, ez);
        outBox.minB = c - ext;
        outBox.maxB = c + ext;
        outCentroid = c;
    }
    else if (g.type == MESH) {
        outBox.minB = g.bboxMin;
        outBox.maxB = g.bboxMax;
        outCentroid = 0.5f * (g.bboxMin + g.bboxMax);
    }
    else {
        glm::vec3 T = glm::vec3(M[3]);
        outBox.minB = T - glm::vec3(1e-3f);
        outBox.maxB = T + glm::vec3(1e-3f);
        outCentroid = T;
    }
}


struct TriBuildPrim {
    AABB      box;
    glm::vec3 centroid;
    int       triId;  
};

static int buildTriBVHRecursive(
    std::vector<TriBVHNode>& outNodes,
    std::vector<int>& outPrimIdx,
    std::vector<TriBuildPrim>& bp,
    int begin, int end)
{
    int nodeIdx = (int)outNodes.size();
    outNodes.push_back(TriBVHNode{});   

    AABB bbox = makeEmptyAABB();
    AABB cbox = makeEmptyAABB();
    for (int i = begin; i < end; ++i) {
        expandAABB(bbox, bp[i].box);
        AABB cc; cc.minB = cc.maxB = bp[i].centroid;
        expandAABB(cbox, cc);
    }

    int count = end - begin;
    if (count <= 4) { 
        TriBVHNode leaf;
        leaf.box = bbox;
        leaf.left = -1; leaf.right = -1;
        leaf.firstTri = (int)outPrimIdx.size();
        leaf.triCount = count;
        for (int i = begin; i < end; ++i) outPrimIdx.push_back(bp[i].triId);
        outNodes[nodeIdx] = leaf;
        return nodeIdx;
    }

    glm::vec3 diag = cbox.maxB - cbox.minB;
    int axis = (diag.x > diag.y && diag.x > diag.z) ? 0 : (diag.y > diag.z ? 1 : 2);
    int mid = (begin + end) / 2;
    std::nth_element(bp.begin() + begin, bp.begin() + mid, bp.begin() + end,
        [axis](const TriBuildPrim& a, const TriBuildPrim& b) {
            return a.centroid[axis] < b.centroid[axis];
        });

    int L = buildTriBVHRecursive(outNodes, outPrimIdx, bp, begin, mid);
    int R = buildTriBVHRecursive(outNodes, outPrimIdx, bp, mid, end);

    TriBVHNode inner;
    inner.box = bbox;
    inner.left = L; inner.right = R;
    inner.firstTri = -1; inner.triCount = 0;
    outNodes[nodeIdx] = inner;
    return nodeIdx;
}


static void buildAndUploadTriBVH(Scene* scene)
{
    h_triBVHNodes.clear();
    h_triPrimIdx.clear();

    for (auto& g : scene->geoms) {
        if (g.type != MESH || g.triCount == 0) { g.triBVHRoot = -1; continue; }

        std::vector<TriBuildPrim> bp;
        bp.reserve(g.triCount);

        for (int i = 0; i < g.triCount; ++i) {
            const TriCPU& t = h_allTris[g.triOffset + i];
            AABB b = makeEmptyAABB();
            b.minB = glm::min(t.v0, glm::min(t.v1, t.v2));
            b.maxB = glm::max(t.v0, glm::max(t.v1, t.v2));
            glm::vec3 c = (b.minB + b.maxB) * 0.5f;

            bp.push_back(TriBuildPrim{ b, c, g.triOffset + i });
        }

        int root = buildTriBVHRecursive(h_triBVHNodes, h_triPrimIdx, bp, 0, (int)bp.size());
        g.triBVHRoot = root;
    }


    cudaFree(dev_triBVHNodes); dev_triBVHNodes = nullptr;
    cudaFree(dev_triPrimIdx);  dev_triPrimIdx = nullptr;

    if (!h_triBVHNodes.empty()) {
        cudaMalloc(&dev_triBVHNodes, h_triBVHNodes.size() * sizeof(TriBVHNode));
        cudaMemcpy(dev_triBVHNodes, h_triBVHNodes.data(),
            h_triBVHNodes.size() * sizeof(TriBVHNode), cudaMemcpyHostToDevice);
    }
    if (!h_triPrimIdx.empty()) {
        cudaMalloc(&dev_triPrimIdx, h_triPrimIdx.size() * sizeof(int));
        cudaMemcpy(dev_triPrimIdx, h_triPrimIdx.data(),
            h_triPrimIdx.size() * sizeof(int), cudaMemcpyHostToDevice);
    }

    printf("[TriBVH] nodes=%zu, prims=%zu\n", h_triBVHNodes.size(), h_triPrimIdx.size());
}

#if ENABLE_BVH

static void buildAndUploadBVH(Scene* scene) {

    std::vector<BuildPrim> bp;
    bp.reserve(scene->geoms.size());
    for (int i = 0; i < (int)scene->geoms.size(); ++i) {
        AABB b; glm::vec3 c;
        getGeomAABBAndCentroid(scene->geoms[i], b, c);
        bp.push_back(BuildPrim{ b, c, i });
    }

    if (bp.empty()) {
        h_bvhNodes.clear(); h_primIndices.clear();
        cudaFree(dev_bvhNodes);    dev_bvhNodes = nullptr;
        cudaFree(dev_primIndices); dev_primIndices = nullptr;
        return;
    }

    h_bvhNodes.clear();
    h_primIndices.clear();
    h_bvhNodes.reserve(bp.size() * 2);
    buildBVHRecursive(h_bvhNodes, h_primIndices, bp, 0, (int)bp.size());


    cudaFree(dev_bvhNodes);    dev_bvhNodes = nullptr;
    cudaFree(dev_primIndices); dev_primIndices = nullptr;

    cudaMalloc(&dev_bvhNodes, h_bvhNodes.size() * sizeof(BVHNode));
    cudaMemcpy(dev_bvhNodes, h_bvhNodes.data(),
        h_bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_primIndices, h_primIndices.size() * sizeof(int));
    cudaMemcpy(dev_primIndices, h_primIndices.data(),
        h_primIndices.size() * sizeof(int), cudaMemcpyHostToDevice);


    printf("[BVH] nodes=%zu, prims=%zu\n", h_bvhNodes.size(), h_primIndices.size());

}



static void freeBVH() {
    cudaFree(dev_bvhNodes);     dev_bvhNodes = nullptr;
    cudaFree(dev_primIndices);  dev_primIndices = nullptr;
}
#endif

// ===== Device-side primitive intersection =====
__device__ inline bool intersectPrimitiveGeom(
    const Geom* geoms, int primId, const Ray& ray,
    const Tri* __restrict__ tris,
    const TriBVHNode* __restrict__ triNodes,
    const int* __restrict__ triPrimIdx,
    float& tHit, glm::vec3& nHit, int& matId)
{
    const Geom& g = geoms[primId];
    float t = -1.0f; glm::vec3 pTmp, nTmp; bool outside = true;

    if (g.type == CUBE) {
        t = boxIntersectionTest(g, ray, pTmp, nTmp, outside);
        if (t > 0.0f && t < tHit) { tHit = t; nHit = nTmp; matId = g.materialid; return true; }
        return false;
    }
    else if (g.type == SPHERE) {
        t = sphereIntersectionTest(g, ray, pTmp, nTmp, outside);
        if (t > 0.0f && t < tHit) { tHit = t; nHit = nTmp; matId = g.materialid; return true; }
        return false;
    }
    else if (g.type == MESH) {
        AABB mbox; mbox.minB = g.bboxMin; mbox.maxB = g.bboxMax;
        if (!intersectAABB(mbox, ray, tHit)) return false;

        if (triNodes == nullptr || triPrimIdx == nullptr || g.triBVHRoot < 0) {
            bool any = false;
            for (int k = 0; k < g.triCount; ++k) {
                const Tri& tr = tris[g.triOffset + k];
                float th = intersectTriangleMT(ray, tr);
                if (th > 0.0f && th < tHit) {
                    tHit = th;
                    nHit = glm::normalize(glm::cross(tr.v1 - tr.v0, tr.v2 - tr.v0));
                    matId = tr.materialId;
                    any = true;
                }
            }
            return any;
        }

        int stack[64]; int sp = 0;
        stack[sp++] = g.triBVHRoot;

        bool hit = false;
        while (sp) {
            const int ni = stack[--sp];
            const TriBVHNode& node = triNodes[ni];

            if (!intersectAABB(node.box, ray, tHit)) continue;

            if (node.triCount > 0) {
                for (int i = 0; i < node.triCount; ++i) {
                    const int triIdx = triPrimIdx[node.firstTri + i];
                    const Tri& tr = tris[triIdx];

                    float th = intersectTriangleMT(ray, tr);
                    if (th > 0.0f && th < tHit) {
                        tHit = th;
                        nHit = glm::normalize(glm::cross(tr.v1 - tr.v0, tr.v2 - tr.v0));
                        matId = tr.materialId;
                        hit = true;
                    }
                }
            }
            else {

                if (node.left >= 0) stack[sp++] = node.left;
                if (node.right >= 0) stack[sp++] = node.right;
            }
        }
        return hit;
    }

    return false;
}



// ===== BVH traversal =====
__device__ inline void traverseBVH(
    const Ray& ray,
    const BVHNode* __restrict__ nodes,
    const int* __restrict__ primIdx,
    const Geom* __restrict__ geoms,
    const Tri* __restrict__ tris,
    const TriBVHNode* __restrict__ triNodes,
    const int* __restrict__ triPrimIdx,
    float& outT, int& outGeom, glm::vec3& outN, int& outMat)
{
    int stack[64]; int sp = 0;
    float tClosest = FLT_MAX; int hitGeom = -1; glm::vec3 nHit(0.f); int mId = -1;

    stack[sp++] = 0;
    while (sp) {
        const int ni = stack[--sp];
        const BVHNode& node = nodes[ni];
        if (!intersectAABB(node.box, ray, tClosest)) continue;

        if (node.primCount > 0) {
            for (int i = 0; i < node.primCount; ++i) {
                const int pid = primIdx[node.firstPrim + i];
                float t = tClosest; glm::vec3 n; int mat;
                if (intersectPrimitiveGeom(geoms, pid, ray, tris, triNodes, triPrimIdx, t, n, mat)) {
                    if (t < tClosest) { tClosest = t; hitGeom = pid; nHit = n; mId = mat; }
                }
            }
        }
        else {
            if (node.left >= 0) stack[sp++] = node.left;
            if (node.right >= 0) stack[sp++] = node.right;
        }
    }
    outT = tClosest; outGeom = hitGeom; outN = nHit; outMat = mId;
}

__device__ inline bool traverseTriBVH(
    const Ray& ray,
    const TriBVHNode* __restrict__ nodes,
    const int* __restrict__ primIdx,
    int root,
    const Tri* __restrict__ tris,
    float& tHit, glm::vec3& nHit, int& matId)
{
    int stack[64]; int sp = 0;
    stack[sp++] = root;
    bool any = false;

    while (sp) {
        const int ni = stack[--sp];
        const TriBVHNode& n = nodes[ni];

        if (!intersectAABB(n.box, ray, tHit)) continue;

        if (n.triCount > 0) {
            // leaf
            for (int i = 0; i < n.triCount; ++i) {
                const int tid = primIdx[n.firstTri + i];  
                const Tri& tr = tris[tid];
                float th = intersectTriangleMT(ray, tr);
                if (th > 0.f && th < tHit) {
                    tHit = th;
                    nHit = glm::normalize(glm::cross(tr.v1 - tr.v0, tr.v2 - tr.v0));
                    matId = tr.materialId;
                    any = true;
                }
            }
        }
        else {
            if (n.left >= 0) stack[sp++] = n.left;
            if (n.right >= 0) stack[sp++] = n.right;
        }
    }
    return any;
}




// ===== CPU - side triangle upload =====



static void UploadTrisToGPU() {
    if (h_allTris.empty()) { g_numTris = 0; cudaFree(dev_tris); dev_tris = nullptr; return; }
    g_numTris = (int)h_allTris.size();
    cudaFree(dev_tris);
    cudaMalloc(&dev_tris, g_numTris * sizeof(Tri));

    std::vector<Tri> temp(g_numTris);
    for (int i = 0; i < g_numTris; ++i) {
        temp[i].v0 = h_allTris[i].v0;
        temp[i].v1 = h_allTris[i].v1;
        temp[i].v2 = h_allTris[i].v2;
        temp[i].materialId = h_allTris[i].materialId;
    }
    cudaMemcpy(dev_tris, temp.data(), g_numTris * sizeof(Tri), cudaMemcpyHostToDevice);
}


static void BakeMeshesIntoSceneAndCPUTris(Scene* scene) {
    h_allTris.clear();

    for (const auto& mi : scene->meshInstances) {
        std::vector<TriCPU> local;
        std::string err;
        if (!LoadGLTF_AsTris(mi.path, mi.M_world, mi.materialId, local, &err)) {
            printf("[GLTF] load failed: %s\n", err.c_str());
            continue;
        }


        glm::vec3 bbMin(FLT_MAX), bbMax(-FLT_MAX);
        for (const auto& t : local) {
            bbMin = glm::min(bbMin, glm::min(t.v0, glm::min(t.v1, t.v2)));
            bbMax = glm::max(bbMax, glm::max(t.v0, glm::max(t.v1, t.v2)));
        }

        Geom g{};
        g.type = MESH;
        g.materialid = mi.materialId;       
        g.triOffset = (int)h_allTris.size();
        g.triCount = (int)local.size();
        g.bboxMin = bbMin;
        g.bboxMax = bbMax;
        g.transform = mi.M_world;
        g.inverseTransform = glm::inverse(mi.M_world);
        g.invTranspose = glm::transpose(g.inverseTransform);
        g.triBVHRoot = -1;

        scene->geoms.push_back(g);


        h_allTris.insert(h_allTris.end(), local.begin(), local.end());
    }
}


// ===== Russian Roulette helper functions =====
__device__ __forceinline__ float luminance(const glm::vec3& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

__device__ __forceinline__ bool russianRoulette(
    PathSegment& path,
    thrust::default_random_engine& rng,
    int bouncesDone, int rrMinDepth)
{
    if (bouncesDone < rrMinDepth) return false;

    float p = fminf(fmaxf(luminance(path.color), 0.05f), 0.95f);
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    if (u01(rng) > p) {
        path.color = glm::vec3(0.0f);
        path.remainingBounces = 0;
        return true;
    }
    else {
        path.color /= p;
        return false;
    }
}


void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;


    BakeMeshesIntoSceneAndCPUTris(scene);
    UploadTrisToGPU();


    buildAndUploadTriBVH(scene);          
#if ENABLE_BVH
    buildAndUploadBVH(scene);              
#endif



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

    cudaFree(dev_tris); dev_tris = nullptr; g_numTris = 0;


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


#if ENABLE_BVH
    freeBVH();
#endif

    cudaFree(dev_triBVHNodes); dev_triBVHNodes = nullptr;
    cudaFree(dev_triPrimIdx);  dev_triPrimIdx = nullptr;


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
    ShadeableIntersection* intersections,
    const BVHNode* __restrict__ bvhNodes,
    const int* __restrict__ primIdx,
    const Tri* __restrict__ tris,
    const TriBVHNode* __restrict__ triNodes,
    const int* __restrict__ triPrimIdx)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index >= num_paths) return;

    const PathSegment seg = pathSegments[path_index];
    float     bestT = FLT_MAX;
    glm::vec3 bestN = glm::vec3(0.f);
    int       bestGeom = -1;
    int       bestMat = -1;

#if ENABLE_BVH
    if (bvhNodes != nullptr && primIdx != nullptr) {
        traverseBVH(seg.ray, bvhNodes, primIdx, geoms, tris,
            triNodes, triPrimIdx,              
            bestT, bestGeom, bestN, bestMat);

    }
    else
#endif
    {
        bool outside = true; glm::vec3 pTmp, nTmp;
        for (int i = 0; i < geoms_size; ++i) {
            const Geom& g = geoms[i];

            if (g.type == CUBE) {
                float t = boxIntersectionTest(g, seg.ray, pTmp, nTmp, outside);
                if (t > 0.0f && t < bestT) { bestT = t; bestN = nTmp; bestGeom = i; bestMat = g.materialid; }
            }
            else if (g.type == SPHERE) {
                float t = sphereIntersectionTest(g, seg.ray, pTmp, nTmp, outside);
                if (t > 0.0f && t < bestT) { bestT = t; bestN = nTmp; bestGeom = i; bestMat = g.materialid; }
            }
            else if (g.type == MESH) {
                if (tris == nullptr) continue;                   
                AABB box; box.minB = g.bboxMin; box.maxB = g.bboxMax;
                if (!intersectAABB(box, seg.ray, bestT)) continue;

                for (int k = 0; k < g.triCount; ++k) {
                    const Tri& tr = tris[g.triOffset + k];
                    float tHit = intersectTriangleMT(seg.ray, tr);
                    if (tHit > 0.0f && tHit < bestT) {
                        bestT = tHit;
                        bestN = glm::normalize(glm::cross(tr.v1 - tr.v0, tr.v2 - tr.v0));
                        bestGeom = i; bestMat = tr.materialId;
                    }
                }
            }
        }
    }

    if (bestGeom < 0) {
        intersections[path_index].t = -1.0f;
    }
    else {
        if (glm::dot(bestN, seg.ray.direction) > 0.0f) bestN = -bestN;
        intersections[path_index].t = bestT;
        intersections[path_index].surfaceNormal = bestN;
        intersections[path_index].materialId = bestMat;
    }
}


__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    int traceDepth,
    int rrMinDepth,
    bool rrEnabled)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    PathSegment& pathSegment = pathSegments[idx];
    if (pathSegment.remainingBounces <= 0) return;

    const ShadeableIntersection isect = shadeableIntersections[idx];
    if (isect.t <= 0.0f) {
        pathSegment.color = glm::vec3(0.0f);
        pathSegment.remainingBounces = 0;
        return;
    }

    const Material material = materials[isect.materialId];

    // Emissive
    if (material.emittance > 0.0f) {
        if (glm::dot(isect.surfaceNormal, -pathSegment.ray.direction) > 0.0f) {
            pathSegment.color *= (material.color * material.emittance);
        }
        pathSegment.remainingBounces = 0;
        return;
    }

    glm::vec3 p = pathSegment.ray.origin + isect.t * pathSegment.ray.direction;
    glm::vec3 n = glm::normalize(isect.surfaceNormal);
    if (glm::dot(n, -pathSegment.ray.direction) < 0.0f) n = -n;

    thrust::default_random_engine rng =
        makeSeededRandomEngine(iter, pathSegment.pixelIndex, pathSegment.remainingBounces);
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    const int bouncesDone = traceDepth - pathSegment.remainingBounces;

    // Refractive / glass
    if (material.hasRefractive > 0.0f) {
        if (rrEnabled && bouncesDone >= rrMinDepth) {
            float pSurvive = fmaxf(fmaxf(pathSegment.color.x, pathSegment.color.y), pathSegment.color.z);
            pSurvive = fminf(fmaxf(pSurvive, 0.05f), 0.99f);
            if (u01(rng) > pSurvive) {
                pathSegment.color = glm::vec3(0.0f);
                pathSegment.remainingBounces = 0;
                return;
            }
            else {
                pathSegment.color *= (1.0f / pSurvive);
            }
        }

        const glm::vec3 wo = pathSegment.ray.direction;
        const bool entering = (glm::dot(wo, n) < 0.0f);
        const glm::vec3 N = entering ? n : -n;

        float etaI = entering ? 1.0f : material.indexOfRefraction;
        float etaT = entering ? material.indexOfRefraction : 1.0f;
        float eta = etaI / etaT;

        float cosI = fminf(1.0f, fmaxf(0.0f, -glm::dot(wo, N)));
        glm::vec3 idealT = glm::refract(wo, N, eta);
        bool tir = (idealT.x == 0.0f && idealT.y == 0.0f && idealT.z == 0.0f);

        float r0 = (etaI - etaT) / (etaI + etaT); r0 *= r0;
        float R = r0 + (1.0f - r0) * powf(1.0f - cosI, 5.0f);

        // Roughness 
        float rough = fmaxf(0.0f, fminf(material.roughness, 1.0f));

        auto sampleAroundDir = [&](const glm::vec3& dir) -> glm::vec3 {
            if (rough <= 1e-6f) return glm::normalize(dir);
            float alpha = fmaxf(1e-4f, rough);
            float k = fmaxf(0.0f, 1.0f / (alpha * alpha) - 1.0f);
            float u1 = u01(rng);
            float u2 = u01(rng);
            float cosTheta = powf(u1, 1.0f / (k + 1.0f));
            float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
            float phi = 2.0f * PI * u2;

            glm::vec3 d = glm::normalize(dir);
            glm::vec3 t = (fabsf(d.z) < 0.999f)
                ? glm::normalize(glm::cross(glm::vec3(0, 0, 1), d))
                : glm::normalize(glm::cross(glm::vec3(0, 1, 0), d));
            glm::vec3 b = glm::cross(d, t);

            glm::vec3 local(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);
            glm::vec3 world = local.x * t + local.y * b + local.z * d;
            return glm::normalize(world);
            };

        // Continuous reflectivity 
        float reflectiveMix = glm::clamp(material.hasReflective, 0.0f, 1.0f);
        float reflectProb = reflectiveMix * R;

        const float EPS = 2e-3f;
        float xi = u01(rng);

        if (tir || xi < reflectProb) {
            glm::vec3 idealR = glm::reflect(wo, N);
            glm::vec3 rdir = sampleAroundDir(idealR);
            pathSegment.ray.origin = p + N * EPS;
            pathSegment.ray.direction = rdir;
        }
        else {
            glm::vec3 tdir = sampleAroundDir(idealT);
            pathSegment.ray.origin = p - N * EPS;
            pathSegment.ray.direction = tdir;
            pathSegment.color *= glm::clamp(material.color, glm::vec3(0.0f), glm::vec3(1.0f));


        }

        pathSegment.remainingBounces--;
        return;
    }



// Perfect specular 
    if (material.hasReflective > 0.0f && material.hasRefractive <= 0.0f) {
        if (rrEnabled && bouncesDone >= rrMinDepth) {
            float pSurvive = fmaxf(fmaxf(pathSegment.color.x, pathSegment.color.y), pathSegment.color.z);
            pSurvive = fminf(fmaxf(pSurvive, 0.05f), 0.99f);
            if (u01(rng) > pSurvive) {
                pathSegment.color = glm::vec3(0.0f);
                pathSegment.remainingBounces = 0;
                return;
            }
            else {
                pathSegment.color *= (1.0f / pSurvive);
            }
        }

        const glm::vec3 wo = pathSegment.ray.direction;
        const glm::vec3 N = n; 

        glm::vec3 idealR = glm::reflect(wo, N);

        float rough = fmaxf(0.0f, fminf(material.roughness, 1.0f));
        auto sampleAroundDir = [&](const glm::vec3& dir) -> glm::vec3 {
            if (rough <= 1e-6f) return glm::normalize(dir);
            float alpha = fmaxf(1e-4f, rough);
            float k = fmaxf(0.0f, 1.0f / (alpha * alpha) - 1.0f);
            float u1 = u01(rng), u2 = u01(rng);
            float cosTheta = powf(u1, 1.0f / (k + 1.0f));
            float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
            float phi = 2.0f * PI * u2;

            glm::vec3 d = glm::normalize(dir);
            glm::vec3 t = (fabsf(d.z) < 0.999f)
                ? glm::normalize(glm::cross(glm::vec3(0, 0, 1), d))
                : glm::normalize(glm::cross(glm::vec3(0, 1, 0), d));
            glm::vec3 b = glm::cross(d, t);
            glm::vec3 local(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);
            return glm::normalize(local.x * t + local.y * b + local.z * d);
            };

        glm::vec3 rdir = sampleAroundDir(idealR);

        glm::vec3 F = glm::clamp(material.color, glm::vec3(0.0f), glm::vec3(1.0f));
        pathSegment.color *= F;

        const float EPS = 2e-3f;
        pathSegment.ray.origin = p + N * EPS;
        pathSegment.ray.direction = rdir;
        pathSegment.remainingBounces--;
        return;
    }

    // Diffuse 
    {
        glm::vec3 albedo = glm::clamp(material.color, glm::vec3(0.f), glm::vec3(1.f));
        glm::vec3 prospective = pathSegment.color * albedo;

        if (rrEnabled && bouncesDone >= rrMinDepth) {
            float pSurvive = fmaxf(fmaxf(prospective.x, prospective.y), prospective.z);
            pSurvive = fminf(fmaxf(pSurvive, 0.05f), 0.99f);
            if (u01(rng) > pSurvive) {
                pathSegment.color = glm::vec3(0.0f);
                pathSegment.remainingBounces = 0;
                return;
            }
            else {
                pathSegment.color *= (1.0f / pSurvive);
            }
        }

        scatterRay(pathSegment, p, n, material, rng);
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
            dev_intersections,
#if ENABLE_BVH
            (gEnableBVH ? dev_bvhNodes : nullptr),
            (gEnableBVH ? dev_primIndices : nullptr),
#else
            nullptr,
            nullptr,
#endif
            dev_tris,
            (gEnableTriBVH ? dev_triBVHNodes : nullptr),  
            (gEnableTriBVH ? dev_triPrimIdx : nullptr)   
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


        int rrEnabled =
#if ENABLE_RR
        (GetRREnabled() ? 1 : 0);
#else
            0;
#endif

        int rrMinDepth = GetRRMinDepth();
        rrMinDepth = glm::clamp(rrMinDepth, 1, traceDepth - 1);

        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            traceDepth,
            rrMinDepth,
            rrEnabled
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

                if (iter == 1) {
                    printf("Bounce %d: %d -> %d rays (%.1f%% alive)\n",
                        depth, num_paths, newCount,
                        100.0f * newCount / pixelcount);
                }

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
