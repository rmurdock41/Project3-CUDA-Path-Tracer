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
#include <unordered_map>
#include "mesh_loader.h"

#include "tinygltf/stb_image.h"

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
    glm::vec2 uv0, uv1, uv2;
    glm::vec4 tan0, tan1, tan2;
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
    // Non-blocking launch check. Synchronizing here serialized the entire
    // renderer after nearly every kernel and caused periodic GUI stalls.
    // Execution-time errors are still reported by later blocking CUDA calls
    // such as device-to-host copies and final synchronization points.
    cudaError_t err = cudaPeekAtLastError();
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



__device__ __forceinline__ glm::vec3 sampleAccumulatedImage(
    const glm::vec3* image,
    glm::ivec2 resolution,
    int iter,
    float x,
    float y)
{
    int sx = static_cast<int>(floorf(x + 0.5f));
    int sy = static_cast<int>(floorf(y + 0.5f));
    sx = sx < 0 ? 0 : (sx >= resolution.x ? resolution.x - 1 : sx);
    sy = sy < 0 ? 0 : (sy >= resolution.y ? resolution.y - 1 : sy);
    return image[sx + sy * resolution.x] / static_cast<float>(iter);
}

__device__ __forceinline__ glm::vec3 hdrBrightPass(const glm::vec3& color, float threshold)
{
    const float brightness = fmaxf(color.x, fmaxf(color.y, color.z));
    if (brightness <= threshold) {
        return glm::vec3(0.0f);
    }
    const float scale = fminf(fmaxf((brightness - threshold) / fmaxf(brightness, 1.0e-5f), 0.0f), 1.0f);
    return color * scale;
}

__device__ __forceinline__ glm::vec3 acesToneMap(const glm::vec3& color)
{
    const glm::vec3 numerator = color * (2.51f * color + glm::vec3(0.03f));
    const glm::vec3 denominator = color * (2.43f * color + glm::vec3(0.59f)) + glm::vec3(0.14f);
    return glm::clamp(numerator / denominator, glm::vec3(0.0f), glm::vec3(1.0f));
}

__device__ __forceinline__ glm::vec3 gammaCorrect(const glm::vec3& color, float gamma)
{
    const float invGamma = 1.0f / fmaxf(gamma, 0.01f);
    return glm::vec3(
        powf(fminf(fmaxf(color.x, 0.0f), 1.0f), invGamma),
        powf(fminf(fmaxf(color.y, 0.0f), 1.0f), invGamma),
        powf(fminf(fmaxf(color.z, 0.0f), 1.0f), invGamma));
}

__device__ __forceinline__ float postSmoothstep(float edge0, float edge1, float value)
{
    const float range = fmaxf(edge1 - edge0, 1.0e-5f);
    const float t = fminf(fmaxf((value - edge0) / range, 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

__device__ __forceinline__ float godRayConeMask(const glm::vec2& uv, const Camera& camera)
{
    glm::vec2 direction = camera.godRaysTarget - camera.godRaysCenter;
    const float directionLength = glm::length(direction);
    if (directionLength < 1.0e-5f) {
        return 1.0f;
    }
    direction /= directionLength;

    const glm::vec2 relative = uv - camera.godRaysCenter;
    const float along = glm::dot(relative, direction);
    const float maxLength = fmaxf(camera.godRaysLength, 1.0e-4f);
    if (along < 0.0f || along > maxLength) {
        return 0.0f;
    }

    const float progress = fminf(fmaxf(along / maxLength, 0.0f), 1.0f);
    const float halfWidth = fmaxf(camera.godRaysWidth, 1.0e-4f) *
        (0.18f + 0.82f * progress);
    const float perpendicular = fabsf(relative.x * direction.y - relative.y * direction.x);
    const float softness = fminf(fmaxf(camera.godRaysSoftness, 0.01f), 0.99f);
    const float sideMask = 1.0f - postSmoothstep(
        halfWidth * (1.0f - softness), halfWidth, perpendicular);
    const float endMask = 1.0f - postSmoothstep(maxLength * 0.82f, maxLength, along);
    return sideMask * endMask;
}

__device__ __forceinline__ float godRayHazeMask(const glm::vec2& uv, const Camera& camera)
{
    const glm::vec2 radius = glm::max(camera.godRaysHazeRadius, glm::vec2(1.0e-4f));
    const glm::vec2 offset = (uv - camera.godRaysHazeCenter) / radius;
    const float distance = glm::length(offset);
    const float falloff = 1.0f - postSmoothstep(0.05f, 1.0f, distance);
    return falloff * falloff * (3.0f - 2.0f * falloff);
}

__device__ __forceinline__ void godRaySubjectMasks(
    const glm::vec2& uv,
    const Camera& camera,
    float& subjectMask,
    float& rimMask)
{
    const glm::vec2 radius = glm::max(
        camera.godRaysHazeSubjectRadius, glm::vec2(1.0e-4f));
    const glm::vec2 offset = (uv - camera.godRaysHazeSubjectCenter) / radius;
    const float distance = glm::length(offset);
    const float rimWidth = fminf(fmaxf(camera.godRaysHazeRimWidth, 0.02f), 0.45f);

    subjectMask = 1.0f - postSmoothstep(1.0f - rimWidth, 1.0f, distance);
    const float outerMask = 1.0f - postSmoothstep(1.0f, 1.0f + rimWidth, distance);
    const float innerMask = 1.0f - postSmoothstep(1.0f - rimWidth, 1.0f, distance);
    rimMask = fminf(fmaxf(outerMask - innerMask, 0.0f), 1.0f);
}

// Kernel that writes the image to the OpenGL PBO directly. Post-processing is
// evaluated from the HDR accumulation buffer so the preview matches saved PNGs.
__global__ void sendImageToPBO(
    uchar4* pbo,
    glm::ivec2 resolution,
    int iter,
    glm::vec3* image,
    Camera camera)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index] / static_cast<float>(iter);

        if (camera.postEnabled)
        {
            const float exposure = fmaxf(camera.exposure, 0.0f);
            glm::vec3 combined = glm::max(pix * exposure, glm::vec3(0.0f));

            if (camera.bloomEnabled && camera.bloomStrength > 0.0f && camera.bloomRadius > 0.0f)
            {
                const int sampleCount = camera.bloomSamples < 1 ? 1 :
                    (camera.bloomSamples > 64 ? 64 : camera.bloomSamples);
                const float goldenAngle = 2.39996323f;
                glm::vec3 bloom(0.0f);
                float totalWeight = 0.0f;

                for (int i = 0; i < sampleCount; ++i)
                {
                    const float t = (static_cast<float>(i) + 0.5f) / static_cast<float>(sampleCount);
                    const float radius = camera.bloomRadius * sqrtf(t);
                    const float angle = goldenAngle * static_cast<float>(i);
                    const float weight = 1.0f - 0.75f * t;
                    const glm::vec3 sample = sampleAccumulatedImage(
                        image,
                        resolution,
                        iter,
                        static_cast<float>(x) + cosf(angle) * radius,
                        static_cast<float>(y) + sinf(angle) * radius) * exposure;
                    bloom += hdrBrightPass(sample, camera.bloomThreshold) * weight;
                    totalWeight += weight;
                }

                combined += bloom * (camera.bloomStrength / fmaxf(totalWeight, 1.0e-5f));
            }

            if (camera.godRaysEnabled)
            {
                const glm::vec2 pixelUv(
                    (static_cast<float>(x) + 0.5f) / static_cast<float>(resolution.x),
                    (static_cast<float>(y) + 0.5f) / static_cast<float>(resolution.y));
                if (camera.godRaysStrength > 0.0f && camera.godRaysWeight > 0.0f)
                {
                const int sampleCount = camera.godRaysSamples < 1 ? 1 :
                    (camera.godRaysSamples > 96 ? 96 : camera.godRaysSamples);
                glm::vec2 sampleUv = pixelUv;
                glm::vec2 rayStep(0.0f);
                if (camera.godRaysDirectionalEnabled)
                {
                    const float directionLength = glm::length(camera.godRaysDirection);
                    if (directionLength > 1.0e-5f)
                    {
                        rayStep = -(camera.godRaysDirection / directionLength) *
                            (camera.godRaysDensity / static_cast<float>(sampleCount));
                    }
                }
                else if (camera.godRaysConvergeEnabled)
                {
                    const glm::vec2 awayFromTarget = sampleUv - camera.godRaysTarget;
                    const float distanceFromTarget = glm::length(awayFromTarget);
                    if (distanceFromTarget > 1.0e-5f)
                    {
                        rayStep = (awayFromTarget / distanceFromTarget) *
                            (camera.godRaysDensity / static_cast<float>(sampleCount));
                    }
                }
                else
                {
                    rayStep = -((sampleUv - camera.godRaysCenter) *
                        (camera.godRaysDensity / static_cast<float>(sampleCount)));
                }
                glm::vec3 rays(0.0f);
                float illuminationDecay = 1.0f;

                for (int i = 0; i < sampleCount; ++i)
                {
                    sampleUv += rayStep;
                    const glm::vec3 sample = sampleAccumulatedImage(
                        image,
                        resolution,
                        iter,
                        sampleUv.x * static_cast<float>(resolution.x) - 0.5f,
                        sampleUv.y * static_cast<float>(resolution.y) - 0.5f) * exposure;
                    float sourceWeight = 1.0f;
                    if (camera.godRaysConvergeEnabled || camera.godRaysDirectionalEnabled)
                    {
                        const float vertical = postSmoothstep(
                            camera.godRaysVerticalStart,
                            camera.godRaysVerticalEnd,
                            sampleUv.y);
                        sourceWeight = camera.godRaysVerticalMin +
                            (1.0f - camera.godRaysVerticalMin) * vertical;
                    }
                    rays += hdrBrightPass(sample, camera.godRaysThreshold) *
                        (illuminationDecay * camera.godRaysWeight * sourceWeight);
                    illuminationDecay *= camera.godRaysDecay;
                }

                const float focusMask = camera.godRaysFocusEnabled ?
                    godRayConeMask(pixelUv, camera) : 1.0f;
                const float convergenceEndMask =
                    (camera.godRaysConvergeEnabled || camera.godRaysDirectionalEnabled) ?
                    (1.0f - postSmoothstep(
                        camera.godRaysEndY - 0.03f,
                        camera.godRaysEndY,
                        pixelUv.y)) : 1.0f;
                combined += rays *
                    (camera.godRaysStrength * focusMask * convergenceEndMask);
                }
                const float hazeMask = godRayHazeMask(pixelUv, camera);
                float subjectMask = 0.0f;
                float rimMask = 0.0f;
                godRaySubjectMasks(pixelUv, camera, subjectMask, rimMask);
                const float subjectVisibility = 1.0f -
                    fminf(fmaxf(camera.godRaysHazeSubjectProtect, 0.0f), 1.0f) * subjectMask;
                combined += camera.godRaysHazeColor *
                    (camera.godRaysHazeStrength * hazeMask * subjectVisibility);
                combined += camera.godRaysHazeColor *
                    (camera.godRaysHazeRimStrength * hazeMask * rimMask);
            }

            pix = gammaCorrect(acesToneMap(combined), camera.gamma);
        }

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x * 255.0f), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0f), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0f), 0, 255);

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
static FogCard* dev_fogCards = nullptr;
static int g_numFogCards = 0;

// ===== Environment Map =====
static cudaTextureObject_t gEnvTexObj = 0;
static cudaArray_t         gEnvCuArray = nullptr;
static float               gEnvIntensity = 1.0f;
static float               gEnvRotation = 0.0f;
static bool                gHasEnvMap = false;

float GetEnvIntensity() { return gEnvIntensity; }
void  SetEnvIntensity(float v) { gEnvIntensity = v; }
float GetEnvRotation() { return gEnvRotation; }
void  SetEnvRotation(float r) { gEnvRotation = r; }
bool  HasEnvMap() { return gHasEnvMap; }

void ClearEnvMap() {
    if (gEnvTexObj) { cudaDestroyTextureObject(gEnvTexObj);  gEnvTexObj = 0; }
    if (gEnvCuArray) { cudaFreeArray(gEnvCuArray); gEnvCuArray = nullptr; }
    gHasEnvMap = false;
}

static void uploadEnvMap(float* data, int w, int h) {
    std::vector<float4> data4(w * h);
    for (int i = 0; i < w * h; i++)
        data4[i] = make_float4(data[i * 3], data[i * 3 + 1], data[i * 3 + 2], 0.f);

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    cudaMallocArray(&gEnvCuArray, &desc, w, h);
    cudaMemcpy2DToArray(
        gEnvCuArray, 0, 0,
        data4.data(), w * sizeof(float4),
        w * sizeof(float4), h,
        cudaMemcpyHostToDevice);

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = gEnvCuArray;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaCreateTextureObject(&gEnvTexObj, &resDesc, &texDesc, nullptr);
}

void SetEnvMap(const char* hdrPath) {


    int w, h, c;
    float* data = stbi_loadf(hdrPath, &w, &h, &c, 3);
    if (!data) {
        printf("[EnvMap] failed : %s\n", hdrPath);
        return;
    }
    ClearEnvMap();
    uploadEnvMap(data, w, h);
    stbi_image_free(data);
    gHasEnvMap = true;
    printf("[EnvMap] success : %s (%dx%d)\n", hdrPath, w, h);
}



// ===== Texture system =====
struct GpuTexture {
    cudaTextureObject_t texObj = 0;
    cudaArray_t         cuArray = nullptr;
};
static std::vector<GpuTexture> gTextures;
static cudaTextureObject_t* dev_textures = nullptr;

void FreeAllTextures() {
    for (auto& t : gTextures) {
        if (t.texObj)  cudaDestroyTextureObject(t.texObj);
        if (t.cuArray) cudaFreeArray(t.cuArray);
    }
    gTextures.clear();
    cudaFree(dev_textures);
    dev_textures = nullptr;
}

int UploadTexture(const float* pixels, int w, int h) {
    if (!pixels || w <= 0 || h <= 0) return -1;

    GpuTexture tex;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    cudaMallocArray(&tex.cuArray, &desc, w, h);

	// pixels is expected to be in float3 RGB format, we convert it to float4 RGBA with A=0 for CUDA texture
    cudaMemcpy2DToArray(
        tex.cuArray, 0, 0,
        pixels, w * sizeof(float4),
        w * sizeof(float4), h,
        cudaMemcpyHostToDevice);

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = tex.cuArray;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaCreateTextureObject(&tex.texObj, &resDesc, &texDesc, nullptr);

    int id = (int)gTextures.size();
    gTextures.push_back(tex);
    return id;
}

static int UploadTextureFileRGBA(const std::string& path) {
    int w = 0;
    int h = 0;
    int sourceChannels = 0;
    unsigned char* bytes = stbi_load(
        path.c_str(), &w, &h, &sourceChannels, STBI_rgb_alpha);
    if (bytes == nullptr || w <= 0 || h <= 0) {
        printf("[FogCard] failed to load texture: %s\n", path.c_str());
        if (bytes != nullptr) stbi_image_free(bytes);
        return -1;
    }

    std::vector<float> rgba((size_t)w * (size_t)h * 4u);
    for (size_t i = 0; i < (size_t)w * (size_t)h * 4u; ++i) {
        rgba[i] = (float)bytes[i] / 255.0f;
    }
    stbi_image_free(bytes);

    const int textureId = UploadTexture(rgba.data(), w, h);
    printf("[FogCard] texture: %s (%dx%d, id=%d)\n",
        path.c_str(), w, h, textureId);
    return textureId;
}

static void UploadFogCards(Scene* scene) {
    cudaFree(dev_fogCards);
    dev_fogCards = nullptr;
    g_numFogCards = 0;

    std::vector<FogCard> cards;
    cards.reserve(scene->fogCards.size());
    std::unordered_map<std::string, int> textureCache;
    for (const FogCardConfig& cfg : scene->fogCards) {
        FogCard card = cfg.card;
        const auto cached = textureCache.find(cfg.texturePath);
        if (cached != textureCache.end()) {
            card.textureId = cached->second;
        }
        else {
            card.textureId = UploadTextureFileRGBA(cfg.texturePath);
            textureCache[cfg.texturePath] = card.textureId;
        }
        if (card.textureId >= 0) cards.push_back(card);
    }

    g_numFogCards = (int)cards.size();
    if (g_numFogCards > 0) {
        cudaMalloc(&dev_fogCards, g_numFogCards * sizeof(FogCard));
        cudaMemcpy(dev_fogCards, cards.data(),
            g_numFogCards * sizeof(FogCard), cudaMemcpyHostToDevice);
    }
    printf("[FogCard] uploaded %d card(s)\n", g_numFogCards);
}

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




__device__ __forceinline__ float intersectTriangleMT(
    const Ray& r, const Tri& tr,
    float& outU, float& outV)   
{
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
    outU = u;   
    outV = v;  
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
    float& tHit, glm::vec3& nHit, int& matId,
    glm::vec2& uvHit, glm::vec4& tanHit)
{
    const Geom& g = geoms[primId];
    float t = -1.0f; glm::vec3 pTmp, nTmp; bool outside = true;

    if (g.type == CUBE) {
        t = boxIntersectionTest(g, ray, pTmp, nTmp, outside);
        if (t > 0.0f && t < tHit) { tHit = t; nHit = nTmp; matId = g.materialid; uvHit = glm::vec2(0.0f); tanHit = glm::vec4(1, 0, 0, 1); return true; }
        return false;
    }
    else if (g.type == SPHERE) {
        t = sphereIntersectionTest(g, ray, pTmp, nTmp, outside);
        if (t > 0.0f && t < tHit) { tHit = t; nHit = nTmp; matId = g.materialid; uvHit = glm::vec2(0.0f); tanHit = glm::vec4(1, 0, 0, 1); return true; }
        return false;
    }
    else if (g.type == MESH) {
        AABB mbox; mbox.minB = g.bboxMin; mbox.maxB = g.bboxMax;
        if (!intersectAABB(mbox, ray, tHit)) return false;

        if (triNodes == nullptr || triPrimIdx == nullptr || g.triBVHRoot < 0) {
            bool any = false;
            for (int k = 0; k < g.triCount; ++k) {
                const Tri& tr = tris[g.triOffset + k];
                float bu, bv;
                float th = intersectTriangleMT(ray, tr, bu, bv);
                if (th > 0.0f && th < tHit) {
                    tHit = th;
                    nHit = glm::normalize(glm::cross(tr.v1 - tr.v0, tr.v2 - tr.v0));
                    matId = tr.materialId;
                    float bw = 1.0f - bu - bv;
                    uvHit = bw * tr.uv0 + bu * tr.uv1 + bv * tr.uv2;
                    tanHit = bw * tr.tan0 + bu * tr.tan1 + bv * tr.tan2;  
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

                    float bu, bv;
                    float th = intersectTriangleMT(ray, tr, bu, bv);
                    if (th > 0.0f && th < tHit) {
                        tHit = th;
                        nHit = glm::normalize(glm::cross(tr.v1 - tr.v0, tr.v2 - tr.v0));
                        matId = tr.materialId;
                        float bw = 1.0f - bu - bv;
                        uvHit = bw * tr.uv0 + bu * tr.uv1 + bv * tr.uv2;
                        tanHit = bw * tr.tan0 + bu * tr.tan1 + bv * tr.tan2;  
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
    float& outT, int& outGeom, glm::vec3& outN, int& outMat,
    glm::vec2& outUV, glm::vec4& outTan)
{
    int stack[64]; int sp = 0;
    float tClosest = FLT_MAX; int hitGeom = -1; glm::vec3 nHit(0.f); int mId = -1;
    glm::vec2 uvClosest(0.f);
    glm::vec4 tanClosest(1, 0, 0, 1);  

    stack[sp++] = 0;
    while (sp) {
        const int ni = stack[--sp];
        const BVHNode& node = nodes[ni];
        if (!intersectAABB(node.box, ray, tClosest)) continue;

        if (node.primCount > 0) {
            for (int i = 0; i < node.primCount; ++i) {
                const int pid = primIdx[node.firstPrim + i];
                float t = tClosest; glm::vec3 n; int mat;
                glm::vec2 uv(0.f);  
                glm::vec4 tan(1, 0, 0, 1);
                if (intersectPrimitiveGeom(geoms, pid, ray, tris, triNodes, triPrimIdx, t, n, mat, uv, tan)) {
                    if (t < tClosest) { tClosest = t; hitGeom = pid; nHit = n; mId = mat; uvClosest = uv; }  
                }
            }
        }
        else {
            if (node.left >= 0) stack[sp++] = node.left;
            if (node.right >= 0) stack[sp++] = node.right;
        }
    }
    outT = tClosest; outGeom = hitGeom; outN = nHit; outMat = mId; outUV = uvClosest; outTan = tanClosest;  
}

__device__ inline bool traverseTriBVH(
    const Ray& ray,
    const TriBVHNode* __restrict__ nodes,
    const int* __restrict__ primIdx,
    int root,
    const Tri* __restrict__ tris,
    float& tHit, glm::vec3& nHit, int& matId,
    glm::vec2& uvHit, glm::vec4& tanHit)
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
                float bu, bv;
                float th = intersectTriangleMT(ray, tr, bu, bv);
                if (th > 0.f && th < tHit) {
                    tHit = th;
                    nHit = glm::normalize(glm::cross(tr.v1 - tr.v0, tr.v2 - tr.v0));
                    matId = tr.materialId;
                    float bw = 1.0f - bu - bv;
                    uvHit = bw * tr.uv0 + bu * tr.uv1 + bv * tr.uv2;
                    tanHit = bw * tr.tan0 + bu * tr.tan1 + bv * tr.tan2;  

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
        temp[i].uv0 = h_allTris[i].uv0; 
        temp[i].uv1 = h_allTris[i].uv1; 
        temp[i].uv2 = h_allTris[i].uv2;
        temp[i].tan0 = h_allTris[i].tan0;  
        temp[i].tan1 = h_allTris[i].tan1;  
        temp[i].tan2 = h_allTris[i].tan2;
        temp[i].materialId = h_allTris[i].materialId;
    }
    cudaMemcpy(dev_tris, temp.data(), g_numTris * sizeof(Tri), cudaMemcpyHostToDevice);
}


static void BakeMeshesIntoSceneAndCPUTris(Scene* scene) {
    h_allTris.clear();


    scene->geoms.erase(
        std::remove_if(scene->geoms.begin(), scene->geoms.end(),
            [](const Geom& g) { return g.type == MESH; }),
        scene->geoms.end()
    );

    for (const auto& mi : scene->meshInstances) {
        std::vector<TriCPU> local;
        std::string err;
        GltfMaterialTextures gltfTex;
        if (!LoadGLTF_AsTris(mi.path, mi.M_world, mi.materialId, local, &err, &gltfTex)) {
            printf("[GLTF] load failed: %s\n", err.c_str());
            continue;
        }

        Material& mat = scene->materials[mi.materialId];
        if (gltfTex.albedo.w > 0) {
            mat.albedoTexId = UploadTexture(
                gltfTex.albedo.pixels.data(),
                gltfTex.albedo.w, gltfTex.albedo.h);
        }
        if (gltfTex.metallicRoughness.w > 0) {
            mat.metallicRoughnessTexId = UploadTexture(
                gltfTex.metallicRoughness.pixels.data(),
                gltfTex.metallicRoughness.w, gltfTex.metallicRoughness.h);
        }
        if (gltfTex.normal.w > 0) {
            mat.normalTexId = UploadTexture(
                gltfTex.normal.pixels.data(),
                gltfTex.normal.w, gltfTex.normal.h);
        }
        if (gltfTex.emissive.w > 0) {
            mat.emissiveTexId = UploadTexture(
                gltfTex.emissive.pixels.data(),
                gltfTex.emissive.w, gltfTex.emissive.h);
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
    UploadFogCards(scene);


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




    if (!gTextures.empty()) {
        std::vector<cudaTextureObject_t> texObjs;
        for (auto& t : gTextures) texObjs.push_back(t.texObj);
        cudaFree(dev_textures);
        cudaMalloc(&dev_textures, texObjs.size() * sizeof(cudaTextureObject_t));
        cudaMemcpy(dev_textures, texObjs.data(),
            texObjs.size() * sizeof(cudaTextureObject_t),
            cudaMemcpyHostToDevice);
    }

    cudaFree(dev_materials);
    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(),
        scene->materials.size() * sizeof(Material),
        cudaMemcpyHostToDevice);


    static bool firstLoad = true;
    if (firstLoad) {
        SetEnvMap("scenes/default.hdr");
        firstLoad = false;
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_fogCards); dev_fogCards = nullptr; g_numFogCards = 0;

    cudaFree(dev_tris); dev_tris = nullptr; g_numTris = 0;

    FreeAllTextures();

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
        segment.fogCardsProcessed = 0;
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
    const Material* materials,
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
    Ray queryRay = seg.ray;
    float skippedDistance = 0.0f;
    float     bestT = FLT_MAX;
    glm::vec3 bestN = glm::vec3(0.f);
    glm::vec2 bestUV = glm::vec2(0.f);
    glm::vec4 bestTan = glm::vec4(1, 0, 0, 1);  
    int       bestGeom = -1;
    int       bestMat = -1;

    // Resolve camera-invisible geometry inside this intersection query.  The
    // old shader-side pass-through consumed one global tracing iteration per
    // face, so pixels covered by a hidden light card received fewer real
    // bounces and exposed the card as a rectangular color discontinuity.
    // Skipping it here keeps the primary-ray bounce budget uniform.
    const int maxInvisibleSkips = 16;
    for (int skip = 0; skip < maxInvisibleSkips; ++skip) {
        bestT = FLT_MAX;
        bestN = glm::vec3(0.f);
        bestUV = glm::vec2(0.f);
        bestTan = glm::vec4(1, 0, 0, 1);
        bestGeom = -1;
        bestMat = -1;

#if ENABLE_BVH
        if (bvhNodes != nullptr && primIdx != nullptr) {
            traverseBVH(queryRay, bvhNodes, primIdx, geoms, tris,
                triNodes, triPrimIdx,
                bestT, bestGeom, bestN, bestMat, bestUV, bestTan);
        }
        else
#endif
        {
            bool outside = true; glm::vec3 pTmp, nTmp;
            for (int i = 0; i < geoms_size; ++i) {
                const Geom& g = geoms[i];

                if (g.type == CUBE) {
                    float t = boxIntersectionTest(g, queryRay, pTmp, nTmp, outside);
                    if (t > 0.0f && t < bestT) {
                        bestT = t; bestN = nTmp; bestGeom = i; bestMat = g.materialid;
                        bestUV = glm::vec2(0.f);
                        bestTan = glm::vec4(1, 0, 0, 1);
                    }
                }
                else if (g.type == SPHERE) {
                    float t = sphereIntersectionTest(g, queryRay, pTmp, nTmp, outside);
                    if (t > 0.0f && t < bestT) {
                        bestT = t; bestN = nTmp; bestGeom = i; bestMat = g.materialid;
                        bestUV = glm::vec2(0.f);
                        bestTan = glm::vec4(1, 0, 0, 1);
                    }
                }
                else if (g.type == MESH) {
                    if (tris == nullptr) continue;
                    AABB box; box.minB = g.bboxMin; box.maxB = g.bboxMax;
                    if (!intersectAABB(box, queryRay, bestT)) continue;

                    for (int k = 0; k < g.triCount; ++k) {
                        const Tri& tr = tris[g.triOffset + k];
                        float bu, bv;
                        float tHit = intersectTriangleMT(queryRay, tr, bu, bv);
                        if (tHit > 0.0f && tHit < bestT) {
                            bestT = tHit;
                            bestN = glm::normalize(glm::cross(tr.v1 - tr.v0, tr.v2 - tr.v0));
                            bestGeom = i; bestMat = tr.materialId;
                            float bw = 1.0f - bu - bv;
                            bestUV = bw * tr.uv0 + bu * tr.uv1 + bv * tr.uv2;
                            bestTan = bw * tr.tan0 + bu * tr.tan1 + bv * tr.tan2;
                        }
                    }
                }
            }
        }

        if (bestGeom < 0) break;

        if (depth == 0 && materials != nullptr && bestMat >= 0 &&
            materials[bestMat].cameraVisible == 0) {
            const float advance = bestT + 2.0e-3f;
            skippedDistance += advance;
            queryRay.origin += queryRay.direction * advance;
            bestGeom = -1;
            bestMat = -1;
            continue;
        }

        bestT += skippedDistance;
        break;
    }

    if (bestGeom < 0) {
        intersections[path_index].t = -1.0f;
    }
    else {
        if (glm::dot(bestN, seg.ray.direction) > 0.0f) bestN = -bestN;
        intersections[path_index].t = bestT;
        intersections[path_index].surfaceNormal = bestN;
        intersections[path_index].materialId = bestMat;
        intersections[path_index].uv = bestUV;
        intersections[path_index].tangent = bestTan;  
    }
}


__device__ __forceinline__ glm::vec3 sampleEnvMap(
    cudaTextureObject_t envTex,
    const glm::vec3& dir,
    float rotation,
    float intensity)
{
    float phi = atan2f(dir.z, dir.x);
    float theta = acosf(glm::clamp(dir.y, -1.f, 1.f));

    float u = (phi + PI + rotation) / (2.f * PI);
    u = u - floorf(u);
    float v = theta / PI;

    float4 c = tex2D<float4>(envTex, u, v);
    return glm::vec3(c.x, c.y, c.z) * intensity;
}


__device__ __forceinline__ glm::vec4 sampleTex(
    const cudaTextureObject_t* textures,
    int texId, glm::vec2 uv)
{
    float4 c = tex2D<float4>(textures[texId], uv.x, uv.y);
    return glm::vec4(c.x, c.y, c.z, c.w);
}

// Fog cards are not regular scene geometry. Intersect and alpha-compose all
// cards once along a primary ray, in front-to-back order and only up to the
// first opaque surface. This avoids additional BVH traversals per card.
__device__ __forceinline__ void compositeFogCards(
    const Ray& ray,
    float surfaceT,
    const FogCard* fogCards,
    int numFogCards,
    const cudaTextureObject_t* textures,
    int numTextures,
    glm::vec3& fogRadiance,
    float& transmittance)
{
    const int MAX_FOG_CARD_HITS = 32;
    float hitT[MAX_FOG_CARD_HITS];
    int hitCard[MAX_FOG_CARD_HITS];
    glm::vec2 hitUV[MAX_FOG_CARD_HITS];
    int hitCount = 0;

    const int cardCount = min(numFogCards, MAX_FOG_CARD_HITS);
    for (int i = 0; i < cardCount; ++i) {
        const FogCard card = fogCards[i];
        const glm::vec3 normal = glm::cross(card.right, card.up);
        const float normalLength2 = glm::dot(normal, normal);
        if (normalLength2 <= 1e-10f) continue;

        const float denom = glm::dot(ray.direction, normal);
        if (fabsf(denom) <= 1e-7f) continue;
        const float t = glm::dot(card.center - ray.origin, normal) / denom;
        if (t <= 1e-4f || t >= surfaceT) continue;

        const glm::vec3 local = ray.origin + t * ray.direction - card.center;
        const float rightLength2 = glm::dot(card.right, card.right);
        const float upLength2 = glm::dot(card.up, card.up);
        const float localX = glm::dot(local, card.right) / rightLength2;
        const float localY = glm::dot(local, card.up) / upLength2;
        if (fabsf(localX) > 1.0f || fabsf(localY) > 1.0f) continue;

        const glm::vec2 uv(
            localX * 0.5f + 0.5f,
            0.5f - localY * 0.5f);

        // Insertion sort while collecting hits; card counts are intentionally
        // small, making this cheaper than launching or traversing another BVH.
        int insert = hitCount;
        while (insert > 0 && hitT[insert - 1] > t) {
            hitT[insert] = hitT[insert - 1];
            hitCard[insert] = hitCard[insert - 1];
            hitUV[insert] = hitUV[insert - 1];
            --insert;
        }
        hitT[insert] = t;
        hitCard[insert] = i;
        hitUV[insert] = uv;
        ++hitCount;
    }

    fogRadiance = glm::vec3(0.0f);
    transmittance = 1.0f;
    for (int i = 0; i < hitCount; ++i) {
        const FogCard card = fogCards[hitCard[i]];
        if (card.textureId < 0 || card.textureId >= numTextures) continue;
        const float4 sample = tex2D<float4>(
            textures[card.textureId], hitUV[i].x, hitUV[i].y);
        const float edgeDistance = fminf(
            fminf(hitUV[i].x, 1.0f - hitUV[i].x),
            fminf(hitUV[i].y, 1.0f - hitUV[i].y));
        float edgeMask = 1.0f;
        if (card.edgeFade > 1e-5f) {
            const float edgeT = glm::clamp(
                edgeDistance / card.edgeFade, 0.0f, 1.0f);
            edgeMask = edgeT * edgeT * (3.0f - 2.0f * edgeT);
        }
        float depthMask = 1.0f;
        if (card.depthFade > 1e-5f && surfaceT < FLT_MAX * 0.5f) {
            const float separation = surfaceT - hitT[i];
            const float fade = glm::clamp(separation / card.depthFade, 0.0f, 1.0f);
            depthMask = fade * fade * (3.0f - 2.0f * fade);
        }
        const float alpha = glm::clamp(
            sample.w * card.opacity * depthMask * edgeMask, 0.0f, 0.98f);
        if (alpha <= 1e-4f) continue;

        fogRadiance += transmittance * alpha * card.color;
        transmittance *= (1.0f - alpha);
        if (transmittance <= 1e-3f) break;
    }
}

// Return the portion of a ray segment that lies inside the configured medium
// box. Directions are normalized throughout this renderer, so t is a world
// space distance here.
__device__ __forceinline__ bool volumeRayInterval(
    const Ray& ray,
    const glm::vec3& boundsMin,
    const glm::vec3& boundsMax,
    float segmentMax,
    float& tEnter,
    float& tExit)
{
    const float eps = 1e-8f;
    glm::vec3 invD(
        1.0f / ((fabsf(ray.direction.x) > eps) ? ray.direction.x : copysignf(eps, ray.direction.x)),
        1.0f / ((fabsf(ray.direction.y) > eps) ? ray.direction.y : copysignf(eps, ray.direction.y)),
        1.0f / ((fabsf(ray.direction.z) > eps) ? ray.direction.z : copysignf(eps, ray.direction.z)));
    glm::vec3 a = (boundsMin - ray.origin) * invD;
    glm::vec3 b = (boundsMax - ray.origin) * invD;
    glm::vec3 near3 = glm::min(a, b);
    glm::vec3 far3 = glm::max(a, b);

    tEnter = fmaxf(0.0f, fmaxf(near3.x, fmaxf(near3.y, near3.z)));
    tExit = fminf(segmentMax, fminf(far3.x, fminf(far3.y, far3.z)));
    return tExit > tEnter;
}

__device__ __forceinline__ float henyeyGreensteinPhase(float cosTheta, float g)
{
    const float gg = g * g;
    const float denom = fmaxf(1e-6f, 1.0f + gg - 2.0f * g * cosTheta);
    return (1.0f - gg) / (4.0f * PI * denom * sqrtf(denom));
}

__device__ __forceinline__ glm::vec3 sampleHenyeyGreenstein(
    const glm::vec3& forward,
    float g,
    float u1,
    float u2)
{
    float cosTheta;
    if (fabsf(g) < 1e-3f) {
        cosTheta = 1.0f - 2.0f * u1;
    }
    else {
        const float ratio = (1.0f - g * g) / (1.0f - g + 2.0f * g * u1);
        cosTheta = (1.0f + g * g - ratio * ratio) / (2.0f * g);
        cosTheta = glm::clamp(cosTheta, -1.0f, 1.0f);
    }

    const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    const float phi = 2.0f * PI * u2;
    const glm::vec3 w = glm::normalize(forward);
    const glm::vec3 t = (fabsf(w.z) < 0.999f)
        ? glm::normalize(glm::cross(glm::vec3(0, 0, 1), w))
        : glm::normalize(glm::cross(glm::vec3(0, 1, 0), w));
    const glm::vec3 b = glm::cross(w, t);
    return glm::normalize(
        t * (cosf(phi) * sinTheta) +
        b * (sinf(phi) * sinTheta) +
        w * cosTheta);
}

__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    int traceDepth,
    int rrMinDepth,
    bool rrEnabled,
    cudaTextureObject_t envTex,
    float envIntensity,
    float envRotation,
    bool hasEnvMap,
    const cudaTextureObject_t* textures, 
    int numTextures,
    glm::vec3* image,
    const FogCard* fogCards,
    int numFogCards,
    Camera camera,
    Geom* geoms,
    const BVHNode* bvhNodes,
    const int* primIdx,
    const Tri* tris,
    const TriBVHNode* triNodes,
    const int* triPrimIdx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    PathSegment& pathSegment = pathSegments[idx];
    if (pathSegment.remainingBounces <= 0) return;

    const ShadeableIntersection isect = shadeableIntersections[idx];
    const int bouncesDone = traceDepth - pathSegment.remainingBounces;
    thrust::default_random_engine rng =
        makeSeededRandomEngine(iter, pathSegment.pixelIndex, pathSegment.remainingBounces);
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    if (bouncesDone == 0 && pathSegment.fogCardsProcessed == 0) {
        pathSegment.fogCardsProcessed = 1;
        if (fogCards != nullptr && numFogCards > 0 &&
            textures != nullptr && image != nullptr) {
            const float surfaceT = (isect.t > 0.0f) ? isect.t : FLT_MAX;
            glm::vec3 fogRadiance(0.0f);
            float fogTransmittance = 1.0f;
            compositeFogCards(
                pathSegment.ray,
                surfaceT,
                fogCards,
                numFogCards,
                textures,
                numTextures,
                fogRadiance,
                fogTransmittance);
            image[pathSegment.pixelIndex] += pathSegment.color * fogRadiance;
            pathSegment.color *= fogTransmittance;
        }
    }

    // Analog free-flight sampling in a bounded homogeneous medium. A surface
    // behind the box is reached only by samples that survive the exponential
    // extinction probability, so transmittance must not be multiplied again.
    const float sigmaT = camera.volumeSigmaA + camera.volumeSigmaS;
    if (camera.volumeEnabled && sigmaT > 0.0f) {
        const float surfaceT = (isect.t > 0.0f) ? isect.t : FLT_MAX;
        float mediumEnter, mediumExit;
        if (volumeRayInterval(
            pathSegment.ray,
            camera.volumeMin,
            camera.volumeMax,
            surfaceT,
            mediumEnter,
            mediumExit)) {
            const float xi = fmaxf(1e-7f, 1.0f - u01(rng));
            const float freeFlight = -logf(xi) / sigmaT;
            const float mediumLength = mediumExit - mediumEnter;

            if (freeFlight < mediumLength) {
                const float scatterT = mediumEnter + freeFlight;
                const glm::vec3 scatterPoint =
                    pathSegment.ray.origin + scatterT * pathSegment.ray.direction;

                // The exponential distance PDF cancels sigma_t * Tr. What is
                // left in path throughput is the single-scattering albedo.
                const float scalarAlbedo = camera.volumeSigmaS / sigmaT;
                pathSegment.color *= scalarAlbedo * camera.volumeScatterColor;

                // One-sample NEE toward a large analytic rectangle behind the
                // clock. Geometry visibility makes all clock openings work as
                // a distributed source instead of one dominant bright hole.
                if (camera.volumeLightEnabled && image != nullptr) {
                    const float su = 2.0f * u01(rng) - 1.0f;
                    const float sv = 2.0f * u01(rng) - 1.0f;
                    const glm::vec3 lightPoint = camera.volumeLightCenter +
                        su * camera.volumeLightU + sv * camera.volumeLightV;
                    glm::vec3 toLight = lightPoint - scatterPoint;
                    const float dist2 = glm::dot(toLight, toLight);

                    if (dist2 > 1e-6f) {
                        const float distToLight = sqrtf(dist2);
                        const glm::vec3 wi = toLight / distToLight;
                        const glm::vec3 lightCross = glm::cross(
                            camera.volumeLightU, camera.volumeLightV);
                        const float quarterArea = glm::length(lightCross);
                        const float lightArea = 4.0f * quarterArea;
                        const glm::vec3 lightNormal = (quarterArea > 1e-8f)
                            ? lightCross / quarterArea
                            : glm::vec3(0.0f, 0.0f, 1.0f);
                        const float cosAtLight = fabsf(glm::dot(lightNormal, -wi));

                        bool occluded = false;
                        if (lightArea > 1e-6f && cosAtLight > 1e-5f &&
                            bvhNodes != nullptr && primIdx != nullptr) {
                            Ray shadowRay;
                            shadowRay.origin = scatterPoint + wi * 2e-3f;
                            shadowRay.direction = wi;
                            float shadowT = distToLight - 4e-3f;
                            int shadowGeom = -1;
                            int shadowMat = -1;
                            glm::vec3 shadowN(0.0f);
                            glm::vec2 shadowUV(0.0f);
                            glm::vec4 shadowTan(1, 0, 0, 1);
                            traverseBVH(
                                shadowRay, bvhNodes, primIdx, geoms, tris,
                                triNodes, triPrimIdx,
                                shadowT, shadowGeom, shadowN, shadowMat,
                                shadowUV, shadowTan);
                            occluded = shadowGeom >= 0 && shadowT < distToLight - 4e-3f;
                        }

                        if (!occluded && lightArea > 1e-6f && cosAtLight > 1e-5f) {
                            Ray lightRay;
                            lightRay.origin = scatterPoint;
                            lightRay.direction = wi;
                            float lightMediumEnter, lightMediumExit;
                            float distanceInMedium = 0.0f;
                            if (volumeRayInterval(
                                lightRay,
                                camera.volumeMin,
                                camera.volumeMax,
                                distToLight,
                                lightMediumEnter,
                                lightMediumExit)) {
                                distanceInMedium = lightMediumExit - lightMediumEnter;
                            }
                            const float transmittance = expf(-sigmaT * distanceInMedium);
                            // Both directions point away from the scattering
                            // point in the path integral convention. For a
                            // rear light, dot(ray.direction, wi) represents
                            // forward scattering toward the camera.
                            const float phase = henyeyGreensteinPhase(
                                glm::dot(pathSegment.ray.direction, wi),
                                camera.volumeG);
                            const float geometryOverPdf =
                                cosAtLight * lightArea / dist2;
                            const glm::vec3 direct = pathSegment.color *
                                camera.volumeLightRadiance *
                                (phase * geometryOverPdf * transmittance);
                            image[pathSegment.pixelIndex] += direct;
                        }
                    }
                }

                const glm::vec3 newDirection = sampleHenyeyGreenstein(
                    pathSegment.ray.direction,
                    camera.volumeG,
                    u01(rng),
                    u01(rng));
                pathSegment.ray.origin = scatterPoint + newDirection * 2e-3f;
                pathSegment.ray.direction = newDirection;
                pathSegment.remainingBounces--;
                return;
            }
        }
    }

    if (isect.t <= 0.0f) {
        if (hasEnvMap) {
            pathSegment.color *= sampleEnvMap(
                envTex,
                glm::normalize(pathSegment.ray.direction),
                envRotation,
                envIntensity);
        }
        else {
            pathSegment.color = glm::vec3(0.0f);
        }
        pathSegment.remainingBounces = 0;
        return;
    }

    const Material material = materials[isect.materialId];
    glm::vec3 p = pathSegment.ray.origin + isect.t * pathSegment.ray.direction;

    // Camera-invisible light cards are transparent only to primary rays. They
    // remain emissive geometry for bounced rays, so they still light the scene
    // without appearing as rectangles in the rendered image.
    if (material.cameraVisible == 0 && bouncesDone == 0) {
        pathSegment.ray.origin = p + glm::normalize(pathSegment.ray.direction) * 2e-3f;
        return;
    }


    if (textures != nullptr && material.emissiveTexId >= 0 && material.emissiveTexId < numTextures) {
        float4 s = tex2D<float4>(textures[material.emissiveTexId], isect.uv.x, isect.uv.y);
        glm::vec3 emissive = glm::vec3(s.x, s.y, s.z);
        if (emissive.x + emissive.y + emissive.z > 0.01f) {
            pathSegment.color *= emissive;
            pathSegment.remainingBounces = 0;
            return;
        }
    }

    if (material.emittance > 0.0f) {
        if (glm::dot(isect.surfaceNormal, -pathSegment.ray.direction) > 0.0f) {
            pathSegment.color *= (material.color * material.emittance);
        } else {
            pathSegment.color = glm::vec3(0.0f);
        }
        pathSegment.remainingBounces = 0;
        return;
    }

    glm::vec3 n = glm::normalize(isect.surfaceNormal);
    if (glm::dot(n, -pathSegment.ray.direction) < 0.0f) n = -n;

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
            return glm::normalize(local.x * t + local.y * b + local.z * d);
            };

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

    const glm::vec3 V = -pathSegment.ray.direction;

    // albedo
    glm::vec3 albedo = glm::clamp(material.color, glm::vec3(0.f), glm::vec3(1.f));
    if (material.albedoTexId >= 0 && material.albedoTexId < numTextures && textures != nullptr) {
        float4 s = tex2D<float4>(textures[material.albedoTexId], isect.uv.x, isect.uv.y);
        albedo = glm::vec3(s.x, s.y, s.z);
    }

    // roughness / metallic
    float roughness = glm::clamp(material.roughness, 0.04f, 1.0f);
    float metallic = glm::clamp(material.metallic, 0.0f, 1.0f);
    if (material.metallicRoughnessTexId >= 0 && material.metallicRoughnessTexId < numTextures && textures != nullptr) {
        float4 s = tex2D<float4>(textures[material.metallicRoughnessTexId], isect.uv.x, isect.uv.y);
        roughness = glm::clamp(s.y, 0.04f, 1.0f);  // G
        metallic = glm::clamp(s.z, 0.0f, 1.0f);  // B
    }

	// normal mapping
    if (textures != nullptr && material.normalTexId >= 0 && material.normalTexId < numTextures) {
		// sample normal map (in tangent space)
        float4 s = tex2D<float4>(textures[material.normalTexId], isect.uv.x, isect.uv.y);
        glm::vec3 nTangent = glm::normalize(glm::vec3(
            s.x * 2.0f - 1.0f,
            s.y * 2.0f - 1.0f,
            s.z * 2.0f - 1.0f
        ));

		// construct tangent and bitangent 
        glm::vec3 T = glm::normalize(glm::vec3(isect.tangent));
        T = glm::normalize(T - glm::dot(T, n) * n);  // Gram-Schmidt 
        glm::vec3 B = glm::cross(n, T) * isect.tangent.w;  // w 

		// tangent to world space
        n = glm::normalize(T * nTangent.x + B * nTangent.y + n * nTangent.z);
    }


    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), albedo, metallic);

    float NdotV = fmaxf(glm::dot(n, V), 1e-4f);
    glm::vec3 Fapx = F_Schlick(NdotV, F0);
    float specProb = (Fapx.x + Fapx.y + Fapx.z) / 3.0f;

    if (material.hasReflective <= 0.0f && metallic <= 0.0f)
        specProb = 0.0f;
    if (metallic >= 0.99f)
        specProb = 1.0f;

    specProb = glm::clamp(specProb, 0.0f, 1.0f);

    glm::vec3 newDir;
    glm::vec3 weight;
    const float EPS = 1e-4f;

    if (u01(rng) < specProb) {
        glm::vec3 H = sampleGGX(n, roughness, rng);
        newDir = glm::reflect(-V, H);

        if (glm::dot(newDir, n) <= 0.0f) {
            newDir = calculateRandomDirectionInHemisphere(n, rng);
            weight = albedo * (1.0f - metallic) / fmaxf(1.0f - specProb, 1e-4f);
        }
        else {
            float NdotL = fmaxf(glm::dot(n, newDir), 1e-4f);
            float NdotH = fmaxf(glm::dot(n, H), 1e-4f);
            float VdotH = fmaxf(glm::dot(V, H), 1e-4f);
            glm::vec3 F = F_Schlick(VdotH, F0);
            float     G = G_Smith(NdotV, NdotL, roughness);
            weight = (F * G * VdotH) / fmaxf(NdotH * NdotV * specProb, 1e-7f);
            weight *= NdotL;
        }
        pathSegment.ray.origin = p + n * EPS;
    }
    else {
        newDir = calculateRandomDirectionInHemisphere(n, rng);
        glm::vec3 H = glm::normalize(V + newDir);
        float VdotH = fmaxf(glm::dot(V, H), 1e-4f);
        glm::vec3 F = F_Schlick(VdotH, F0);
        glm::vec3 kd = (glm::vec3(1.0f) - F) * (1.0f - metallic);
        weight = kd * albedo / fmaxf(1.0f - specProb, 1e-4f);
        pathSegment.ray.origin = p + n * EPS;
    }

    pathSegment.color *= glm::clamp(weight, glm::vec3(0.f), glm::vec3(1.f));
    pathSegment.ray.direction = glm::normalize(newDir);
    pathSegment.remainingBounces--;
}



// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        if (iterationPath.remainingBounces <= 0)
        {
            image[iterationPath.pixelIndex] += iterationPath.color;
        }
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
            dev_materials,
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
            rrEnabled,
            gEnvTexObj,
            gEnvIntensity,
            gEnvRotation,
            gHasEnvMap,
            dev_textures,           
            (int)gTextures.size(),
            dev_image,
            dev_fogCards,
            g_numFogCards,
            cam,
            dev_geoms,
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
    if (!gEnableStreamCompaction) {
        finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);
    }

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image, cam);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
