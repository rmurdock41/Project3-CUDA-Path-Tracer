#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE = 0,
    CUBE = 1,
    MESH = 2,
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

// Camera-facing or artist-oriented textured plane used for inexpensive,
// depth-aware fog layering. right/up are half-edge vectors in world space.
struct FogCard
{
    glm::vec3 center = glm::vec3(0.0f);
    glm::vec3 right = glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 color = glm::vec3(0.12f);
    float opacity = 0.15f;
    float depthFade = 0.75f;
    float edgeFade = 0.18f;
    int textureId = -1;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    int       triOffset = -1;         
    int       triCount = 0;          
    glm::vec3 bboxMin = glm::vec3(0.0f); 
    glm::vec3 bboxMax = glm::vec3(0.0f);
    int       triBVHRoot = -1;

};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
    float roughness;
    float metallic = 0.0f;
    int cameraVisible = 1;  // 0: primary rays pass through; secondary rays still see the light

    int albedoTexId = -1;
    int metallicRoughnessTexId = -1;
    int normalTexId = -1;
    int emissiveTexId = -1;
    int occlusionTexId = -1;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;

    float apertureRadius = 0.0f;   // lens radius; 0 = pinhole (no DoF)
    float focalDistance = 0.0f;   // focus distance along cam.view; if <=0 we'll fallback

    // HDR post-processing. Disabled by default so existing scenes keep their
    // original appearance until a PostProcess block is added to the JSON.
    int postEnabled = 0;
    float exposure = 1.0f;
    float gamma = 2.2f;

    int bloomEnabled = 0;
    float bloomThreshold = 1.0f;
    float bloomStrength = 0.15f;
    float bloomRadius = 18.0f;
    int bloomSamples = 32;

    int godRaysEnabled = 0;
    int godRaysFocusEnabled = 0;
    int godRaysConvergeEnabled = 0;
    int godRaysDirectionalEnabled = 0;
    glm::vec2 godRaysCenter = glm::vec2(0.5f, 0.3f);
    glm::vec2 godRaysTarget = glm::vec2(0.5f, 0.55f);
    glm::vec2 godRaysDirection = glm::vec2(0.0f, 1.0f);
    float godRaysThreshold = 0.75f;
    int godRaysSamples = 48;
    float godRaysDensity = 0.85f;
    float godRaysDecay = 0.965f;
    float godRaysWeight = 0.018f;
    float godRaysStrength = 0.12f;
    float godRaysLength = 0.4f;
    float godRaysWidth = 0.16f;
    float godRaysSoftness = 0.5f;
    glm::vec2 godRaysHazeCenter = glm::vec2(0.5f, 0.48f);
    glm::vec2 godRaysHazeRadius = glm::vec2(0.3f, 0.22f);
    glm::vec3 godRaysHazeColor = glm::vec3(0.72f, 0.78f, 0.82f);
    float godRaysHazeStrength = 0.0f;
    glm::vec2 godRaysHazeSubjectCenter = glm::vec2(0.5f, 0.47f);
    glm::vec2 godRaysHazeSubjectRadius = glm::vec2(0.09f, 0.12f);
    float godRaysHazeSubjectProtect = 0.92f;
    float godRaysHazeRimStrength = 0.025f;
    float godRaysHazeRimWidth = 0.18f;
    float godRaysVerticalMin = 0.15f;
    float godRaysVerticalStart = 0.12f;
    float godRaysVerticalEnd = 0.45f;
    float godRaysEndY = 0.55f;

    // Optional bounded homogeneous participating medium.  Keeping this data
    // in Camera makes it part of the small render-configuration block that is
    // already copied by value to the CUDA kernels; it does not affect camera
    // projection or any existing scene when disabled.
    int volumeEnabled = 0;
    glm::vec3 volumeMin = glm::vec3(0.0f);
    glm::vec3 volumeMax = glm::vec3(0.0f);
    float volumeSigmaA = 0.0f;
    float volumeSigmaS = 0.0f;
    float volumeG = 0.0f;
    glm::vec3 volumeScatterColor = glm::vec3(1.0f);

    // Analytic rectangular emitter used only for volumetric next-event
    // estimation. U and V are half-edge vectors in world space.
    int volumeLightEnabled = 0;
    glm::vec3 volumeLightCenter = glm::vec3(0.0f);
    glm::vec3 volumeLightU = glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 volumeLightV = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 volumeLightRadiance = glm::vec3(0.0f);

};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    int fogCardsProcessed;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  glm::vec2 uv = glm::vec2(0.0f); // TODO: for texturing
  glm::vec4 tangent = glm::vec4(1, 0, 0, 1);
};
