#pragma once
#include <vector>
#include <string>
#include <glm/glm.hpp>

struct TriCPU {
    glm::vec3 v0, v1, v2;   
    glm::vec2 uv0, uv1, uv2;
    glm::vec4 tan0, tan1, tan2;
    int       materialId;
};

struct GltfImage {
    std::vector<float> pixels;  // float RGBA
    int w = 0, h = 0;
};


struct GltfMaterialTextures {
    GltfImage albedo;
    GltfImage metallicRoughness;  
    GltfImage normal;
    GltfImage emissive;
    GltfImage occlusion;
};

bool LoadGLTF_AsTris(const std::string& filepath,
    const glm::mat4& M_world,
    int                materialId,
    std::vector<TriCPU>& outTris,
    std::string* errOut,
    GltfMaterialTextures* outTextures = nullptr);