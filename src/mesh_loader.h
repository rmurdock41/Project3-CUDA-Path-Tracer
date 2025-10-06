#pragma once
#include <vector>
#include <string>
#include <glm/glm.hpp>

struct TriCPU {
    glm::vec3 v0, v1, v2;   
    int       materialId;
};

bool LoadGLTF_AsTris(const std::string& filepath,
    const glm::mat4& M_world,
    int                materialId,
    std::vector<TriCPU>& outTris,
    std::string* errOut);
