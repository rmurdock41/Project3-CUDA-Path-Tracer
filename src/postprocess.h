#pragma once

#include <vector>

#include <glm/glm.hpp>

#include "sceneStructs.h"

// Applies the same HDR post-processing used by the CUDA preview to the
// accumulated image before it is written to PNG.
std::vector<glm::vec3> applyPostProcessCPU(
    const std::vector<glm::vec3>& linearImage,
    int width,
    int height,
    const Camera& camera);
