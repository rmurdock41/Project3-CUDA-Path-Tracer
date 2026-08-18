#include "postprocess.h"

#include <algorithm>
#include <cmath>

namespace {

float clamp01(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

glm::vec3 sampleNearest(
    const std::vector<glm::vec3>& image,
    int width,
    int height,
    float x,
    float y) {
    const int sx = std::max(0, std::min(width - 1, static_cast<int>(std::round(x))));
    const int sy = std::max(0, std::min(height - 1, static_cast<int>(std::round(y))));
    return image[sx + sy * width];
}

glm::vec3 brightPass(const glm::vec3& color, float threshold) {
    const float brightness = std::max(color.x, std::max(color.y, color.z));
    if (brightness <= threshold) {
        return glm::vec3(0.0f);
    }
    const float scale = (brightness - threshold) / std::max(brightness, 1.0e-5f);
    return color * clamp01(scale);
}

glm::vec3 acesToneMap(const glm::vec3& color) {
    const glm::vec3 a = color * (2.51f * color + glm::vec3(0.03f));
    const glm::vec3 b = color * (2.43f * color + glm::vec3(0.59f)) + glm::vec3(0.14f);
    return glm::clamp(a / b, glm::vec3(0.0f), glm::vec3(1.0f));
}

glm::vec3 gammaCorrect(const glm::vec3& color, float gamma) {
    const float invGamma = 1.0f / std::max(gamma, 0.01f);
    return glm::vec3(
        std::pow(clamp01(color.x), invGamma),
        std::pow(clamp01(color.y), invGamma),
        std::pow(clamp01(color.z), invGamma));
}

float smoothstep(float edge0, float edge1, float value) {
    const float range = std::max(edge1 - edge0, 1.0e-5f);
    const float t = clamp01((value - edge0) / range);
    return t * t * (3.0f - 2.0f * t);
}

float godRayConeMask(const glm::vec2& uv, const Camera& camera) {
    glm::vec2 direction = camera.godRaysTarget - camera.godRaysCenter;
    const float directionLength = glm::length(direction);
    if (directionLength < 1.0e-5f) {
        return 1.0f;
    }
    direction /= directionLength;

    const glm::vec2 relative = uv - camera.godRaysCenter;
    const float along = glm::dot(relative, direction);
    const float maxLength = std::max(camera.godRaysLength, 1.0e-4f);
    if (along < 0.0f || along > maxLength) {
        return 0.0f;
    }

    const float progress = clamp01(along / maxLength);
    const float halfWidth = std::max(camera.godRaysWidth, 1.0e-4f) *
        (0.18f + 0.82f * progress);
    const float perpendicular = std::abs(relative.x * direction.y - relative.y * direction.x);
    const float softness = std::max(0.01f, std::min(camera.godRaysSoftness, 0.99f));
    const float sideMask = 1.0f - smoothstep(halfWidth * (1.0f - softness), halfWidth, perpendicular);
    const float endMask = 1.0f - smoothstep(maxLength * 0.82f, maxLength, along);
    return sideMask * endMask;
}

float godRayHazeMask(const glm::vec2& uv, const Camera& camera) {
    const glm::vec2 radius = glm::max(camera.godRaysHazeRadius, glm::vec2(1.0e-4f));
    const glm::vec2 offset = (uv - camera.godRaysHazeCenter) / radius;
    const float distance = glm::length(offset);
    const float falloff = 1.0f - smoothstep(0.05f, 1.0f, distance);
    return falloff * falloff * (3.0f - 2.0f * falloff);
}

void godRaySubjectMasks(
    const glm::vec2& uv,
    const Camera& camera,
    float& subjectMask,
    float& rimMask) {
    const glm::vec2 radius = glm::max(
        camera.godRaysHazeSubjectRadius, glm::vec2(1.0e-4f));
    const glm::vec2 offset = (uv - camera.godRaysHazeSubjectCenter) / radius;
    const float distance = glm::length(offset);
    const float rimWidth = std::max(0.02f, std::min(camera.godRaysHazeRimWidth, 0.45f));

    subjectMask = 1.0f - smoothstep(1.0f - rimWidth, 1.0f, distance);
    const float outerMask = 1.0f - smoothstep(1.0f, 1.0f + rimWidth, distance);
    const float innerMask = 1.0f - smoothstep(1.0f - rimWidth, 1.0f, distance);
    rimMask = clamp01(outerMask - innerMask);
}

} // namespace

std::vector<glm::vec3> applyPostProcessCPU(
    const std::vector<glm::vec3>& linearImage,
    int width,
    int height,
    const Camera& camera) {
    if (!camera.postEnabled || width <= 0 || height <= 0) {
        return linearImage;
    }

    std::vector<glm::vec3> exposed(linearImage.size());
    std::vector<glm::vec3> output(linearImage.size());
    for (size_t i = 0; i < linearImage.size(); ++i) {
        exposed[i] = glm::max(linearImage[i] * camera.exposure, glm::vec3(0.0f));
    }

    const int bloomSamples = std::max(1, std::min(camera.bloomSamples, 64));
    const float bloomRadius = std::max(camera.bloomRadius, 0.0f);
    const int raySamples = std::max(1, std::min(camera.godRaysSamples, 96));
    constexpr float goldenAngle = 2.39996323f;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int index = x + y * width;
            glm::vec3 combined = exposed[index];

            if (camera.bloomEnabled && camera.bloomStrength > 0.0f && bloomRadius > 0.0f) {
                glm::vec3 bloom(0.0f);
                float totalWeight = 0.0f;
                for (int i = 0; i < bloomSamples; ++i) {
                    const float t = (static_cast<float>(i) + 0.5f) / static_cast<float>(bloomSamples);
                    const float radius = bloomRadius * std::sqrt(t);
                    const float angle = goldenAngle * static_cast<float>(i);
                    const float weight = 1.0f - 0.75f * t;
                    const glm::vec3 sample = sampleNearest(
                        exposed,
                        width,
                        height,
                        static_cast<float>(x) + std::cos(angle) * radius,
                        static_cast<float>(y) + std::sin(angle) * radius);
                    bloom += brightPass(sample, camera.bloomThreshold) * weight;
                    totalWeight += weight;
                }
                combined += bloom * (camera.bloomStrength / std::max(totalWeight, 1.0e-5f));
            }

            if (camera.godRaysEnabled) {
                const glm::vec2 pixelUv(
                    (static_cast<float>(x) + 0.5f) / static_cast<float>(width),
                    (static_cast<float>(y) + 0.5f) / static_cast<float>(height));
                if (camera.godRaysStrength > 0.0f && camera.godRaysWeight > 0.0f) {
                glm::vec2 sampleUv = pixelUv;
                glm::vec2 rayStep(0.0f);
                if (camera.godRaysDirectionalEnabled) {
                    const float directionLength = glm::length(camera.godRaysDirection);
                    if (directionLength > 1.0e-5f) {
                        rayStep = -(camera.godRaysDirection / directionLength) *
                            (camera.godRaysDensity / static_cast<float>(raySamples));
                    }
                }
                else if (camera.godRaysConvergeEnabled) {
                    const glm::vec2 awayFromTarget = sampleUv - camera.godRaysTarget;
                    const float distanceFromTarget = glm::length(awayFromTarget);
                    if (distanceFromTarget > 1.0e-5f) {
                        rayStep = (awayFromTarget / distanceFromTarget) *
                            (camera.godRaysDensity / static_cast<float>(raySamples));
                    }
                }
                else {
                    rayStep = -((sampleUv - camera.godRaysCenter) *
                        (camera.godRaysDensity / static_cast<float>(raySamples)));
                }
                float illuminationDecay = 1.0f;
                glm::vec3 rays(0.0f);
                for (int i = 0; i < raySamples; ++i) {
                    sampleUv += rayStep;
                    const glm::vec3 sample = sampleNearest(
                        exposed,
                        width,
                        height,
                        sampleUv.x * static_cast<float>(width) - 0.5f,
                        sampleUv.y * static_cast<float>(height) - 0.5f);
                    float sourceWeight = 1.0f;
                    if (camera.godRaysConvergeEnabled || camera.godRaysDirectionalEnabled) {
                        const float vertical = smoothstep(
                            camera.godRaysVerticalStart,
                            camera.godRaysVerticalEnd,
                            sampleUv.y);
                        sourceWeight = camera.godRaysVerticalMin +
                            (1.0f - camera.godRaysVerticalMin) * vertical;
                    }
                    rays += brightPass(sample, camera.godRaysThreshold) *
                        (illuminationDecay * camera.godRaysWeight * sourceWeight);
                    illuminationDecay *= camera.godRaysDecay;
                }
                const float focusMask = camera.godRaysFocusEnabled ?
                    godRayConeMask(pixelUv, camera) : 1.0f;
                const float convergenceEndMask =
                    (camera.godRaysConvergeEnabled || camera.godRaysDirectionalEnabled) ?
                    (1.0f - smoothstep(
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
                    clamp01(camera.godRaysHazeSubjectProtect) * subjectMask;
                combined += camera.godRaysHazeColor *
                    (camera.godRaysHazeStrength * hazeMask * subjectVisibility);
                combined += camera.godRaysHazeColor *
                    (camera.godRaysHazeRimStrength * hazeMask * rimMask);
            }

            output[index] = gammaCorrect(acesToneMap(combined), camera.gamma);
        }
    }

    return output;
}
