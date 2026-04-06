#pragma once

#include "sceneStructs.h"

#include "utilities.h"
#include <glm/glm.hpp>

#include <thrust/random.h>

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, 
    thrust::default_random_engine& rng);

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng);

__host__ __device__ inline float G_Smith(float NdotV, float NdotL, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float gv = NdotV + sqrtf(a2 + (1.0f - a2) * NdotV * NdotV);
    float gl = NdotL + sqrtf(a2 + (1.0f - a2) * NdotL * NdotL);
    return (2.0f * NdotV * NdotL) / fmaxf(gv * gl, 1e-7f);
}

__host__ __device__ inline glm::vec3 F_Schlick(float VdotH, glm::vec3 F0) {
    float f = powf(1.0f - fmaxf(VdotH, 0.0f), 5.0f);
    return F0 + (glm::vec3(1.0f) - F0) * f;
}

__host__ __device__ inline glm::vec3 sampleGGX(
    const glm::vec3& N, float roughness,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    float u1 = u01(rng);
    float u2 = u01(rng);
    float a = roughness * roughness;
    float cosTheta = sqrtf((1.0f - u1) / fmaxf(1.0f + (a * a - 1.0f) * u1, 1e-7f));
    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = TWO_PI * u2;
    glm::vec3 up = (fabsf(N.z) < 0.999f) ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0);
    glm::vec3 T = glm::normalize(glm::cross(up, N));
    glm::vec3 B = glm::cross(N, T);
    glm::vec3 H_local(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
    return glm::normalize(H_local.x * T + H_local.y * B + H_local.z * N);
}
