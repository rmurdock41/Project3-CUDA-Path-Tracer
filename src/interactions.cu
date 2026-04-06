#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    if (glm::dot(normal, -pathSegment.ray.direction) < 0.f)
        normal = -normal;

    const glm::vec3 N = normal;
    const glm::vec3 V = -pathSegment.ray.direction;
    const glm::vec3 albedo = glm::clamp(m.color, glm::vec3(0.f), glm::vec3(1.f));
    const float roughness = glm::clamp(m.roughness, 0.04f, 1.0f);
    const float metallic = glm::clamp(m.metallic, 0.0f, 1.0f);

    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), albedo, metallic);

    float NdotV = fmaxf(glm::dot(N, V), 1e-4f);
    glm::vec3 F_approx = F_Schlick(NdotV, F0);
    float specProb = glm::clamp((F_approx.x + F_approx.y + F_approx.z) / 3.0f, 0.0f, 1.0f);

    if (metallic >= 0.99f) specProb = 1.0f;

    glm::vec3 newDir;
    glm::vec3 weight;

    if (u01(rng) < specProb) {
        glm::vec3 H = sampleGGX(N, roughness, rng);
        newDir = glm::reflect(-V, H);

        if (glm::dot(newDir, N) <= 0.0f) {
            newDir = calculateRandomDirectionInHemisphere(N, rng);
            weight = albedo * (1.0f - metallic) / fmaxf(1.0f - specProb, 1e-4f);
        }
        else {
            float NdotL = fmaxf(glm::dot(N, newDir), 1e-4f);
            float NdotH = fmaxf(glm::dot(N, H), 1e-4f);
            float VdotH = fmaxf(glm::dot(V, H), 1e-4f);
            glm::vec3 F = F_Schlick(VdotH, F0);
            float     G = G_Smith(NdotV, NdotL, roughness);
            weight = F * G / fmaxf(NdotV * specProb, 1e-7f);
        }
    }
    else {
        newDir = calculateRandomDirectionInHemisphere(N, rng);
        glm::vec3 H = glm::normalize(V + newDir);
        float VdotH = fmaxf(glm::dot(V, H), 1e-4f);
        glm::vec3 F = F_Schlick(VdotH, F0);
        glm::vec3 kd = (glm::vec3(1.0f) - F) * (1.0f - metallic);
        weight = kd * albedo / fmaxf(1.0f - specProb, 1e-4f);
    }

    pathSegment.color *= glm::clamp(weight, glm::vec3(0.f), glm::vec3(1.f));
    pathSegment.ray.origin = intersect + normal * 1e-4f;
    pathSegment.ray.direction = glm::normalize(newDir);
    pathSegment.remainingBounces--;
}