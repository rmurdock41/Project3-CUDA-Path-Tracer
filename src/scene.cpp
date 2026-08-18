#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);


    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        newMaterial.roughness = 0.0f;
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            if (p.contains("ROUGHNESS")) {
                newMaterial.roughness = (float)p["ROUGHNESS"];
            }
            if (p.contains("METALLIC")) {
                newMaterial.metallic = (float)p["METALLIC"];
            }
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = p.contains("REFLECTIVE")
                ? (float)p["REFLECTIVE"]   
                : 1.0f;                   
            newMaterial.hasRefractive = 0.0f;

            if (p.contains("ROUGHNESS")) {
                newMaterial.roughness = (float)p["ROUGHNESS"];
            }
            else {
                newMaterial.roughness = 0.0f;
            }
            newMaterial.emittance = 0.0f;
            if (p.contains("METALLIC")) {
                newMaterial.metallic = (float)p["METALLIC"];
            }
        }
        else if (p["TYPE"] == "Refractive")
        {
            // Support glass
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0f;
            if (p.contains("REFLECTIVE")) {
                newMaterial.hasReflective = (float)p["REFLECTIVE"];
            }
            else {
                newMaterial.hasReflective = 1.0f; // default full Fresnel
            } 
            if (p.contains("IOR")) {
                newMaterial.indexOfRefraction = (float)p["IOR"];
            }
            else if (p.contains("indexOfRefraction")) {
                newMaterial.indexOfRefraction = (float)p["indexOfRefraction"];
            }
            else {
                newMaterial.indexOfRefraction = 1.5f; // default glass IOR
            }
            if (p.contains("ROUGHNESS")) {
                newMaterial.roughness = (float)p["ROUGHNESS"];
            }
        }

        else if (p["TYPE"] == "PBR")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.0f;
            newMaterial.hasRefractive = 0.0f;
            if (p.contains("ROUGHNESS")) {
                newMaterial.roughness = (float)p["ROUGHNESS"];
            }
            if (p.contains("METALLIC")) {
                newMaterial.metallic = (float)p["METALLIC"];
            }
        }
        if (p.contains("CAMERA_VISIBLE")) {
            newMaterial.cameraVisible = (bool)p["CAMERA_VISIBLE"] ? 1 : 0;
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        if (type == "mesh" || type == "gltf")
        {
            int matId = MatNameToID[p["MATERIAL"]];
            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];
            glm::vec3 T(trans[0], trans[1], trans[2]);
            glm::vec3 R(rotat[0], rotat[1], rotat[2]);
            glm::vec3 S(scale[0], scale[1], scale[2]);
            glm::mat4 M = utilityCore::buildTransformationMatrix(T, R, S);

            MeshInstance mi;
            mi.path = p["FILE"];   
            mi.materialId = matId;
            mi.M_world = M;
            meshInstances.push_back(mi);
            continue; 
        }

        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    // ===== DoF params (optional; default to 0 == disabled) =====
    if (cameraData.contains("APERTURE_RADIUS"))
        camera.apertureRadius = cameraData["APERTURE_RADIUS"];
    else
        camera.apertureRadius = 0.0f;

    if (cameraData.contains("FOCAL_DISTANCE"))
        camera.focalDistance = cameraData["FOCAL_DISTANCE"];
    else
        camera.focalDistance = 0.0f;

    // ===== HDR post-processing (optional) =====
    // Kept outside Camera in the JSON because these settings describe the
    // output pipeline rather than the physical lens transform.
    if (data.contains("PostProcess")) {
        const auto& post = data["PostProcess"];
        camera.postEnabled = post.value("ENABLED", true) ? 1 : 0;
        camera.exposure = post.value("EXPOSURE", 1.0f);
        camera.gamma = post.value("GAMMA", 2.2f);

        if (post.contains("Bloom")) {
            const auto& bloom = post["Bloom"];
            camera.bloomEnabled = bloom.value("ENABLED", true) ? 1 : 0;
            camera.bloomThreshold = bloom.value("THRESHOLD", 1.0f);
            camera.bloomStrength = bloom.value("STRENGTH", 0.15f);
            camera.bloomRadius = bloom.value("RADIUS", 18.0f);
            camera.bloomSamples = bloom.value("SAMPLES", 32);
        }

        if (post.contains("GodRays")) {
            const auto& rays = post["GodRays"];
            camera.godRaysEnabled = rays.value("ENABLED", true) ? 1 : 0;
            camera.godRaysFocusEnabled = rays.value("FOCUS_ENABLED", false) ? 1 : 0;
            camera.godRaysConvergeEnabled = rays.value("CONVERGE_ENABLED", false) ? 1 : 0;
            camera.godRaysDirectionalEnabled = rays.value("DIRECTIONAL_ENABLED", false) ? 1 : 0;
            if (rays.contains("CENTER") && rays["CENTER"].size() >= 2) {
                camera.godRaysCenter = glm::vec2(rays["CENTER"][0], rays["CENTER"][1]);
            }
            if (rays.contains("TARGET") && rays["TARGET"].size() >= 2) {
                camera.godRaysTarget = glm::vec2(rays["TARGET"][0], rays["TARGET"][1]);
            }
            if (rays.contains("DIRECTION") && rays["DIRECTION"].size() >= 2) {
                camera.godRaysDirection = glm::vec2(
                    rays["DIRECTION"][0], rays["DIRECTION"][1]);
            }
            camera.godRaysThreshold = rays.value("THRESHOLD", 0.75f);
            camera.godRaysSamples = rays.value("SAMPLES", 48);
            camera.godRaysDensity = rays.value("DENSITY", 0.85f);
            camera.godRaysDecay = rays.value("DECAY", 0.965f);
            camera.godRaysWeight = rays.value("WEIGHT", 0.018f);
            camera.godRaysStrength = rays.value("STRENGTH", 0.12f);
            camera.godRaysLength = rays.value("LENGTH", 0.4f);
            camera.godRaysWidth = rays.value("WIDTH", 0.16f);
            camera.godRaysSoftness = rays.value("SOFTNESS", 0.5f);
            if (rays.contains("HAZE_CENTER") && rays["HAZE_CENTER"].size() >= 2) {
                camera.godRaysHazeCenter = glm::vec2(
                    rays["HAZE_CENTER"][0], rays["HAZE_CENTER"][1]);
            }
            if (rays.contains("HAZE_RADIUS") && rays["HAZE_RADIUS"].size() >= 2) {
                camera.godRaysHazeRadius = glm::vec2(
                    rays["HAZE_RADIUS"][0], rays["HAZE_RADIUS"][1]);
            }
            if (rays.contains("HAZE_COLOR") && rays["HAZE_COLOR"].size() >= 3) {
                camera.godRaysHazeColor = glm::vec3(
                    rays["HAZE_COLOR"][0], rays["HAZE_COLOR"][1], rays["HAZE_COLOR"][2]);
            }
            camera.godRaysHazeStrength = rays.value("HAZE_STRENGTH", 0.0f);
            if (rays.contains("HAZE_SUBJECT_CENTER") && rays["HAZE_SUBJECT_CENTER"].size() >= 2) {
                camera.godRaysHazeSubjectCenter = glm::vec2(
                    rays["HAZE_SUBJECT_CENTER"][0], rays["HAZE_SUBJECT_CENTER"][1]);
            }
            if (rays.contains("HAZE_SUBJECT_RADIUS") && rays["HAZE_SUBJECT_RADIUS"].size() >= 2) {
                camera.godRaysHazeSubjectRadius = glm::vec2(
                    rays["HAZE_SUBJECT_RADIUS"][0], rays["HAZE_SUBJECT_RADIUS"][1]);
            }
            camera.godRaysHazeSubjectProtect = rays.value("HAZE_SUBJECT_PROTECT", 0.92f);
            camera.godRaysHazeRimStrength = rays.value("HAZE_RIM_STRENGTH", 0.025f);
            camera.godRaysHazeRimWidth = rays.value("HAZE_RIM_WIDTH", 0.18f);
            camera.godRaysVerticalMin = rays.value("VERTICAL_MIN", 0.15f);
            camera.godRaysVerticalStart = rays.value("VERTICAL_START", 0.12f);
            camera.godRaysVerticalEnd = rays.value("VERTICAL_END", 0.45f);
            camera.godRaysEndY = rays.value("END_Y", camera.godRaysTarget.y);
        }
    }

    // ===== Bounded homogeneous participating medium (optional) =====
    // This is a physical path-traced volume, independent of the screen-space
    // GodRays post effect. Existing JSON files remain unchanged by default.
    if (data.contains("Volume")) {
        const auto& volume = data["Volume"];
        camera.volumeEnabled = volume.value("ENABLED", true) ? 1 : 0;
        if (volume.contains("BOUNDS_MIN") && volume["BOUNDS_MIN"].size() >= 3) {
            camera.volumeMin = glm::vec3(
                volume["BOUNDS_MIN"][0],
                volume["BOUNDS_MIN"][1],
                volume["BOUNDS_MIN"][2]);
        }
        if (volume.contains("BOUNDS_MAX") && volume["BOUNDS_MAX"].size() >= 3) {
            camera.volumeMax = glm::vec3(
                volume["BOUNDS_MAX"][0],
                volume["BOUNDS_MAX"][1],
                volume["BOUNDS_MAX"][2]);
        }
        camera.volumeSigmaA = glm::max(0.0f, volume.value("SIGMA_A", 0.0f));
        camera.volumeSigmaS = glm::max(0.0f, volume.value("SIGMA_S", 0.0f));
        camera.volumeG = glm::clamp(volume.value("ANISOTROPY", 0.0f), -0.95f, 0.95f);
        if (volume.contains("SCATTER_COLOR") && volume["SCATTER_COLOR"].size() >= 3) {
            camera.volumeScatterColor = glm::clamp(glm::vec3(
                volume["SCATTER_COLOR"][0],
                volume["SCATTER_COLOR"][1],
                volume["SCATTER_COLOR"][2]), glm::vec3(0.0f), glm::vec3(1.0f));
        }

        if (volume.contains("Light")) {
            const auto& light = volume["Light"];
            camera.volumeLightEnabled = light.value("ENABLED", true) ? 1 : 0;
            if (light.contains("CENTER") && light["CENTER"].size() >= 3) {
                camera.volumeLightCenter = glm::vec3(
                    light["CENTER"][0], light["CENTER"][1], light["CENTER"][2]);
            }
            if (light.contains("U") && light["U"].size() >= 3) {
                camera.volumeLightU = glm::vec3(
                    light["U"][0], light["U"][1], light["U"][2]);
            }
            if (light.contains("V") && light["V"].size() >= 3) {
                camera.volumeLightV = glm::vec3(
                    light["V"][0], light["V"][1], light["V"][2]);
            }
            if (light.contains("RADIANCE") && light["RADIANCE"].size() >= 3) {
                camera.volumeLightRadiance = glm::max(glm::vec3(
                    light["RADIANCE"][0],
                    light["RADIANCE"][1],
                    light["RADIANCE"][2]), glm::vec3(0.0f));
            }
        }
    }

    // ===== Depth-aware alpha fog cards (optional) =====
    // Cards are kept out of the geometry BVH and are composited once along
    // primary camera rays. This keeps them cheap and prevents them from
    // participating in reflections, indirect lighting, or shadow rays.
    if (data.contains("FogCards")) {
        const auto& fogBlock = data["FogCards"];
        const bool enabled = fogBlock.is_array()
            ? true
            : fogBlock.value("ENABLED", true);
        if (enabled &&
            (fogBlock.is_array() ||
                (fogBlock.contains("CARDS") && fogBlock["CARDS"].is_array()))) {
            const auto& cards = fogBlock.is_array() ? fogBlock : fogBlock["CARDS"];
            for (const auto& p : cards) {
                if (!p.contains("TEXTURE") ||
                    !p.contains("CENTER") ||
                    !p.contains("RIGHT") ||
                    !p.contains("UP") ||
                    p["CENTER"].size() < 3 ||
                    p["RIGHT"].size() < 3 ||
                    p["UP"].size() < 3) {
                    continue;
                }

                FogCardConfig cfg;
                cfg.texturePath = p["TEXTURE"].get<std::string>();
                cfg.card.center = glm::vec3(
                    p["CENTER"][0], p["CENTER"][1], p["CENTER"][2]);
                cfg.card.right = glm::vec3(
                    p["RIGHT"][0], p["RIGHT"][1], p["RIGHT"][2]);
                cfg.card.up = glm::vec3(
                    p["UP"][0], p["UP"][1], p["UP"][2]);
                cfg.card.opacity = glm::clamp(
                    p.value("OPACITY", 0.15f), 0.0f, 1.0f);
                cfg.card.depthFade = glm::max(
                    p.value("DEPTH_FADE", 0.75f), 0.0f);
                cfg.card.edgeFade = glm::clamp(
                    p.value("EDGE_FADE", 0.18f), 0.0f, 0.49f);
                if (p.contains("COLOR") && p["COLOR"].size() >= 3) {
                    cfg.card.color = glm::max(glm::vec3(
                        p["COLOR"][0], p["COLOR"][1], p["COLOR"][2]),
                        glm::vec3(0.0f));
                }

                if (glm::dot(cfg.card.right, cfg.card.right) > 1e-8f &&
                    glm::dot(cfg.card.up, cfg.card.up) > 1e-8f) {
                    fogCards.push_back(cfg);
                }
            }
        }
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
