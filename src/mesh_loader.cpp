#include "mesh_loader.h"
#include "tinygltf/tiny_gltf.h"
#include <cstdio>

static bool ReadAccessorFloats3(const tinygltf::Model& model,
    const tinygltf::Accessor& acc,
    std::vector<glm::vec3>& out)
{
    const auto& view = model.bufferViews[acc.bufferView];
    const auto& buffer = model.buffers[view.buffer];
    const unsigned char* data = buffer.data.data() + view.byteOffset + acc.byteOffset;
    const size_t stride = acc.ByteStride(view);
    if (acc.type != TINYGLTF_TYPE_VEC3 || acc.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT)
        return false;

    out.resize(acc.count);
    for (size_t i = 0; i < acc.count; ++i) {
        const float* p = reinterpret_cast<const float*>(data + i * stride);
        out[i] = glm::vec3(p[0], p[1], p[2]);
    }
    return true;
}

static bool ReadIndicesAsU32(const tinygltf::Model& model,
    const tinygltf::Accessor& acc,
    std::vector<uint32_t>& out)
{
    const auto& view = model.bufferViews[acc.bufferView];
    const auto& buffer = model.buffers[view.buffer];
    const unsigned char* data = buffer.data.data() + view.byteOffset + acc.byteOffset;
    const size_t stride = acc.ByteStride(view);
    out.resize(acc.count);

    switch (acc.componentType) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        for (size_t i = 0; i < acc.count; ++i)
            out[i] = *(reinterpret_cast<const uint16_t*>(data + i * stride));
        return true;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        for (size_t i = 0; i < acc.count; ++i)
            out[i] = *(reinterpret_cast<const uint32_t*>(data + i * stride));
        return true;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        for (size_t i = 0; i < acc.count; ++i)
            out[i] = *(reinterpret_cast<const uint8_t*>(data + i * stride));
        return true;
    default:
        return false;
    }
}

bool LoadGLTF_AsTris(const std::string& filepath,
    const glm::mat4& M_world,
    int                materialId,
    std::vector<TriCPU>& outTris,
    std::string* errOut)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string warn, err;

    bool ok = false;
    if (filepath.size() >= 4 && filepath.substr(filepath.size() - 4) == ".glb")
        ok = loader.LoadBinaryFromFile(&model, &err, &warn, filepath);
    else
        ok = loader.LoadASCIIFromFile(&model, &err, &warn, filepath);
    if (!ok) {
        if (errOut) *errOut = warn + err;
        std::fprintf(stderr, "[tinygltf] load failed: %s %s\n", warn.c_str(), err.c_str());
        return false;
    }

    size_t triBefore = outTris.size();
    for (const auto& mesh : model.meshes) {
        for (const auto& prim : mesh.primitives) {
            if (prim.mode != TINYGLTF_MODE_TRIANGLES) continue;


            auto itPos = prim.attributes.find("POSITION");
            if (itPos == prim.attributes.end()) continue;
            const tinygltf::Accessor& accPos = model.accessors[itPos->second];
            std::vector<glm::vec3> positions;
            if (!ReadAccessorFloats3(model, accPos, positions)) continue;


            std::vector<uint32_t> indices;
            if (prim.indices >= 0) {
                const tinygltf::Accessor& accIdx = model.accessors[prim.indices];
                if (!ReadIndicesAsU32(model, accIdx, indices)) continue;
            }
            else {
                indices.resize(positions.size());
                for (uint32_t i = 0; i < indices.size(); ++i) indices[i] = i;
            }
            if (indices.size() % 3 != 0) continue;

            for (size_t i = 0; i + 2 < indices.size(); i += 3) {
                uint32_t i0 = indices[i + 0], i1 = indices[i + 1], i2 = indices[i + 2];
                glm::vec3 p0 = glm::vec3(M_world * glm::vec4(positions[i0], 1.f));
                glm::vec3 p1 = glm::vec3(M_world * glm::vec4(positions[i1], 1.f));
                glm::vec3 p2 = glm::vec3(M_world * glm::vec4(positions[i2], 1.f));
                outTris.push_back(TriCPU{ p0,p1,p2, materialId });
            }
        }
    }

    std::printf("[GLTF] %s -> tris +%zu (total %zu)\n",
        filepath.c_str(), outTris.size() - triBefore, outTris.size());
    return true;
}
