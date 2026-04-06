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

static bool ReadAccessorFloats2(const tinygltf::Model& model,
    const tinygltf::Accessor& acc,
    std::vector<glm::vec2>& out)
{
    const auto& view = model.bufferViews[acc.bufferView];
    const auto& buffer = model.buffers[view.buffer];
    const unsigned char* data = buffer.data.data() + view.byteOffset + acc.byteOffset;
    const size_t stride = acc.ByteStride(view);
    if (acc.type != TINYGLTF_TYPE_VEC2 || acc.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT)
        return false;

    out.resize(acc.count);
    for (size_t i = 0; i < acc.count; ++i) {
        const float* p = reinterpret_cast<const float*>(data + i * stride);
        out[i] = glm::vec2(p[0], p[1]);
    }
    return true;
}

static bool ReadAccessorFloats4(const tinygltf::Model& model,
    const tinygltf::Accessor& acc,
    std::vector<glm::vec4>& out)
{
    const auto& view = model.bufferViews[acc.bufferView];
    const auto& buffer = model.buffers[view.buffer];
    const unsigned char* data = buffer.data.data() + view.byteOffset + acc.byteOffset;
    const size_t stride = acc.ByteStride(view);
    if (acc.type != TINYGLTF_TYPE_VEC4 || acc.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT)
        return false;

    out.resize(acc.count);
    for (size_t i = 0; i < acc.count; ++i) {
        const float* p = reinterpret_cast<const float*>(data + i * stride);
        out[i] = glm::vec4(p[0], p[1], p[2], p[3]);
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


static void ExtractGltfImage(
    const tinygltf::Model& model,
    int textureIndex,
    GltfImage& out)
{
    if (textureIndex < 0) return;
    const tinygltf::Texture& tex = model.textures[textureIndex];
    if (tex.source < 0) return;
    const tinygltf::Image& img = model.images[tex.source];

    out.w = img.width;
    out.h = img.height;
    int nPix = img.width * img.height;
    out.pixels.resize(nPix * 4);

	// tinygltf stores image data as unsigned char (0-255) with 3 or 4 components. We convert it to float RGBA (0.0-1.0).
    for (int i = 0; i < nPix; i++) {
        int comp = img.component;  // 3 or 4
        out.pixels[i * 4 + 0] = (comp > 0) ? img.image[i * comp + 0] / 255.f : 0.f;
        out.pixels[i * 4 + 1] = (comp > 1) ? img.image[i * comp + 1] / 255.f : 0.f;
        out.pixels[i * 4 + 2] = (comp > 2) ? img.image[i * comp + 2] / 255.f : 0.f;
        out.pixels[i * 4 + 3] = (comp > 3) ? img.image[i * comp + 3] / 255.f : 1.f;
    }
}

bool LoadGLTF_AsTris(const std::string& filepath,
    const glm::mat4& M_world,
    int                materialId,
    std::vector<TriCPU>& outTris,
    std::string* errOut,
    GltfMaterialTextures* outTextures)
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

			//read texcoords if available, otherwise use default texcoords (0,0)
            std::vector<glm::vec2> texcoords;
            auto itTex = prim.attributes.find("TEXCOORD_0");
            if (itTex != prim.attributes.end()) {
                const tinygltf::Accessor& accTex = model.accessors[itTex->second];
                ReadAccessorFloats2(model, accTex, texcoords);
            }
            if (texcoords.empty()) {
                texcoords.resize(positions.size(), glm::vec2(0.0f));
            }


			// read tangents if available, otherwise use default tangents (1,0,0,1)
            std::vector<glm::vec4> tangents;
            auto itTan = prim.attributes.find("TANGENT");
            if (itTan != prim.attributes.end()) {
                const tinygltf::Accessor& accTan = model.accessors[itTan->second];
                ReadAccessorFloats4(model, accTan, tangents);
            }
            if (tangents.empty()) {
                tangents.resize(positions.size(), glm::vec4(1, 0, 0, 1));  
            }

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
                glm::vec2 t0 = texcoords[i0];
                glm::vec2 t1 = texcoords[i1];
                glm::vec2 t2 = texcoords[i2];
                glm::vec4 ta0 = tangents[i0];
                glm::vec4 ta1 = tangents[i1];
                glm::vec4 ta2 = tangents[i2];
                outTris.push_back(TriCPU{ p0, p1, p2, t0, t1, t2, ta0, ta1, ta2, materialId });
            }
        }
    }


	// read textures from the first material (if any)
    if (outTextures != nullptr && !model.materials.empty()) {
        const tinygltf::Material& mat = model.materials[0];

        // albedo
        int albedoIdx = mat.pbrMetallicRoughness.baseColorTexture.index;
        ExtractGltfImage(model, albedoIdx, outTextures->albedo);

        // metallicRoughness
        int mrIdx = mat.pbrMetallicRoughness.metallicRoughnessTexture.index;
        ExtractGltfImage(model, mrIdx, outTextures->metallicRoughness);

        // normal
        int normalIdx = mat.normalTexture.index;
        ExtractGltfImage(model, normalIdx, outTextures->normal);

        // emissive
        int emissiveIdx = mat.emissiveTexture.index;
        ExtractGltfImage(model, emissiveIdx, outTextures->emissive);

        // occlusion
        int occlusionIdx = mat.occlusionTexture.index;
        ExtractGltfImage(model, occlusionIdx, outTextures->occlusion);

        printf("[GLTF] textures: albedo=%s mr=%s normal=%s\n",
            outTextures->albedo.w > 0 ? "OK" : "none",
            outTextures->metallicRoughness.w > 0 ? "OK" : "none",
            outTextures->normal.w > 0 ? "OK" : "none");
    }

    std::printf("[GLTF] %s -> tris +%zu (total %zu)\n",
        filepath.c_str(), outTris.size() - triBefore, outTris.size());
    return true;
}
