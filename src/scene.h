#pragma once

#include "sceneStructs.h"
#include <vector>
#include <string>       
#include <glm/glm.hpp> 



struct MeshInstance {
    std::string path;   // JSON: "FILE"
    int         materialId;   // JSON: "MATERIAL"
    glm::mat4   M_world;      
};

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    std::vector<MeshInstance> meshInstances;
};
