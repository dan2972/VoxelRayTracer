#ifndef MESH_H
#define MESH_H

#include <vector>

struct Mesh {
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
};

#endif // MESH_H