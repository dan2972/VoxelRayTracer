#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <glad/glad.h>
#include "mesh.h"

struct RenderInfo {
    GLuint vao;
    GLuint indicesCount;
};

class Model {
public:
    Model() = default;
    ~Model();

    void addData(const Mesh& mesh, const std::vector<int>& vbo_dimensions = {2,2});
    void deleteData();

    void genVAO();
    void addEBO(const std::vector<GLuint> &indices);
    void bind() const;

    void addVBO(const std::vector<int>& dimensions, const std::vector<GLfloat> &data);
    
    const RenderInfo& getRenderInfo() const;

private:
    RenderInfo m_renderInfo;

    int m_vboCount = 0;
    std::vector<GLuint> m_buffers;
};

#endif // MODEL_H