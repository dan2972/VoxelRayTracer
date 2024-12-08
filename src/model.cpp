#include "model.h"

Model::~Model() {
    deleteData();
}

void Model::addData(const Mesh& mesh, const std::vector<int>& vboDimensions) {
    genVAO();

    addVBO(vboDimensions, mesh.vertices);
    addEBO(mesh.indices);
    glBindVertexArray(0);
}

void Model::deleteData() {
    if (m_renderInfo.vao)
        glDeleteVertexArrays(1, &m_renderInfo.vao);
    if (m_buffers.size() > 0)
        glDeleteBuffers(static_cast<GLsizei>(m_buffers.size()),
                        m_buffers.data());

    m_buffers.clear();

    m_vboCount = 0;
    m_renderInfo.vao = 0;
    m_renderInfo.indicesCount = 0;
}

void Model::genVAO() {
    if (m_renderInfo.vao != 0)
        deleteData();

    glGenVertexArrays(1, &m_renderInfo.vao);
    glBindVertexArray(m_renderInfo.vao);
}

void Model::addEBO(const std::vector<GLuint> &indices) {
    m_renderInfo.indicesCount = static_cast<GLuint>(indices.size());
    GLuint ebo;
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint),
                 indices.data(), GL_STATIC_DRAW);
}

void Model::addVBO(const std::vector<int>& dimensions, const std::vector<GLfloat> &data){
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(GLfloat), data.data(),
                 GL_STATIC_DRAW);

    unsigned dimSum = 0;
    for (auto& n : dimensions)
        dimSum += n;

    unsigned accumulator = 0;
    for (int dimension : dimensions) {
        glVertexAttribPointer(static_cast<GLuint>(m_vboCount), dimension, GL_FLOAT,
                          GL_FALSE, dimSum*sizeof(GL_FLOAT), (void*)(accumulator*sizeof(GL_FLOAT)));

        glEnableVertexAttribArray(static_cast<GLuint>(m_vboCount++));

        accumulator += dimension;
    }
    m_buffers.push_back(vbo);
}

void Model::bind() const{
    glBindVertexArray(m_renderInfo.vao);
}

const RenderInfo& Model::getRenderInfo() const {
    return m_renderInfo;
}