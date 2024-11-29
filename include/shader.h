#ifndef SHADER_H
#define SHADER_H

#include <string>
#include <glad/glad.h>

class Shader
{
public:
    Shader() { }

    Shader &use();

    void compile(const char *vertexSource, const char *fragmentSource, const char *geometrySource = nullptr);

    unsigned int ID; 
private:
    void checkCompileErrors(unsigned int object, std::string type); 
};

#endif