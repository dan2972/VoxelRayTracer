#ifndef TEXTURE_H
#define TEXTURE_H

#include <glad/glad.h>

class Texture2D
{
public:
    Texture2D();

    void generate(unsigned int width, unsigned int height, unsigned char* data);
    void bind() const;

    unsigned int ID;
    unsigned int width, height;
    unsigned int internalFormat;
    unsigned int imageFormat;
    unsigned int wrapS;
    unsigned int wrapT;
    unsigned int filterMin;
    unsigned int filterMax;
};

#endif