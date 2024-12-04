#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>
#include <iostream>
#include <chrono>
#include "shader.h"
#include "resource_manager.h"
#include "model.h"

__global__ void generateRandomPixels(uchar4* pixels, int width, int height, unsigned int seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;

        // Initialize curand state
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Generate random colors
        unsigned char r = curand(&state) % 256;
        unsigned char g = curand(&state) % 256;
        unsigned char b = curand(&state) % 256;

        pixels[idx] = make_uchar4(r, g, b, 255); // RGBA format
    }
}

// Check for CUDA Errors
void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int width = 1280;
    const int height = 720;

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(width, height, "CUDA + OpenGL Example", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    // Initialize Glad
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize Glad" << std::endl;
        return -1;
    }

    // OpenGL Quad Data
    std::vector<float> quadVertices = {
        // positions    // texCoords
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f,
        -1.0f,  1.0f,  0.0f, 1.0f
    };

    std::vector<unsigned int> quadIndices = {
        0, 1, 2,
        0, 2, 3
    };

    Model screenQuad;
    screenQuad.addData({quadVertices, quadIndices}, {2, 2});

    // Compile Shaders and Link Program
    Shader screen_quad_shader = ResourceManager::loadShader("res/shaders/basic_quad.vert", "res/shaders/basic_quad.frag", nullptr, "screen_quad");

    Texture2D screenQuadTexture;
    screenQuadTexture.generate(width, height, nullptr);
    ResourceManager::addTexture(screenQuadTexture, "screenQuadTexture");

    // Register OpenGL Texture with CUDA
    cudaGraphicsResource* cudaResource;
    checkCuda(cudaGraphicsGLRegisterImage(&cudaResource, screenQuadTexture.ID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard), "Registering OpenGL texture with CUDA");

    // Allocate CUDA memory once
    uchar4* devPixels;
    checkCuda(cudaMalloc(&devPixels, width * height * sizeof(uchar4)), "Allocating device memory");

    // FPS Counter Variables
    auto lastTime = std::chrono::high_resolution_clock::now();
    int frames = 0;

    // Main Loop
    while (!glfwWindowShouldClose(window)) {
        // Map CUDA Resource
        cudaArray* textureArray;
        checkCuda(cudaGraphicsMapResources(1, &cudaResource, 0), "Mapping CUDA resource");
        checkCuda(cudaGraphicsSubResourceGetMappedArray(&textureArray, cudaResource, 0, 0), "Getting mapped array");

        // Generate Random Pixels
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        unsigned int seed = static_cast<unsigned int>(frames);
        generateRandomPixels<<<gridSize, blockSize>>>(devPixels, width, height, seed);
        checkCuda(cudaMemcpyToArray(textureArray, 0, 0, devPixels, width * height * sizeof(uchar4), cudaMemcpyDeviceToDevice), "Copying pixels to texture");

        checkCuda(cudaGraphicsUnmapResources(1, &cudaResource, 0), "Unmapping CUDA resource");

        // Render
        glClear(GL_COLOR_BUFFER_BIT);
        screen_quad_shader.use();
        screenQuad.bind();
        screenQuadTexture.bind();
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();

        // Update FPS Counter
        frames++;
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = currentTime - lastTime;
        if (elapsed.count() >= 1.0) {
            std::cout << "FPS: " << frames << std::endl;
            frames = 0;
            lastTime = currentTime;
        }
    }

    // Cleanup
    cudaGraphicsUnregisterResource(cudaResource);
    checkCuda(cudaFree(devPixels), "Freeing device memory");

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
