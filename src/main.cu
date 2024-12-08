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
#include "camera.cuh"
#include "raytracer.cuh"
#include "bbox.cuh"
#include "world.cuh"

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        // glfwSetWindowShouldClose(window, GLFW_TRUE);
        if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_NORMAL)
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        else
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }
}

int main() {
    const int width = 1920;
    const int height = 1080;

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

    glfwSetKeyCallback(window, keyCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    // glfwSwapInterval(0); // Disable VSync

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
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaResource, screenQuadTexture.ID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

    // Allocate CUDA memory once
    uchar4* devPixels;
    checkCudaErrors(cudaMalloc(&devPixels, width * height * sizeof(uchar4)));

    // FPS Counter Variables
    auto lastTime = std::chrono::high_resolution_clock::now();
    int frames = 0;
    
    Camera** camera;
    curandState *dRandState;
    World** dWorld;
    checkCudaErrors(cudaMalloc((void **)&camera, sizeof(Camera *)));
    checkCudaErrors(cudaMalloc((void **)&dRandState, width*height*sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void **)&dWorld, sizeof(World *)));

    create_world<<<1, 1>>>(dWorld, camera, width, height);
    checkCudaErrors(cudaGetLastError());

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    renderInit<<<gridSize, blockSize>>>(width, height, dRandState);
    checkCudaErrors(cudaGetLastError());

    std::cout << "Starting main loop" << std::endl;

    double lastxpos, lastypos;
    glfwGetCursorPos(window, &lastxpos, &lastypos);
    int prevMouseClickState = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);

    // Main Loop
    while (!glfwWindowShouldClose(window)) {

        Vec3 cameraDeltaPos(0,0,0);
        Vec3 cameraDeltaRotation(0,0,0);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            cameraDeltaPos[0] += 1;
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            cameraDeltaPos[0] -= 1;
        }
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            cameraDeltaPos[1] += 1;
        }
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
            cameraDeltaPos[1] -= 1;
        }
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            cameraDeltaPos[2] += 1;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            cameraDeltaPos[2] -= 1;
        }

        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        float xoffset = xpos - lastxpos;
        float yoffset = lastypos - ypos;
        lastxpos = xpos;
        lastypos = ypos;

        int mouseClickState = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
        if (mouseClickState == GLFW_PRESS && prevMouseClickState == GLFW_RELEASE) {
            placeBlock<<<1, 1>>>(camera, dWorld, 20);
        }
        prevMouseClickState = mouseClickState;

        controlCamera<<<1, 1>>>(camera, xoffset, yoffset, cameraDeltaPos);
        
        // Map CUDA Resource
        cudaArray* textureArray;
        checkCudaErrors(cudaGraphicsMapResources(1, &cudaResource, 0));
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&textureArray, cudaResource, 0, 0));
        
        render<<<gridSize, blockSize>>>(devPixels, width, height, camera, dWorld, dRandState);
        checkCudaErrors(cudaMemcpyToArray(textureArray, 0, 0, devPixels, width * height * sizeof(uchar4), cudaMemcpyDeviceToDevice));

        checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaResource, 0));

        // Render
        glClear(GL_COLOR_BUFFER_BIT);
        screen_quad_shader.use();
        screenQuad.bind();
        screenQuadTexture.bind();
        glDrawElements(GL_TRIANGLES, screenQuad.getRenderInfo().indicesCount, GL_UNSIGNED_INT, 0);

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
    checkCudaErrors(cudaFree(devPixels));

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
