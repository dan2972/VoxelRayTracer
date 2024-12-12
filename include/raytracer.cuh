#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <iostream>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>
#include "camera.cuh"
#include "ray.cuh"
#include "vec3.cuh"
#include "world.cuh"

#define checkCudaErrors(val) checkCuda( (val), #val, __FILE__, __LINE__ )

void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void controlCamera(Camera** cam, float xoffset, float yoffset, Vec3 direction, float deltaTime) {
    (*cam)->processMouseMovement(xoffset, yoffset);
    (*cam)->processKeyboardMovement(direction, deltaTime);
}

__global__ void renderInit(int maxX, int maxY, curandState *randState) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= maxX || y >= maxY) return;

    int idx = y * maxX + x;
    curand_init(1984, idx, 0, &randState[idx]);
}

#define RANDVEC3 Vec3(curand_uniform(randState),curand_uniform(randState),curand_uniform(randState))

__device__ Vec3 random_in_unit_sphere(curandState *randState) {
    Vec3 p;
    do {
        p = 2.0f*RANDVEC3 - Vec3(1,1,1);
    } while (p.squaredLength() >= 1.0f);
    return p;
}

__global__ void create_world(World** world, Camera** cam, int width, int height) {
    BBox worldBounds(Vec3(-32, -32, -32), Vec3(32, 32, 32));
    *world = new World(worldBounds, 7);
    int worldSize = (*world)->getSize();
    
    Vec3 bboxSize = worldBounds.getMax() - worldBounds.getMin();
    Vec3 VoxelOffset = Vec3(0.5, 0.5, 0.5) * (bboxSize / worldSize);
    // int offset = 16;
    // for (int z=offset; z < offset+worldSize/4; z++) {
    //     for (int y=offset; y < offset+worldSize/4; y++) {
    //         for (int x=offset; x < offset+worldSize/4; x++) {

    //             Vec3 voxelPos = worldBounds.getMin() + Vec3(x, y, z) * (bboxSize / worldSize) + VoxelOffset;

    //             float distance = (voxelPos - worldBounds.getCenter()).length();

    //             if (distance < 7.5) {
    //                 (*world)->insert(voxelPos, {(float)x/worldSize, (float)y/worldSize, (float)z/worldSize});
    //             }
    //         }
    //     }
    // }

    for (int z=0; z < worldSize; z++) {
        for (int x=0; x < worldSize; x++) {
            Vec3 voxelPos = worldBounds.getMin() + Vec3(x, 0, z) * (bboxSize / worldSize) + VoxelOffset;
            (*world)->insert(voxelPos, {1,1,1});
        }
    }
    (*world)->insert({-1.5, -30.5, 0.5}, {1,0.3,0.3});
    (*world)->insert({0.5, -30.5, 0.5}, {0.3,1,0.3});
    (*world)->insert({2.5, -30.5, 0.5}, {0.3,0.3,1});
    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 17; ++x) {
            (*world)->insert({-7.5f + x, -30.5f + y, -7.5f}, {1, 1, 1});
            if (x < 3 || x > 13)
                (*world)->insert({-7.5f + x, -30.5f + y, 8.5f}, {1, 1, 1});
            (*world)->insert({-7.5f, -30.5f + y, -7.5f + x}, {1, 1, 1});
            (*world)->insert({8.5f, -30.5f + y, -7.5f + x}, {1, 1, 1});
        }
    }
    for (int z = 0; z < 17; ++z) {
        for (int x = 0; x < 17; ++x) {
            if (z != 1)
                (*world)->insert({-7.5f + x, -25.5f, -7.5f + z}, {1, 1, 1});
        }
    }

    // float start = -16.0f;
    // float end = 16.0f;
    // float sphereSize = end - start;
    // for (float z=start + VoxelOffset[2]; z < end + VoxelOffset[2]; z+=(bboxSize/worldSize)[2]) {
    //     for (float y=start + VoxelOffset[1]; y < end + VoxelOffset[1]; y+=(bboxSize/worldSize)[1]) {
    //         for (float x=start + VoxelOffset[0]; x < end + VoxelOffset[0]; x+=(bboxSize/worldSize)[0]) {
    //             Vec3 voxelPos = Vec3(x, y, z);
    //             float distance = (voxelPos - worldBounds.getCenter()).length();
    //             if (distance < 7.5) {
    //                 (*world)->insert(voxelPos, {(x-start)/sphereSize, (y-start)/sphereSize, (z-start)/sphereSize});
    //             }
    //         }
    //     }
    // }
    
    *cam = new Camera(Vec3(-6, -27.5, 6), Vec3(0, 0, 0), Vec3(0, 1, 0), 90, (float)width / (float)height);
}

__device__ Vec3 color(World** world, const Ray& ray, curandState *randState) {
    Ray curRay = ray;
    Vec3 curAttenuation(1, 1, 1);
    for (int i = 0; i < 3; ++i) {
        HitRecord hitRecord;
        if ((*world)->rayIntersect(curRay, 0.001f, FLT_MAX, hitRecord)) {
            Ray scattered;
            Vec3 target = hitRecord.position + hitRecord.normal + random_in_unit_sphere(randState);
            scattered = Ray(hitRecord.position, target - hitRecord.position);
            curAttenuation *= hitRecord.color;
            curRay = scattered;
        } else {
            Vec3 unitDirection = unitVector(curRay.direction);
            float t = 0.5f * (unitDirection.y() + 1.0f);
            Vec3 c = (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
            return curAttenuation * c;
        }
    }

    return Vec3(0, 0, 0);
}

__global__ void render(uchar4* pixels, int width, int height, bool showCrosshair, Camera** cam, World** world, curandState* randState) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height) return;
    int pixelIndex = y * width + x;

    curandState localRandState = randState[pixelIndex];
    Vec3 col(0, 0, 0);
    float u = float(x) / float(width);
    float v = float(y) / float(height);

    int ssp = 1;
    for (int s = 0; s < ssp; ++s) {
        u = float(x + curand_uniform(&localRandState)) / float(width);
        v = float(y + curand_uniform(&localRandState)) / float(height);
        Ray r = (*cam)->getRay(u, v);
        col += color(world, r, &localRandState);
    }
    col /= ssp;

    // Center crosshair
    if (showCrosshair) {
        float cu = 0.5, cv = 0.5;
        float crosshairWidth = 0.001;
        float crosshairSize = 0.01f;
        if ((abs(u-cu) * (*cam)->aspect < crosshairWidth || abs(v-cv) < crosshairWidth) 
                && (abs(u-cu) * (*cam)->aspect + abs(v-cv) < crosshairSize)) {
            col = Vec3(1, 1, 1) - col;
        }
    }
    
    randState[pixelIndex] = localRandState;
    col[0] = std::sqrt(col[0]);
    col[1] = std::sqrt(col[1]);
    col[2] = std::sqrt(col[2]);
    pixels[pixelIndex] = make_uchar4(255 * col.x(), 255 * col.y(), 255 * col.z(), 255);
}

__global__ void placeBlock(Camera** cam, World** world, float maxDistance) {
    Ray centerRay = (*cam)->getRay(0.5, 0.5);
    HitRecord hitRecord;
    if ((*world)->rayIntersect(centerRay, 0.001f, maxDistance, hitRecord)) {
        (*world)->insert(hitRecord.position + hitRecord.normal * 0.1f, {1, 1, 1});
    }
}

__global__ void removeBlock(Camera** cam, World** world, float maxDistance) {
    Ray centerRay = (*cam)->getRay(0.5, 0.5);
    HitRecord hitRecord;
    if ((*world)->rayIntersect(centerRay, 0.001f, maxDistance, hitRecord)) {
        (*world)->insert(hitRecord.position + hitRecord.normal * -0.1f, {1, 1, 1}, true);
    }
}

class RayTracer {
public:
    RayTracer();
    ~RayTracer();

    void render();

    void registerScreenQuadTexture(unsigned int textureID) {
        
    }

private:

};

#endif // RAYTRACER_H