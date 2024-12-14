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

__device__ __constant__ float dKernel[25] = {
    .0030f, .0133f, .0219f, .0133f, .0030f,
    .0133f, .0596f, .0983f, .0596f, .0133f,
    .0219f, .0983f, .1621f, .0983f, .0219f,
    .0133f, .0596f, .0983f, .0596f, .0133f,
    .0030f, .0133f, .0219f, .0133f, .0030f
};

__device__ __constant__ float3 blockPlacements[24] = {
    {-2.5, -30.5, -0.5},
    {-2.5, -29.5, -0.5},
    {-2.5, -30.5, 0.5},
    {-1.5, -30.5, -0.5},
    {2.736012, -30.500000, -2.04484},
    {1.933339, -30.500000, -2.17305},
    {0.986300, -30.500000, -2.25940},
    {1.466906, -30.500000, -1.96533},
    {2.520714, -30.500000, -1.92209},
    {3.471348, -30.500000, -2.45532},
    {3.444158, -30.500000, -1.88442},
    {3.495170, -30.899950, -0.50000},
    {2.780937, -30.831198, -0.50000},
    {3.473401, -30.602715, 0.500000},
    {2.496290, -29.500000, -2.39553},
    {3.310617, -29.500000, -2.32846},
    {3.447140, -29.500000, -1.40007},
    {2.790129, -29.500000, -1.62536},
    {1.602111, -29.500000, -2.37293},
    {3.492708, -29.500000, -0.87358},
    {3.369594, -28.500000, -2.27475},
    {2.820399, -28.500000, -2.41235},
    {3.466890, -28.972002, -1.50000},
    {3.375440, -27.500000, -2.13759}
};

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

    // Create Sphere
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

    //groud surface
    for (int z=0; z < worldSize; z++) {
        for (int x=0; x < worldSize; x++) {
            Vec3 voxelPos = worldBounds.getMin() + Vec3(x, 0, z) * (bboxSize / worldSize) + VoxelOffset;
            (*world)->insert(voxelPos, {1,1,1});
        }
    }
    // initial block placements
    for (int i = 0; i < 24; ++i) {
        (*world)->insert({blockPlacements[i].x, blockPlacements[i].y, blockPlacements[i].z}, {1,1,1});
    }
    (*world)->insert({-2.5, -30.5, 2.5}, {1, 1, 0.3});
    (*world)->insert({-0.5, -30.5, 2.5}, {0.3, 1, 1});
    (*world)->insert({1.5, -30.5, 2.5}, {1, 0.3, 1});
    (*world)->insert({3.5, -30.5, 2.5}, {0,0,0});

    // cornell box size
    float boxSize = 13.0f;
    // box walls
    for (int y = 0; y < boxSize; ++y) {
        for (int x = 0; x < boxSize; ++x) {
            (*world)->insert({-boxSize / 2 + 1 + x, -31.5f + y, -boxSize / 2 + 1}, {1, 1, 1});
            // (*world)->insert({-boxSize / 2 + 1 + x, -31.5f + y, boxSize / 2}, {1, 1, 1});
            (*world)->insert({-boxSize / 2 + 1, -31.5f + y, -boxSize / 2 + 1 + x}, {1,0.3,0.3});
            (*world)->insert({boxSize / 2, -31.5f + y, -boxSize / 2 + 1 + x}, {0.3,1,0.3});
        }
    }
    // box ceiling
    for (int z = 0; z < boxSize; ++z) {
        for (int x = 0; x < boxSize; ++x) {
            if (x < boxSize/4-1 || x > 3*boxSize/4 || z < boxSize/4-1 || z > 3*boxSize/4)
                (*world)->insert({-boxSize / 2 + 1 + x, -31.5f + boxSize - 1, -boxSize / 2 + 1 + z}, {1, 1, 1});
            else
                (*world)->insert({-boxSize / 2 + 1 + x, -31.5f + boxSize - 2, -boxSize / 2 + 1 + z}, {1, 1, 1}, false, true);
        }
    }
    
    *cam = new Camera(Vec3(0.5, -32 + boxSize / 2, boxSize / 2+2), Vec3(0, 0, 0), Vec3(0, 1, 0), 90, (float)width / (float)height);
}

__device__ void rayTrace(World** world, const Ray& ray, curandState *randState, Vec3& color, Vec3& position, Vec3& normal, bool enableSky = false) {
    Ray curRay = ray;
    Vec3 curAttenuation(1, 1, 1);
    position = curRay.at(100000.0f);
    normal = -unitVector(curRay.direction);
    for (int i = 0; i < 3; ++i) {
        HitRecord hitRecord;
        if ((*world)->rayIntersect(curRay, 0.001f, FLT_MAX, hitRecord)) {
            Ray scattered;
            Vec3 target = hitRecord.position + hitRecord.normal + random_in_unit_sphere(randState);
            scattered = Ray(hitRecord.position, target - hitRecord.position);
            curAttenuation *= hitRecord.color;
            curRay = scattered;
            if (i == 0) {
                position = hitRecord.position;
                normal = unitVector(hitRecord.normal);
            }
            if (hitRecord.emissive) {
                color = curAttenuation;
                return;
            }
        } else {
            Vec3 unitDirection = unitVector(curRay.direction);
            float t = 0.5f * (unitDirection.y() + 1.0f);
            Vec3 c;
            if (enableSky)
                c = (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
            else
                c = Vec3(0,0,0);
            color = curAttenuation * c;
            return;
        }
    }

    color = Vec3(0, 0, 0);
}

__global__ void render(
    Vec3* colors, Vec3* positions, Vec3* normals, int width, int height, 
    Camera** cam, World** world, curandState* randState, bool enableSky = false) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height) return;
    int pixelIndex = y * width + x;

    curandState localRandState = randState[pixelIndex];
    Vec3 col(0, 0, 0);
    float u = float(x) / float(width);
    float v = float(y) / float(height);

    Vec3 color, position, normal;
    int ssp = 1;
    for (int s = 0; s < ssp; ++s) {
        u = float(x + curand_uniform(&localRandState)) / float(width);
        v = float(y + curand_uniform(&localRandState)) / float(height);
        // u = float(x) / float(width);
        // v = float(y) / float(height);
        Ray r = (*cam)->getRay(u, v);
        rayTrace(world, r, &localRandState, color, position, normal, enableSky);
        col += color;
    }
    col /= ssp;
    colors[pixelIndex] = col;
    positions[pixelIndex] = position;
    normals[pixelIndex] = normal;
    
    randState[pixelIndex] = localRandState;
}

__global__ void finalizeRender(uchar4* pixels, Vec3* colors, Camera** cam, int width, int height, bool showCrosshair) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height) return;
    int pixelIndex = y * width + x;

    Vec3 col = colors[pixelIndex];
    float u = float(x) / float(width);
    float v = float(y) / float(height);

    col[0] = std::sqrt(col[0]);
    col[1] = std::sqrt(col[1]);
    col[2] = std::sqrt(col[2]);

    col[0] = min(max(col[0], 0.0f), 1.0f);
    col[1] = min(max(col[1], 0.0f), 1.0f);
    col[2] = min(max(col[2], 0.0f), 1.0f);

    if (showCrosshair) {
        float cu = 0.5, cv = 0.5;
        float crosshairWidth = 0.001;
        float crosshairSize = 0.01f;
        if ((abs(u-cu) * (*cam)->aspect < crosshairWidth || abs(v-cv) < crosshairWidth) 
                && (abs(u-cu) * (*cam)->aspect + abs(v-cv) < crosshairSize)) {
            col = Vec3(1, 1, 1) - col;
        }
    }

    pixels[pixelIndex] = make_uchar4(255 * col.x(), 255 * col.y(), 255 * col.z(), 255);
}

__global__ void atrousDenoiser(
    Vec3* output, Vec3* colors, Vec3* positions, Vec3* normals, 
    float cPhi, float nPhi, float pPhi, 
    int stepwidth, int width, int height) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixelIndex = y * width + x;
    Vec3 color = colors[pixelIndex];
    Vec3 position = positions[pixelIndex];
    Vec3 normal = normals[pixelIndex];

    Vec3 sum = Vec3(0, 0, 0);
    float sumWeight = 0.0f;

    for (int i = -2; i <= 2; ++i) {
        for (int j = -2; j <= 2; ++j) {
            int qx = x + j * stepwidth;
            int qy = y + i * stepwidth;
            
            if (qx < 0 || qx >= width || qy < 0 || qy >= height) continue;

            int nIndex = qy * width + qx;

            Vec3 cTemp = colors[nIndex];
            Vec3 nTemp = normals[nIndex];
            Vec3 pTemp = positions[nIndex];

            Vec3 cDiff = color - cTemp;
            float cDist2 = cDiff.squaredLength();
            float cWeight = min(1.0f, exp(-cDist2 / (cPhi+FLT_EPSILON)));

            Vec3 nDiff = normal - nTemp;
            float nDist2 = nDiff.squaredLength();
            float nWeight = min(1.0f, exp(-nDist2 / (nPhi+FLT_EPSILON)));

            Vec3 pDiff = position - pTemp;
            float pDist2 = pDiff.squaredLength();
            float pWeight = min(1.0f, exp(-pDist2 / (pPhi+FLT_EPSILON)));

            float weight = cWeight * nWeight * pWeight * dKernel[(i+2)*5 + j+2];
            sum += cTemp * weight;
            sumWeight += weight;
        }
    }

    output[pixelIndex] = sumWeight < FLT_EPSILON ? color : sum / sumWeight;
    Vec3 posC = (positions[pixelIndex] + Vec3(8, 8, 8))/ 16.0f;
    Vec3 normC = (normals[pixelIndex] + Vec3(1, 1, 1)) / 2.0f;
    // output[pixelIndex] = sum / sumWeight;
}

void denoiseImage(
    Vec3* output, Vec3* colors, Vec3* positions, Vec3* normals, 
    float cPhi, float nPhi, float pPhi, 
    int width, int height, const dim3& gridSize, const dim3& blockSize) 
{

    int stepwidth = 1;
    int filterSize = 5;
    while(filterSize < 60) {
        atrousDenoiser<<<gridSize, blockSize>>>(
            output, colors, positions, normals, cPhi, nPhi, pPhi, stepwidth, width, height);
        checkCudaErrors(cudaGetLastError());
        colors = output;
        cPhi /= 2.0f;
        stepwidth++;
        filterSize = (5-1) * stepwidth + 1;
    }
}

__global__ void placeBlock(Camera** cam, World** world, float maxDistance, bool emissive = false) {
    Ray centerRay = (*cam)->getRay(0.5, 0.5);
    HitRecord hitRecord;
    if ((*world)->rayIntersect(centerRay, 0.001f, maxDistance, hitRecord)) {
        (*world)->insert(hitRecord.position + hitRecord.normal * 0.1f, {1, 1, 1}, false, emissive);
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