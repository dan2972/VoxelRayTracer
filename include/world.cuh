#ifndef WORLD_H
#define WORLD_H

#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "vec3.cuh"
#include "bbox.cuh"

struct OctreeNode {
    bool isAir = false;
    bool emissive = false;
    unsigned char r, g, b, childMask;
    BBox bounds;
    OctreeNode* children[8];

    __device__ OctreeNode(unsigned char childMask, BBox bounds) 
        : childMask(childMask), bounds(bounds) {}
    __device__ OctreeNode(Vec3 rgb, BBox bounds) 
        : r(rgb[0] * 255), g(rgb[1] * 255), b(rgb[2] * 255), childMask(0), bounds(bounds) {}
};

struct HitRecord {
    bool emissive;
    float t;
    Vec3 position;
    Vec3 normal;
    Vec3 color;
};

class World {
public:
    __device__ World(BBox bounds, int maxDepth) : m_worldBounds(bounds), m_maxDepth(maxDepth) {}

    __device__ ~World() {
        if (m_octree != nullptr) {
            OctreeNode* stack[64];
            unsigned int stackPtr = 0;
            stack[stackPtr++] = m_octree;
            while (stackPtr > 0) {
                OctreeNode* curNode = stack[--stackPtr];
                for (int i = 0; i < 8; ++i) {
                    if (curNode->childMask & (1 << i)) {
                        stack[stackPtr++] = curNode->children[i];
                    }
                }
                delete curNode;
            }
        }
    }

    __device__ void insert(Vec3 position, Vec3 rgb, bool isAir = false, bool emissive = false) {
        if (m_octree == nullptr) {
            m_octree = new OctreeNode{0, m_worldBounds};
        }

        OctreeNode* curNode = m_octree;
        for (int d = 0; d < m_maxDepth; ++d) {
            OctreeNode* node = curNode;

            if (d == m_maxDepth - 1) {
                node->r = rgb[0] * 255;
                node->g = rgb[1] * 255;
                node->b = rgb[2] * 255;
                node->isAir = isAir;
                node->emissive = emissive;
                return;
            }

            Vec3 center = node->bounds.getCenter();
            for (int i = 0; i < 8; ++i) {
                Vec3 corner = node->bounds.getCorner(i);
                Vec3 newMin = Vec3(min(center.x(), corner.x()), min(center.y(), corner.y()), min(center.z(), corner.z()));
                Vec3 newMax = Vec3(max(center.x(), corner.x()), max(center.y(), corner.y()), max(center.z(), corner.z()));

                BBox newBounds(newMin, newMax);
                if (newBounds.contains(position)) {
                    if (node->childMask & (1 << i)) {
                        curNode = node->children[i];
                    } else {
                        node->childMask |= 1 << i;
                        node->children[i] = new OctreeNode{0, newBounds};
                        curNode = node->children[i];
                    }
                }
            }
        }
    }

    __device__ bool rayIntersect(const Ray& ray, float rayTMin, float rayTMax, HitRecord& hitRecord) {
        if (m_octree == nullptr) return false;
        OctreeNode* stack[64];
        unsigned int stackPtr = 0;

        float nearestT = FLT_MAX;
        bool hit = false;

        if (!m_worldBounds.rayIntersects(ray))
            return false;
        stack[stackPtr++] = m_octree;
        while (stackPtr > 0) {
            OctreeNode* curNode = stack[--stackPtr];
            
            for (int i = 0; i < 8; ++i) {
                if (curNode->childMask & (1 << i)) {
                    OctreeNode* child = curNode->children[i];
                    
                    float nearT, farT;
                    if (child->bounds.rayIntersects(ray, rayTMin, rayTMax, nearT, farT) && nearT < nearestT) {
                        if (child->childMask == 0 && !child->isAir) {
                            hitRecord.t = nearT;
                            hitRecord.position = ray.at(nearT);
                            hitRecord.normal = child->bounds.getNormal(hitRecord.position);
                            hitRecord.color = Vec3(child->r / 255.0f, child->g / 255.0f, child->b / 255.0f);
                            hitRecord.emissive = child->emissive;
                            nearestT = nearT;
                            hit = true;
                        } else {
                            stack[stackPtr++] = child;
                        }
                    }
                }
            }
        }
        return hit;
    }

    // returns the length of the world along one of its dimensions at its highest resolution
    __device__ int getSize() {
        return pow(2, m_maxDepth - 1);
    }

private:
    OctreeNode* m_octree = nullptr;
    BBox m_worldBounds;
    int m_maxDepth;
};

#endif // WORLD_H