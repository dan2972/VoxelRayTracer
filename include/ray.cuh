#ifndef RAY_H
#define RAY_H

#include "vec3.cuh"

class Ray
{
    public:
        __device__ Ray() {}
        __device__ Ray(const Vec3& o, const Vec3& d) : origin(o), direction(d)  {}
        __device__ Vec3 at(float t) const { return origin + t*direction; }

        __device__ inline Ray& operator=(const Ray& r) {
            origin = r.origin;
            direction = r.direction;
            return *this;
        }

        __device__ Ray reflect(const Vec3& normal, const Vec3& hitPoint) const {
            return Ray(hitPoint, direction - 2 * dot(direction, normal) * normal);
        }

        Vec3 origin;
        Vec3 direction;
};

#endif