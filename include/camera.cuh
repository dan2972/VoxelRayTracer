#ifndef CAMERAH
#define CAMERAH

#include <cuda_runtime.h>
#include "ray.cuh"
#include "constants.h"

class Camera {
public:
    Camera() = default;
    // vfov is top to bottom in degrees
    __device__ Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov, float aspect) : vfov(vfov), aspect(aspect) {
        worldUp = unitVector(vup);
        update(lookfrom, lookat, vup);
    }

    __device__ void processMouseMovement(float xoffset, float yoffset) {
        xoffset *= mouseSensitivity;
        yoffset *= mouseSensitivity;

        yaw += xoffset;
        pitch += yoffset;

        if (pitch > 89.9f) pitch = 89.9f;
        if (pitch < -89.9f) pitch = -89.9f;

        front = Vec3(0, 0, 0);
        front[0] = cos(yaw * PI / 180) * cos(pitch * PI / 180);
        front[1] = sin(pitch * PI / 180);
        front[2] = sin(yaw * PI / 180) * cos(pitch * PI / 180);
        front = unitVector(front);
        right = unitVector(cross(front, worldUp));
        up = unitVector(cross(right, front));
        update(origin, origin + front, up);
    }

    // direction is a Vec3 where x, y, z are 1, 0, or -1
    __device__ void processKeyboardMovement(Vec3 direction, float deltaTime) {
        Vec3 forward = unitVector(Vec3(front.x(), 0, front.z())) * movementSpeed * deltaTime;
        
        if (direction[0] != 0) {
            origin += direction[0] * right * movementSpeed * deltaTime;
        }
        if (direction[1] != 0) {
            origin += direction[1] * worldUp * movementSpeed * deltaTime;
        }
        if (direction[2] != 0) {
            origin[0] += direction[2] * forward[0];
            origin[2] += direction[2] * forward[2];
        }
        update(origin, origin + front, up);
    }

    __device__ void update(Vec3 lookfrom, Vec3 lookat, Vec3 vup) {
        Vec3 u, v, w;
        float theta = vfov*PI/180;
        float half_height = tan(theta/2);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unitVector(lookfrom - lookat);
        u = unitVector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin - half_width*u -half_height*v - w;
        horizontal = 2*half_width*u;
        vertical = 2*half_height*v;
    }

    __device__ Ray getRay(float u, float v) { return Ray(origin, lower_left_corner + u*horizontal + v*vertical - origin); }

    Vec3 origin;
    Vec3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    Vec3 front;
    Vec3 right;
    Vec3 up;
    Vec3 worldUp;
    float yaw=-90, pitch=0;
    float vfov, aspect;
    float movementSpeed = 10.0f;
    float mouseSensitivity = 0.05f;
};

#endif