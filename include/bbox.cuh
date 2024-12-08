#ifndef BBOX_H
#define BBOX_H

#include <algorithm>
#include <cuda_runtime.h>
#include "vec3.cuh"
#include "ray.cuh"

class BBox {
public :
    BBox() = default;
    __host__ __device__ BBox(Vec3 min, Vec3 max) : m_min(min), m_max(max) {}

    __host__ __device__ void set(Vec3 min, Vec3 max) { m_min = min; m_max = max; }
    __host__ __device__ void setMin(Vec3 min) { m_min = min; }
    __host__ __device__ void setMax(Vec3 max) { m_max = max; }

    __host__ __device__ Vec3 getCenter() const { return (m_min + m_max) * 0.5; }
    __host__ __device__ Vec3 getMin() const { return m_min; }
    __host__ __device__ Vec3 getMax() const { return m_max; }

    __host__ __device__ Vec3 getCorner(int index) const {
        Vec3 corner;
        for (int i = 0; i < 3; i++) {
            corner[i] = (index & (1 << i)) ? m_max[i] : m_min[i];
        }
        return corner;
    }

    __host__ __device__ bool overlaps(const BBox& other) const {
        return (m_min.x() <= other.m_max.x() && m_max.x() >= other.m_min.x()) &&
               (m_min.y() <= other.m_max.y() && m_max.y() >= other.m_min.y()) &&
               (m_min.z() <= other.m_max.z() && m_max.z() >= other.m_min.z());
    }

    __host__ __device__ bool contains(const Vec3& point) const {
        return (point.x() >= m_min.x() && point.x() <= m_max.x() &&
                point.y() >= m_min.y() && point.y() <= m_max.y() &&
                point.z() >= m_min.z() && point.z() <= m_max.z());
    }

    __device__ bool rayIntersects(const Ray& r) const {
        float nearT = -FLT_MAX;
        float farT = FLT_MAX;

        for (int i=0; i<3; i++) {
            float origin = r.origin[i];
            float minVal = m_min[i], maxVal = m_max[i];

            if (r.direction[i] == 0) {
                if (origin < minVal || origin > maxVal)
                    return false;
                continue;
            } else {
                float invDir = 1.0f / r.direction[i];
                float t1 = (minVal - origin) * invDir;
                float t2 = (maxVal - origin) * invDir;

                if (t1 > t2) {
                    float temp = t1;
                    t1 = t2;
                    t2 = temp;
                }

                nearT = fmax(t1, nearT);
                farT = fmin(t2, farT);

                if (nearT > farT)
                    return false;
            }
        }

        return true;
    }

    __device__ bool rayIntersects(const Ray& r, float& nearT, float& farT) const {
        nearT = -FLT_MAX;
        farT = FLT_MAX;

        for (int i=0; i<3; i++) {
            float origin = r.origin[i];
            float minVal = m_min[i], maxVal = m_max[i];

            if (r.direction[i] == 0) {
                if (origin < minVal || origin > maxVal)
                    return false;
                continue;
            } else {
                float invDir = 1.0f / r.direction[i];
                float t1 = (minVal - origin) * invDir;
                float t2 = (maxVal - origin) * invDir;

                if (t1 > t2) {
                    float temp = t1;
                    t1 = t2;
                    t2 = temp;
                }

                nearT = fmax(t1, nearT);
                farT = fmin(t2, farT);

                if (nearT > farT)
                    return false;
            }
        }

        return true;
    }

    __device__ bool rayIntersects(const Ray& r, float rayMinT, float rayMaxT, float& nearT, float& farT) const {
        nearT = -FLT_MAX;
        farT = FLT_MAX;

        for (int i=0; i<3; i++) {
            float origin = r.origin[i];
            float minVal = m_min[i], maxVal = m_max[i];

            if (r.direction[i] == 0) {
                if (origin < minVal || origin > maxVal)
                    return false;
                continue;
            } else {
                float invDir = 1.0f / r.direction[i];
                float t1 = (minVal - origin) * invDir;
                float t2 = (maxVal - origin) * invDir;

                if (t1 > t2) {
                    float temp = t1;
                    t1 = t2;
                    t2 = temp;
                }

                nearT = fmax(t1, nearT);
                farT = fmin(t2, farT);

                if (nearT > farT)
                    return false;
            }
        }

        return rayMinT <= farT && nearT <= rayMaxT;
    }

    __host__ __device__ Vec3 getNormal(const Vec3& point) const {

        float dLeft = point.x() - m_min.x();
        float dRight = m_max.x() - point.x();
        float dBottom = point.y() - m_min.y();
        float dTop = m_max.y() - point.y();
        float dBack = point.z() - m_min.z();
        float dFront = m_max.z() - point.z();

        float minDist = min(min(dLeft, dRight), min(min(dBottom, dTop), min(dBack, dFront)));

        Vec3 normal;

        if (minDist == dLeft) {
            normal = Vec3(-1, 0, 0);
        } else if (minDist == dRight) {
            normal = Vec3(1, 0, 0);
        } else if (minDist == dBottom) {
            normal = Vec3(0, -1, 0);
        } else if (minDist == dTop) {
            normal = Vec3(0, 1, 0);
        } else if (minDist == dBack) {
            normal = Vec3(0, 0, -1);
        } else {
            normal = Vec3(0, 0, 1);
        }

        return normal;
    }

private:
    Vec3 m_min, m_max;
};



#endif // BBOX_H