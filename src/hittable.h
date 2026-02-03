#pragma once

#include "ray.h"
#include "vec3.h"
struct HitRecord
{
    point3 p;
    vec3 normal;
    double t;
    bool front_face;
    __host__ __device__ void set_face_normal(const Ray &r, const vec3 &outward_normal)
    {
        front_face = dot_product(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};
/*
class Hittable
{
public:
    Hittable() = default;
    virtual ~Hittable() = default;
    virtual bool Hit(const Ray& r, double ray_tmin, double ray_tmax, HitRecord&
record) const = 0; virtual color DrawPoint(const Ray& r) const = 0;
};
*/