#pragma once

#include "vec3.h"

struct HitRecord {
    point3 p;
    vec3 normal;
    double t;
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