#pragma once

#include "common/vec3.h"
#include "ray.h"
#include "hittable.h"

struct Sphere
{
    point3 _center;
    point3 _color;
    real_t _radius;
    real_t _reflectivity;


    __device__ bool Hit(const Ray &r, real_t ray_tmin, real_t ray_tmax, HitRecord &record) const
    {
        vec3 relative_center = _center - r.origin();
        vec3 dir = r.direction();
        real_t a = dot_product(dir, dir);
        real_t h = dot_product(dir, relative_center);
        real_t c = dot_product(relative_center, relative_center) - _radius * _radius;
        // real_t discriminant = b * b - 4 * a * c; //оптимизируется
        real_t discriminant = h * h - a * c;
        if (discriminant < 0)
            return false;
        //(-b - std::sqrt(discriminant)) * 0.5 / a;
        auto sqrtd = std::sqrt(discriminant);
        // Find the nearest root that lies in the acceptable range.
        auto root = (h - sqrtd) / a;
        if (root <= ray_tmin || ray_tmax <= root)
        {
            root = (h + sqrtd) / a;
            if (root <= ray_tmin || ray_tmax <= root)
                return false;
        }
        record.t = root;
        record.p = r.lerp(record.t);
        record.set_face_normal(r, unit_vector((record.p - _center) / _radius));
        return true;
    }
};

/*
struct ShapeContainer {
    int type_id;
    union {
        Sphere s;
        Plane p;
        Triangle t;
    } data;

    template <typename F>
    __device__ void dispatch(F&& func) const {
        if (type_id == 0) func(data.s);
        else if (type_id == 1) func(data.p);
    }
};
*/