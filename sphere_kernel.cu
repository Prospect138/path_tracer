#include "sphere_kernel.cuh"

// #include <__clang_cuda_builtin_vars.h>
#include <cstdio>
#include <cuda_runtime.h>

__device__ bool Hit(const Sphere *s, const Ray &r, double ray_tmin, double ray_tmax, HitRecord &record)
{
    vec3 relative_center = r.origin() - s->_center;
    vec3 normalized_dir = unit_vector(r.direction());
    double a = dot_product(normalized_dir, normalized_dir);
    double h = dot_product(normalized_dir, (relative_center));
    double c = dot_product(relative_center, relative_center) - s->_radius * s->_radius;
    // double discriminant = b * b - 4 * a * c; //оптимизируется
    double discriminant = h * h - a * c;
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
    return true;
}

__device__ color SphereDrawPoint(const Sphere *s, size_t sz, const Ray &r)
{
    HitRecord prev_record{};
    prev_record.t = 1e12;
    HitRecord record{};
    int hit_index = -1;
    for (size_t i = 0; i < sz; i++)
    {
        if (Hit(&s[i], r, 0.0, 100.0, record))
        {
            record = fmin(record.t, prev_record.t) == record.t ? record : prev_record;
            prev_record = record;
            hit_index = i;
        }
    }
    if (hit_index != -1)
        return s[hit_index]._color;
    return color(50.0, 25.0, 200.0);
}