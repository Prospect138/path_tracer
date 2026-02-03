#include "hittable.h"
#include "sphere_kernel.cuh"
#include "vec3.h"

// #include <__clang_cuda_builtin_vars.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <sys/types.h>

__device__ static int MAX_REFLECTIONS = 5;

__device__ vec3 reflectedDir(const vec3 &vector, const vec3 &normal)
{
    return vector - normal * 2.0 * dot_product(vector, normal);
}

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
    record.p = r.lerp(record.t);
    record.set_face_normal(r, unit_vector((record.p - s->_center) / s->_radius));
    return true;
}

__device__ color SphereDrawPoint(const Sphere *s, size_t sz, const Ray &r)
{
    Ray current_ray = r;
    color final_color(0.0, 0.0, 0.0);
    int depth = 0;
    double attenuation = 1.0;
    while (depth < MAX_REFLECTIONS)
    {
        HitRecord best_record{};
        best_record.t = 1e12;
        int hit_index = -1;
        for (size_t i = 0; i < sz; i++)
        {
            HitRecord record{};
            if (Hit(&s[i], current_ray, 0.001, 100.0, record))
            {
                if (best_record.t > record.t)
                {
                    best_record = record;
                    hit_index = i;
                }
            }
        }

        if (hit_index != -1)
        {
            final_color += attenuation * s[hit_index]._color * (1.0 - s[hit_index]._reflectivity);
            vec3 reflected_direction = reflectedDir(current_ray.direction(), best_record.normal);
            Ray reflected_ray(best_record.p, reflected_direction);
            current_ray = reflected_ray;
            attenuation *= s[hit_index]._reflectivity;
            if (attenuation < 0.01)
                break;
        }
        else if (hit_index == -1)
            break;
        depth++;
    }
    return final_color;
}