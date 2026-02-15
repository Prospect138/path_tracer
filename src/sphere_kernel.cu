#include "camera.h"
#include "hittable.h"
#include "sphere_kernel.cuh"
#include "vec3.h"

// #include <__clang_cuda_builtin_vars.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <sys/types.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ static real_t SKY_R = 0.05;
__device__ static real_t SKY_G = 0.2;
__device__ static real_t SKY_B = 0.65;
__device__ static int MAX_REFLECTIONS = 5;

__device__ vec3 ReflectedDir(const vec3 &vector, const vec3 &normal)
{
    return vector - normal * 2.0 * dot_product(vector, normal);
}

__device__ bool Hit(const Sphere *s, const Ray &r, real_t ray_tmin, real_t ray_tmax, HitRecord &record)
{
    vec3 relative_center = s->_center - r.origin();
    vec3 dir = r.direction();
    real_t a = dot_product(dir, dir);
    real_t h = dot_product(dir, relative_center);
    real_t c = dot_product(relative_center, relative_center) - s->_radius * s->_radius;
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
    record.set_face_normal(r, unit_vector((record.p - s->_center) / s->_radius));
    return true;
}

__device__ void CorrectAndClamp(color &final_color)
{
    // Clamping
    final_color[0] = fminf(1.0f, fmaxf(0.0f, final_color.x()));
    final_color[1] = fminf(1.0f, fmaxf(0.0f, final_color.y()));
    final_color[2] = fminf(1.0f, fmaxf(0.0f, final_color.z()));
    // Gamma Correction
    final_color[0] = sqrt(final_color.x());
    final_color[1] = sqrt(final_color.y());
    final_color[2] = sqrt(final_color.z());
}

__device__ color SphereDrawPoint(const Sphere *s, const size_t o_sz, const LightSource *lights, const size_t l_sz,
                                 const Ray &r)
{
    Ray current_ray = r;
    color final_color(0.0, 0.0, 0.0);
    int depth = 0;
    real_t attenuation = 1.0;
    while (depth < MAX_REFLECTIONS)
    {
        HitRecord best_record{};
        best_record.t = 1e12;
        int hit_index = -1;

        // Find the nearest hit among all spheres
        for (size_t i = 0; i < o_sz; i++)
        {
            HitRecord record{};
            if (Hit(&s[i], current_ray, 0.001, 1e9, record))
            {
                if (best_record.t > record.t)
                {
                    best_record = record;
                    hit_index = i;
                }
            }
        }
        // If found hit
        if (hit_index != -1)
        {
            final_color += attenuation * s[hit_index]._color * (1.0 - s[hit_index]._reflectivity);
            // make reflected ray as current and iterate again
            vec3 reflected_direction = ReflectedDir(unit_vector(current_ray.direction()), best_record.normal);
            Ray reflected_ray(best_record.p + 0.001 * best_record.normal, reflected_direction);
            current_ray = reflected_ray;
            attenuation *= s[hit_index]._reflectivity;
            // if (attenuation < 0.01)
            //     break;
        }
        else if (hit_index == -1)
        {
            vec3 d = unit_vector(current_ray.direction());
            real_t t = 0.5 * (d.y() + 1.0);
            color sky = (1.0 - t) * color(1, 1, 1) + t * color(SKY_R, SKY_G, SKY_B);
            final_color += attenuation * sky;
            break;
        }
        depth++;
    }
    CorrectAndClamp(final_color);
    return final_color;
}