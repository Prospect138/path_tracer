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

__device__ static double SKY_R = 0.05;
__device__ static double SKY_G = 0.2;
__device__ static double SKY_B = 0.65;
__device__ static int MAX_REFLECTIONS = 5;

__device__ vec3 ReflectedDir(const vec3 &vector, const vec3 &normal)
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

__device__ color SphereDrawPoint(const Sphere *s, const size_t o_sz, const LightSource *lights, const size_t l_sz,
                                 const Ray &r)
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

        // Find the nearest hit among all spheres
        for (size_t i = 0; i < o_sz; i++)
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
        // If found hit
        if (hit_index != -1)
        {
            color local_color;
            // Find if hit is illuminated by light source
            for (size_t j = 0; j < l_sz; j++)
            {
                vec3 light_direction = unit_vector(lights[j]._coordinate - best_record.p);
                double light_distance = dist(lights[j]._coordinate, best_record.p);
                Ray light_seeker{best_record.p, light_direction};
                bool is_illuminated = true;
                double cosin = angle_cos(best_record.normal, light_direction);
                if (cosin < 0)
                    cosin = 0; // Lambert's Model
                double illumination_force = cosin;
                for (size_t i = 0; i < o_sz; i++)
                {
                    HitRecord dummy_record{};
                    if (Hit(&s[i], light_seeker, 0.001, light_distance, dummy_record))
                    {
                        is_illuminated = false;
                        break;
                    }
                }
                if (is_illuminated)
                {
                    local_color += ((s[hit_index]._color * illumination_force));
                }
            }
            final_color += attenuation * local_color * (1.0 - s[hit_index]._reflectivity);

            // make reflected ray as current and iterate again
            vec3 reflected_direction = ReflectedDir(current_ray.direction(), best_record.normal);
            Ray reflected_ray(best_record.p, reflected_direction);
            current_ray = reflected_ray;
            attenuation *= s[hit_index]._reflectivity;
            if (attenuation < 0.01)
                break;
        }
        else if (hit_index == -1)
        {
            final_color += attenuation * color{SKY_R, SKY_G, SKY_B};
            break;
        }
        depth++;
    }
    // Gamma Correction
    final_color[0] = sqrt(final_color.x());
    final_color[1] = sqrt(final_color.y());
    final_color[2] = sqrt(final_color.z());
    // Clamping
    final_color[0] = fminf(1.0f, fmaxf(0.0f, final_color.x()));
    final_color[1] = fminf(1.0f, fmaxf(0.0f, final_color.y()));
    final_color[2] = fminf(1.0f, fmaxf(0.0f, final_color.z()));

    return final_color;
}