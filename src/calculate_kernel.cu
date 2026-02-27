#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>

#include "calculate_kernel.cuh"
#include "camera.h"
#include "light.h"
#include "common/vec3.h"

__device__ static real_t SKY_R = 0.05;
__device__ static real_t SKY_G = 0.2;
__device__ static real_t SKY_B = 0.65;
__device__ static int MAX_REFLECTIONS = 5;

__device__ vec3 ReflectedDir(const vec3 &vector, const vec3 &normal)
{
    return vector - normal * 2.0 * dot_product(vector, normal);
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
            if (s[i].Hit(current_ray, 0.001, 1e9, record))
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
            if (attenuation < 0.01)
                break;
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

__global__ void calculate(color *device_colors, const Sphere *objs, const size_t o_sz, const LightSource *lights,
                          const size_t l_sz, const Camera camera)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= camera._image_width || y >= camera._image_height)
        return;
    point3 pixel_center = camera._viewport.pixel_start + (camera._viewport.pixel_du * real_t(x)) + (camera._viewport.pixel_dv * real_t(y)); // coordinate in engine units
    vec3 ray_direction = pixel_center - camera._camera_center;
    Ray r(camera._camera_center, ray_direction);
    device_colors[y * camera._image_width + x] = SphereDrawPoint(objs, o_sz, lights, l_sz, r);
    // device_colors[y * camera._image_width + x] =color(real_t(x) / camera._image_width, real_t(y) /
    // camera._image_height, 0);
}

extern "C" void launchCalculate(color *device_colors, const Sphere *objs, size_t o_sz, const LightSource *lights,
                                const size_t l_sz, const Camera camera)
{
    dim3 threadBlock = {16, 16};
    dim3 grid = {(camera._image_width + threadBlock.x - 1) / threadBlock.x,
                 (camera._image_height + threadBlock.y - 1) / threadBlock.y};
    calculate<<<grid, threadBlock>>>(device_colors, objs, o_sz, lights, l_sz, camera);

    cudaDeviceSynchronize();
}