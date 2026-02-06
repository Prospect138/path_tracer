#include <cstddef>
#include <cuda_runtime.h>

#include "calculate_kernel.cuh"
#include "camera.h"
#include "light.h"
#include "sphere_kernel.cuh"
#include "vec3.h"

__global__ void calculate(color *device_colors, const Sphere *objs, const size_t o_sz, const LightSource *lights,
                          const size_t l_sz, const Camera camera)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= camera._image_width || y >= camera._image_height)
        return;
    vec3 pixel_du = camera._viewport.u / camera._image_width; // длина пикселя по u
    vec3 pixel_dv = camera._viewport.v / camera._image_height;
    // Это что такое?
    point3 viewport_upper_left = camera._camera_center - (unit_vector(camera._direction) * camera._focal_length) -
                                 camera._viewport.u / 2.0 - camera._viewport.v / 2.0;
    point3 pixel_start = viewport_upper_left + 0.5 * (pixel_du + pixel_dv);

    point3 pixel_center = pixel_start + (pixel_du * double(x)) + (pixel_dv * double(y)); // координата пикселя
    vec3 ray_direction = pixel_center - camera._camera_center;
    Ray r(camera._camera_center, ray_direction);
    device_colors[y * camera._image_width + x] = SphereDrawPoint(objs, o_sz, lights, l_sz, r);
    // device_colors[y * camera._image_width + x] =color(double(x) / camera._image_width, double(y) /
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