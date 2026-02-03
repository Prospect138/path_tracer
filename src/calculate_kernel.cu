#include <cstddef>
#include <cuda_runtime.h>

#include "calculate_kernel.cuh"
#include "light.h"
#include "sphere_kernel.cuh"

__global__ void calculate(color *device_colors, const Sphere *objs, const size_t o_sz, const LightSource *lights,
                          const size_t l_sz, const vec3 camera_center, const int width, const int height, vec3 pixel_du,
                          vec3 pixel_dv, point3 pixel_start)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    point3 pixel_center = pixel_start + (pixel_du * double(x)) + (pixel_dv * double(y)); // координата пикселя
    vec3 ray_direction = pixel_center - camera_center;
    Ray r(camera_center, ray_direction);
    device_colors[y * width + x] = SphereDrawPoint(objs, o_sz, lights, l_sz, r);
}

extern "C" void launchCalculate(color *device_colors, const Sphere *objs, const size_t o_sz, const LightSource *lights,
                                const size_t l_sz, const vec3 camera_center, const int width, const int height,
                                vec3 pixel_du, vec3 pixel_dv, point3 pixel_start)
{
    dim3 threadBlock = {16, 16};
    dim3 grid = {(width + threadBlock.x - 1) / threadBlock.x, (height + threadBlock.y - 1) / threadBlock.y};

    calculate<<<grid, threadBlock>>>(device_colors, objs, o_sz, lights, l_sz, camera_center, width, height, pixel_du,
                                     pixel_dv, pixel_start);

    cudaDeviceSynchronize();
}