#include <cuda_runtime.h>

#include "calculate_kernel.cuh"
#include "sphere_kernel.cuh"

__global__ void calculate(color *device_colors, const Sphere *objs, size_t sz, const vec3 camera_center,
                          const int width, const int height, vec3 pixel_du, vec3 pixel_dv, point3 pixel_start)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    point3 pixel_center = pixel_start + (pixel_du * double(x)) + (pixel_dv * double(y)); // координата пикселя
    vec3 ray_direction = pixel_center - camera_center;
    Ray r(camera_center, ray_direction);
    device_colors[y * width + x] = SphereDrawPoint(objs, sz, r);
}

extern "C" void launchCalculate(color *device_colors, const Sphere *objs, size_t sz, const vec3 camera_center,
                                const int width, const int height, vec3 pixel_du, vec3 pixel_dv, point3 pixel_start)
{
    dim3 threadBlock = {16, 16};
    dim3 grid = {(width + threadBlock.x - 1) / threadBlock.x, (height + threadBlock.y - 1) / threadBlock.y};

    for (size_t i = 0; i < sz; i++)
    {
        calculate<<<grid, threadBlock>>>(device_colors, objs, sz, camera_center, width, height, pixel_du, pixel_dv,
                                         pixel_start);
    }

    cudaDeviceSynchronize();
}