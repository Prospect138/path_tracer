#include <cstddef>
#include <cstdio>
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
    //
    vec3 pixel_du = camera._viewport.u / camera._image_width;  // 0.00185185... engine units
    vec3 pixel_dv = camera._viewport.v / camera._image_height; // something like 0.00185185... engine units

    point3 viewport_center = camera._camera_center - (camera._focal_length * camera._direction);
    point3 viewport_upper_left = viewport_center - camera._viewport.u / 2.0 - camera._viewport.v / 2.0; // engine units
    if (x == 1 && y == 1)
    {
        printf("viewport center   : x=%f, y=%f, z=%f\n", viewport_center.x(), viewport_center.y(), viewport_center.z());
        printf("upper      left   : x=%f, y=%f, z=%f\n\n", viewport_upper_left.x(), viewport_upper_left.y(),
               viewport_upper_left.z());
    }
    point3 pixel_start = viewport_upper_left + (0.5 * pixel_du) + (0.5 * pixel_dv);
    point3 pixel_center = pixel_start + (pixel_du * real_t(x)) + (pixel_dv * real_t(y)); // coordinate in engine units
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
    printf("camera center     : x=%f, y=%f, z=%f\n", camera.getPosition().x(), camera.getPosition().y(),
           camera.getPosition().z());
    printf("camera direction  : x=%f, y=%f, z=%f\n", camera.getDirection().x(), camera.getDirection().y(),
           camera.getDirection().z());
    calculate<<<grid, threadBlock>>>(device_colors, objs, o_sz, lights, l_sz, camera);

    cudaDeviceSynchronize();
}