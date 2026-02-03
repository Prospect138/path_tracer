#ifndef CALCULATE_KERNEL_CUH
#define CALCULATE_KERNEL_CUH

#include <cuda_runtime.h>

#include <cstddef>

#include "color.h"
#include "sphere.h"

#ifdef __cplusplus
extern "C" {

__global__ void calculate(color* device_colors, const Sphere* obj, size_t sz,
                          const vec3 camera_center, const int width,
                          const int height, vec3 pixel_du, vec3 pixel_dv,
                          point3 pixel_start);

void launchCalculate(color* device_colors, const Sphere* objs, size_t sz,
                     const vec3 camera_center, const int width,
                     const int height, vec3 pixel_du, vec3 pixel_dv,
                     point3 pixel_start);
}
#endif
#endif