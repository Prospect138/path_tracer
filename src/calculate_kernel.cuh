#ifndef CALCULATE_KERNEL_CUH
#define CALCULATE_KERNEL_CUH

#include <cuda_runtime.h>

#include <cstddef>

#include "color.h"
#include "light.h"
#include "sphere.h"

#ifdef __cplusplus
extern "C"
{

    __global__ void calculate(color *device_colors, const Sphere *obj, size_t sz, const LightSource *lights,
                              const size_t l_sz, const vec3 camera_center, const int width, const int height,
                              vec3 pixel_du, vec3 pixel_dv, point3 pixel_start);

    void launchCalculate(color *device_colors, const Sphere *objs, const size_t o_sz, const LightSource *lights,
                         const size_t l_sz, const vec3 camera_center, const int width, const int height, vec3 pixel_du,
                         vec3 pixel_dv, point3 pixel_start);
}
#endif
#endif