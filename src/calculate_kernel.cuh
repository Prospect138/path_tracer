#ifndef CALCULATE_KERNEL_CUH
#define CALCULATE_KERNEL_CUH

#include <cuda_runtime.h>

#include <cstddef>

#include "camera.h"
#include "color.h"
#include "light.h"
#include "sphere.h"

#ifdef __cplusplus
extern "C"
{

    __global__ void calculate(color *device_colors, const Sphere *obj, size_t sz, const LightSource *lights,
                              const size_t l_sz, const Camera camera);

    void launchCalculate(color *device_colors, const Sphere *objs, size_t o_sz, const LightSource *lights,
                         const size_t l_sz, const Camera camera);
}
#endif
#endif