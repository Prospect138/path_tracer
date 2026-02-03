#ifndef SPHERE_KERNEL_CUH
#define SPHERE_KERNEL_CUH

#include "color.h"
#include "hittable.h"
#include "light.h"
#include "ray.h"
#include "sphere.h"
#include "vec3.h"
#include <cstddef>
#include <cuda_runtime.h>

__device__ bool Hit(const Sphere *s, const Ray &r, double ray_tmin, double ray_tmax, HitRecord &record);

__device__ color SphereDrawPoint(const Sphere *s, const size_t o_sz, const LightSource *lights, const size_t l_sz,
                                 const Ray &r);
#endif