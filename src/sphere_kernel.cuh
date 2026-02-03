#ifndef SPHERE_KERNEL_CUH
#define SPHERE_KERNEL_CUH

#include "color.h"
#include "hittable.h"
#include "ray.h"
#include "sphere.h"
#include "vec3.h"
#include <cuda_runtime.h>

__device__ bool Hit(const Sphere *s, const Ray &r, double ray_tmin, double ray_tmax, HitRecord &record);

__device__ color SphereDrawPoint(const Sphere *s, size_t sz, const Ray &r);
#endif