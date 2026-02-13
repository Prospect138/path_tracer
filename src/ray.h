#pragma once

#include "vec3.h"

class Ray
{
  public:
    __host__ __device__ Ray() = default;
    __host__ __device__ Ray(const point3 &origin, const vec3 &direction) : origin_(origin), direction_(direction) {};

    __host__ __device__ const point3 &origin() const
    {
        return origin_;
    }
    __host__ __device__ const vec3 &direction() const
    {
        return direction_;
    }

    // точка на луче через линейную интерполяцию
    __host__ __device__ point3 lerp(real_t t) const
    {
        return origin_ + t * direction_;
    }

  private:
    point3 origin_;
    vec3 direction_;
};