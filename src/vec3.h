#pragma once

#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

typedef float real_t;
struct vec3
{
    __host__ __device__ vec3() : e_{0, 0, 0} {};
    __host__ __device__ vec3(real_t e0, real_t e1, real_t e2) : e_{e0, e1, e2} {};

    __host__ __device__ real_t x() const
    {
        return e_[0];
    }
    __host__ __device__ real_t y() const
    {
        return e_[1];
    }
    __host__ __device__ real_t z() const
    {
        return e_[2];
    }

    __host__ __device__ vec3 operator-() const
    {
        return vec3(-e_[0], -e_[1], -e_[2]);
    }
    __host__ __device__ real_t operator[](int i) const
    {
        assert(i >= 0 && i < 3 && "Index out of range");
        return e_[i];
    }
    __host__ __device__ real_t &operator[](int i)
    {
        assert(i >= 0 && i < 3 && "Index out of range");
        return e_[i];
    }
    __host__ __device__ vec3 &operator+=(const vec3 &other)
    {
        e_[0] += other[0];
        e_[1] += other[1];
        e_[2] += other[2];
        return *this;
    }
    __host__ __device__ vec3 &operator*=(const int &t)
    {
        e_[0] *= t;
        e_[1] *= t;
        e_[2] *= t;
        return *this;
    }
    __host__ __device__ vec3 &operator/=(const int &t)
    {
        return *this *= (1.0 / t);
    }
    __host__ __device__ real_t length_squared() const
    {
        return (e_[0] * e_[0] + e_[1] * e_[1] + e_[2] * e_[2]);
    }
    __host__ __device__ real_t length() const
    {
        return sqrt(length_squared());
    }
    using point3 = vec3;

    __host__ __device__ real_t e_[3];
};

using point3 = vec3;

inline std::ostream &operator<<(std::ostream &out, const vec3 &v)
{
    return out << v.x() << ' ' << v.y() << ' ' << v.z();
}

__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v)
{
    return vec3(u.x() + v.x(), u.y() + v.y(), u.z() + v.z());
}

__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v)
{
    return vec3(u.x() - v.x(), u.y() - v.y(), u.z() - v.z());
}

__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v)
{
    return vec3(u.x() * v.x(), u.y() * v.y(), u.z() * v.z());
}

__host__ __device__ inline vec3 operator*(real_t t, const vec3 &v)
{
    return vec3(t * v.x(), t * v.y(), t * v.z());
}

__host__ __device__ inline vec3 operator*(const vec3 &v, real_t t)
{
    return t * v;
}

__host__ __device__ inline vec3 operator/(const vec3 &v, real_t t)
{
    return (1 / t) * v;
}

// скалярное произведение
__host__ __device__ inline real_t dot_product(const vec3 &u, const vec3 &v)
{
    return u.x() * v.x() + u.y() * v.y() + u.z() * v.z();
}

// угол между векторами
__host__ __device__ inline real_t angle_cos(const vec3 &u, const vec3 &v)
{
    return dot_product(u, v) / (u.length() * v.length());
}

// векторное произведение
// i  j  k
// x1 y1 z1
// x2 y2 z2
// детерминант i, j и k через алгебраическое дополнение
__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v)
{
    return vec3(u.e_[1] * v.e_[2] - u.e_[2] * v.e_[1], u.e_[2] * v.e_[0] - u.e_[0] * v.e_[2],
                u.e_[0] * v.e_[1] - u.e_[1] * v.e_[0]);
}

// единичный вектор
__host__ __device__ inline vec3 unit_vector(const vec3 &v)
{
    return v / v.length();
}

__host__ __device__ inline real_t dist(const point3 &v, const point3 &u)
{
    return sqrt(powf((v.x() - u.x()), 2) + powf((v.y() - u.y()), 2) + powf((v.z() - u.z()), 2));
}

__host__ vec3 inline rotateX(const vec3 &v, real_t angle)
{

    real_t cos = std::cos(angle);
    real_t sin = std::sin(angle);
    vec3 rotated_vector{v.x(), v.y() * cos - v.z() * sin, v.y() * sin + v.z() * cos};
    return rotated_vector;
}

__host__ vec3 inline rotateY(const vec3 &v, real_t angle)
{

    real_t cos = std::cos(angle);
    real_t sin = std::sin(angle);
    vec3 rotated_vector{v.x() * cos + v.z() * sin, v.y(), v.x() * -sin + v.z() * cos};
    return rotated_vector;
}

__host__ vec3 inline rotateZ(const vec3 &v, real_t angle)
{

    real_t cos = std::cos(angle);
    real_t sin = std::sin(angle);
    vec3 rotated_vector{v.x() * cos - v.x() * sin, v.y() * sin + v.y() * cos, v.z()};
    return rotated_vector;
}

// axis должен быть единичной длины (unit_vector)
__host__ inline vec3 rotateAroundAxis(const vec3 &v, const vec3 &axis, real_t angle)
{
    real_t cosTheta = std::cos(angle);
    real_t sinTheta = std::sin(angle);
    real_t dotAV = dot_product(axis, v);
    vec3 crossAV = cross(axis, v);

    // Формула Родрига:
    // v_rot = v*cos(θ) + (axis × v)*sin(θ) + axis*(axis · v)*(1 - cos(θ))
    return v * cosTheta + crossAV * sinTheta + axis * dotAV * (1.0 - cosTheta);
}