#pragma once

#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

struct vec3
{
    __host__ __device__ vec3() : e_{0, 0, 0} {};
    __host__ __device__ vec3(double e0, double e1, double e2) : e_{e0, e1, e2} {};

    __host__ __device__ double x() const
    {
        return e_[0];
    }
    __host__ __device__ double y() const
    {
        return e_[1];
    }
    __host__ __device__ double z() const
    {
        return e_[2];
    }

    __host__ __device__ vec3 operator-() const
    {
        return vec3(-e_[0], -e_[1], -e_[2]);
    }
    __host__ __device__ double operator[](int i) const
    {
        assert(i >= 0 && i < 3 && "Index out of range");
        return e_[i];
    }
    __host__ __device__ double &operator[](int i)
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
    __host__ __device__ double length_squared() const
    {
        return (e_[0] * e_[0] + e_[1] * e_[1] + e_[2] * e_[2]);
    }
    __host__ __device__ double length() const
    {
        return sqrt(length_squared());
    }
    using point3 = vec3;

    __host__ __device__ double e_[3];
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

__host__ __device__ inline vec3 operator*(double t, const vec3 &v)
{
    return vec3(t * v.x(), t * v.y(), t * v.z());
}

__host__ __device__ inline vec3 operator*(const vec3 &v, double t)
{
    return t * v;
}

__host__ __device__ inline vec3 operator/(const vec3 &v, double t)
{
    return (1 / t) * v;
}

// скалярное произведение
__host__ __device__ inline double dot_product(const vec3 &u, const vec3 &v)
{
    return u.x() * v.x() + u.y() * v.y() + u.z() * v.z();
}

// угол между векторами
__host__ __device__ inline double angle_cos(const vec3 &u, const vec3 &v)
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
    return vec3{(u.y() * v.z() - u.z() * v.y()), (u.x() * v.z() - u.z() * v.x()),
                (u.x() * v.y() - u.y() * v.x())}; // conponents i, j and k
}

// единичный вектор
__host__ __device__ inline vec3 unit_vector(const vec3 &v)
{
    return v / v.length();
}

__host__ __device__ inline double dist(const point3 &v, const point3 &u)
{
    return sqrt(powf((v.x() - u.x()), 2) + powf((v.y() - u.y()), 2) + powf((v.z() - u.z()), 2));
}

__host__ vec3 inline rotateX(const vec3 &v, double angle)
{
    double cos = std::cos(angle);
    double sin = std::sin(angle);
    vec3 rotated_vector{v.x(), v.y() * cos - v.z() * sin, v.y() * sin + v.z() * cos};
    return rotated_vector;
}

__host__ vec3 inline rotateY(const vec3 &v, double angle)
{
    double cos = std::cos(angle);
    double sin = std::sin(angle);
    vec3 rotated_vector{v.x() * cos + v.z() * sin, v.y(), v.x() * -sin + v.z() * cos};
    return rotated_vector;
}

__host__ vec3 inline rotateZ(const vec3 &v, double angle)
{
    double cos = std::cos(angle);
    double sin = std::sin(angle);
    vec3 rotated_vector{v.x() * cos - v.x() * sin, v.y() * sin + v.y() * cos, v.z()};
    return rotated_vector;
}

__device__ inline double yaw(const double &x, const double &angle)
{
    return x * std::cos(angle) + x * std::sin(angle);
}