#pragma once
#include "vec3.h"

class Quaternion
{
  public:
    Quaternion();
    Quaternion(double w, double x, double y, double z);
    static Quaternion fromAxisAngle(const vec3 &axis, double angle);
    Quaternion operator*(const Quaternion &other) const;
    Quaternion operator*(double scalar) const;
    Quaternion operator+(const Quaternion &other) const;
    Quaternion operator-(const Quaternion &other) const;
    Quaternion Conjugate() const;
    double Norm() const;
    Quaternion Inverse() const;
    vec3 Rotate(const vec3 &v) const;

  private:
    double _w, _x, _y, _z;
};