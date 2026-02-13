#include "quaternion.h"

Quaternion::Quaternion() : _w(1.0), _x(0.0), _y(0.0), _z(0.0) {};

Quaternion::Quaternion(double w, double x, double y, double z) : _w(w), _x(x), _y(y), _z(z) {};

Quaternion Quaternion::fromAxisAngle(const vec3 &axis, double angle)
{
    double half_angle = angle / 2.0;
    double sin_half_angle = std::sin(half_angle);
    double cos_half_angle = std::cos(half_angle);

    return Quaternion(cos_half_angle, axis.x() * sin_half_angle, axis.y() * sin_half_angle, axis.z() * sin_half_angle);
}

Quaternion Quaternion::operator*(const Quaternion &other) const
{
    return Quaternion(_w * other._w - _x * other._x - _y * other._y - _z * other._z,
                      _w * other._x + _x * other._w + _y * other._z - _z * other._y,
                      _w * other._y - _x * other._z + _y * other._w + _z * other._x,
                      _w * other._z + _x * other._y - _y * other._x + _z * other._w);
}
// Quaternion::Quaternion operator*(double scalar) const;
// Quaternion::Quaternion operator+(const Quaternion &other) const;
// Quaternion::Quaternion operator-(const Quaternion &other) const;
// Quaternion::Quaternion Conjugate() const;
// double Quaternion::Norm() const;
// Quaternion::Quaternion Inverse() const;
// vec3 Quaternion::Rotate(const vec3 &v) const;