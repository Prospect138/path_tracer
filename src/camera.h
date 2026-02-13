#pragma once

#include "quaternion.h"
#include "vec3.h"

struct Viewport
{
    Viewport() : u({0.0, 0.0, 0.0}), v({0.0, 0.0, 0.0})
    {
    }
    Viewport(vec3 u, vec3 v) : u(u), v(v)
    {
    }
    vec3 u;
    vec3 v;
};

struct Camera
{
    Camera();
    Camera(const vec3 &direction, const point3 &position);

    void SetDirection(const vec3 &direction);
    void SetPosition(const point3 &position);
    void RecalculateCamera();

    Quaternion getOrientation() const;
    point3 getPosition() const;

    Viewport _viewport;
    point3 _camera_center;
    vec3 _dir_up{0.0, 1.0, 0.0}; // Y is up;
    double _focal_length;
    int _image_width;
    int _image_height;
    vec3 _direction;

    double _speed;
};