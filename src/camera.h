#pragma once

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
    Camera() = default;
    Camera(const vec3 &direction, const point3 &position);
    void SetDirection(const vec3 &direction);
    void SetPosition(const point3 &position);

    vec3 getDirection() const;
    point3 getPosition() const;

    Viewport _viewport;
    point3 _camera_center;
    double _focal_length;
    int _image_width;
    int _image_height;
    vec3 _direction;
    point3 _position;

    double _speed;
};