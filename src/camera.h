#pragma once

#include "common/vec3.h"

struct Viewport
{
    Viewport() = default;
    Viewport(vec3 u, vec3 v) : u(u), v(v)
    {
    }
    // viewport left with full width length in engine units
    vec3 u;
    // viewport up with full width length in engine units
    vec3 v;
    point3 pixel_start;
    real_t viewport_height;
    real_t viewport_width;
    vec3 pixel_du;
    vec3 pixel_dv;
};

struct Camera
{
    Camera();
    Camera(const vec3 &direction, const point3 &position);

    void SetDirection(const vec3 &direction);
    void SetPosition(const point3 &position);
    void SetViewport();
    void RecalculateCamera();

    vec3 getDirection() const;
    point3 getPosition() const;

    Viewport _viewport;
    point3 _camera_center;
    const vec3 _dir_up{0.0, 1.0, 0.0}; // Y is up;
    real_t _focal_length;
    int _image_width;
    int _image_height;
    vec3 _direction;
    real_t _speed;
};