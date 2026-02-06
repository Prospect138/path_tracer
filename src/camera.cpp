#include "camera.h"
#include "vec3.h"

Camera::Camera()
{
    this->SetDirection({1.0, 0.0, 0.0});
    this->SetPosition({0.0, 0.0, 0.0});
    _speed = 0.1;
    RecalculateCamera();
}

Camera::Camera(const vec3 &direction, const point3 &position) : _direction(direction), _camera_center(position)
{
    _speed = 0.1;
    RecalculateCamera();
}

void Camera::RecalculateCamera()
{
    double aspect_ratio = 16.0 / 9.0;
    _image_width = 1920;
    _image_height = static_cast<int>(_image_width / aspect_ratio);
    _image_height = (_image_height < 1) ? 1 : _image_height;
    _focal_length = 1.0;          // расстояние от viewport до начала координат
    double viewport_height = 2.0; // высота в единицах измерения движка
    double viewport_width = viewport_height * static_cast<double>(_image_width) / static_cast<double>(_image_height);

    vec3 w = unit_vector(-_direction);       // направление
    vec3 u = unit_vector(cross(_dir_up, w)); // тангаж
    vec3 v = cross(w, u);                    // рыскание
    _viewport.u = u * viewport_width;
    _viewport.v = v * viewport_height;
}

void Camera::SetDirection(const vec3 &direction)
{
    _direction = direction;
}

void Camera::SetPosition(const point3 &position)
{
    _camera_center = position;
}

vec3 Camera::getDirection() const
{
    return _direction;
}

point3 Camera::getPosition() const
{
    return _camera_center;
}