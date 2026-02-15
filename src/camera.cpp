#include "camera.h"
#include "vec3.h"

Camera::Camera()
{
    this->SetDirection({1.0, 0.0, 0.0});
    this->SetPosition({0.0, 0.0, 0.0});
    _speed = 0.2;
    RecalculateCamera();
}

Camera::Camera(const vec3 &direction, const point3 &position) : _direction(direction), _camera_center(position)
{
    _speed = 0.2;
    RecalculateCamera();
}

void Camera::RecalculateCamera()
{
    real_t aspect_ratio = 16.0 / 9.0; // 1.77
    _image_width = 1920;
    _image_height = static_cast<int>(_image_width / aspect_ratio); // 1080
    _image_height = (_image_height < 1) ? 1 : _image_height;       // 1080
    _focal_length = -1.0;                                           // расстояние от viewport до начала координат
    real_t viewport_height = 2.0 * _focal_length;                                  // высота в единицах измерения движка
    real_t viewport_width =
        viewport_height * static_cast<real_t>(_image_width) / static_cast<real_t>(_image_height); // 3.5555

    vec3 w = unit_vector(-_direction);
    vec3 u = unit_vector(cross(_dir_up, w));
    vec3 v = cross(w, u);
    _viewport.u = u * viewport_width;   // 1 * 3.55555
    _viewport.v = -v * viewport_height; // 1 * 2.0
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