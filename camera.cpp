#include "camera.h"

Camera::Camera(const vec3 &direction, const point3 &position) : _direction(direction), _position(position)
{
    double aspect_ratio = 16.0 / 9.0;
    _image_width = 1920;
    _image_height = static_cast<int>(_image_width / aspect_ratio);
    _image_height = (_image_height < 1) ? 1 : _image_height;
    double width_diviser = 1.0 / (_image_width - 1);
    double height_diviser = 1.0 / (_image_height - 1);
    _focal_length = 1.0;          // расстояние от viewport до начала координат
    double viewport_height = 2.0; // высота в единицах измерения движка
    double viewport_width = viewport_height * static_cast<double>(_image_width) / static_cast<double>(_image_height);
    _camera_center = point3(0, 0, 0);
    _viewport.u = vec3(viewport_width, 0, 0);
    _viewport.v = vec3(0, -viewport_height, 0);
}

void Camera::SetDirection(const vec3 &direction)
{
    _direction = direction;
}

void Camera::SetPosition(const point3 &position)
{
    _position = position;
}
