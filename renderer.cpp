#include "renderer.h"

#include "calculate_kernel.cuh"
#include "sphere.h"

Renderer::Renderer() : _camera(std::make_shared<Camera>(vec3{0.0, 0.0, 0.0}, point3{0.0, 0.0, 0.0})) {};

Renderer::Renderer(std::shared_ptr<Camera> camera) : _camera(camera) {};

void Renderer::DrawFrame(const point3 &camera_position, const vec3 &camera_direction, const Sphere *objs, size_t size)
{
    _camera->SetPosition(camera_position);
    _camera->SetDirection(camera_direction);
    RenderSpheres(objs, size);
}

std::shared_ptr<Camera> Renderer::getCamera()
{
    return _camera;
}

void Renderer::RenderSpheres(const Sphere *objs, size_t obj_count)
{
    size_t mem_size = _camera->_image_width * _camera->_image_height * sizeof(vec3) * obj_count;
    color *host_colors;
    cudaError(cudaMallocHost(&host_colors, mem_size));
    color *device_colors;
    cudaError(cudaMalloc(reinterpret_cast<void **>(&device_colors), mem_size));

    Sphere *device_objs;
    cudaError(cudaMalloc(reinterpret_cast<void **>(&device_objs), obj_count * sizeof(Sphere)));

    cudaMemcpy(device_objs, objs, obj_count * sizeof(Sphere), cudaMemcpyHostToDevice);

    vec3 pixel_du = _camera->_viewport.u / _camera->_image_width; // длина пикселя по u
    vec3 pixel_dv = _camera->_viewport.v / _camera->_image_height;
    point3 viewport_upper_left = _camera->_camera_center - vec3(0, 0, _camera->_focal_length) -
                                 _camera->_viewport.u / 2 - _camera->_viewport.v / 2;
    point3 pixel_start = viewport_upper_left + 0.5 * (pixel_du + pixel_dv);

    launchCalculate(device_colors, objs, obj_count, _camera->_camera_center, _camera->_image_width,
                    _camera->_image_height, pixel_du, pixel_dv, pixel_start);
    cudaMemcpy(host_colors, device_colors, mem_size, cudaMemcpyDeviceToHost);

    std::cout << "P3\n" << _camera->_image_width << ' ' << _camera->_image_height << "\n255\n";

    for (int y = 0; y < _camera->_image_height; ++y)
        for (int x = 0; x < _camera->_image_width; ++x)
        {
            color pixel_color = host_colors[y * _camera->_image_width + x];
            int ir = static_cast<int>(pixel_color.x());
            int ig = static_cast<int>(pixel_color.y());
            int ib = static_cast<int>(pixel_color.z());
            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    cudaFree(device_objs);
    cudaFree(device_colors);
    cudaFreeHost(host_colors);
}