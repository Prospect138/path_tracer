#pragma once

#include <memory>
#include <vector>

#include "camera.h"
#include "light.h"
#include "sphere.h"
#include "vec3.h"

class Renderer
{
  public:
    Renderer();
    Renderer(std::shared_ptr<Camera> camera);
    std::shared_ptr<Camera> getCamera();
    void DrawFrame(const point3 &camera_position, const vec3 &camera_direction, const Sphere *objs, size_t size);
    void RenderSpheres(const Sphere *objs, size_t obj_count);
    ~Renderer() = default;

  private:
    std::shared_ptr<Camera> _camera;
    std::vector<LightSource> _lights;
};