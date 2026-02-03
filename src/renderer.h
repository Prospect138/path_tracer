#pragma once
#include <GL/glew.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_scancode.h>
#include <memory>
#include <thread>
#include <vector>

#include "camera.h"
#include "keyboard_handler.h"
#include "light.h"
#include "sphere.h"
#include "vec3.h"

class Renderer
{
  public:
    Renderer();
    Renderer(std::shared_ptr<Camera> camera);
    std::shared_ptr<Camera> setCamera(std::shared_ptr<Camera> camera);
    std::shared_ptr<Camera> getCamera();
    void StartMainLoop();
    void DrawFrame(const point3 &camera_position, const vec3 &camera_direction, const Sphere *objs, size_t size);
    void RenderSpheres(const Sphere *objs, size_t obj_count);
    ~Renderer();

  private:
    bool initGL();
    void ProcessInput();
    void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags);

    KeyboardHandler _handler;
    std::shared_ptr<Camera> _camera;
    std::vector<LightSource> _lights;
    SDL_Window *_window{nullptr};
    SDL_Renderer *_renderer{nullptr};
    SDL_GLContext _glContext{nullptr};
    GLuint _vbo;
    struct cudaGraphicsResource *_cuda_vbo_resource;
};