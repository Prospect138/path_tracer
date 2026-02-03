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
    void DrawFrame(const point3 &camera_position, const vec3 &camera_direction);
    void RenderSpheres();
    void SetSpheres(const Sphere *objs, size_t count);
    void SetLights(const LightSource *lights, size_t count);
    bool InitSDL();
    ~Renderer();

  private:
    bool initGL();
    void ProcessInput();
    void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags);

    KeyboardHandler _handler;
    std::shared_ptr<Camera> _camera;
    const LightSource *_lights{nullptr};
    size_t _light_count{0};
    const Sphere *_spheres{nullptr};
    size_t _sphere_count{0};
    SDL_Window *_window{nullptr};
    SDL_Renderer *_renderer{nullptr};
};