#pragma once

#include <SDL3/SDL.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_scancode.h>
#include <cstddef>
#include <memory>
#include <thread>
#include <vector>

#include "camera.h"
#include "color.h"
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
    void DrawFrame();
    void RenderSpheres();

    void setColors();
    void SetSpheres(Sphere *objs, size_t count);
    void SetLights(LightSource *lights, size_t count);

    ~Renderer();

  private:
    bool InitSDL();
    void ProcessInput();

    std::shared_ptr<Camera> _camera;

    color *_colours{nullptr};
    color *_device_colours{nullptr};
    uint32_t *_texture_buffer{nullptr};
    size_t _color_size;

    LightSource *_lights{nullptr};
    LightSource *_device_lights{nullptr};
    size_t _light_count{0};

    Sphere *_spheres{nullptr};
    Sphere *_device_spheres{nullptr};
    size_t _sphere_count{0};

    bool _quit = false;
    KeyboardHandler _handler;
    SDL_Window *_window{nullptr};
    SDL_Renderer *_renderer{nullptr};
    SDL_Texture *_texture{nullptr};
};