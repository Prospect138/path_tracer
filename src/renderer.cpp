#include "renderer.h"
#include "calculate_kernel.cuh"
#include "sphere.h"
#include "vec3.h"

#include <SDL3/SDL_events.h>
#include <cuda_gl_interop.h>
#include <driver_types.h>

static const int MOUSE_SENSETIVITY = 1.0;
static const int SCREEN_FPS = 60;
static const int SCREEN_TICKS_PER_FRAME = 1000 / SCREEN_FPS;

extern __host__ cudaError_t CUDARTAPI cudaGraphicsGLRegisterBuffer(struct cudaGraphicsResource **resource,
                                                                   GLuint buffer, unsigned int flags);

Renderer::Renderer()
    : _camera(std::make_shared<Camera>(vec3{0.0, 0.0, 0.0}, point3{0.0, 0.0, 0.0})) {
          // if (!initSDL())
          //{
          //     std::cerr << "Failed to initialize SDL context." << std::endl;
          // }
      };

Renderer::Renderer(std::shared_ptr<Camera> camera)
    : _camera(camera) {
          // if (!initSDL())
          //{
          //     std::cerr << "Failed to initialize SDL context." << std::endl;
          // }
      };

void Renderer::DrawFrame(const point3 &camera_position, const vec3 &camera_direction)
{
    _camera->SetPosition(camera_position);
    _camera->SetDirection(camera_direction);
    RenderSpheres();
}

std::shared_ptr<Camera> Renderer::setCamera(std::shared_ptr<Camera> camera)
{
    _camera = camera;
    return _camera;
}

std::shared_ptr<Camera> Renderer::getCamera()
{
    return _camera;
}

/*
void Renderer::StartMainLoop()
{
   bool quit_ = false;
   SDL_SetWindowMouseGrab(_window, true);
   // SDL_SetRelativeMouseMode(true);

   signed short int mouseDX;
   SDL_Event event;
   while (quit_ == false)
   {
       SDL_PollEvent(&event);
       _handler.handleInput(event);
       auto start = std::chrono::high_resolution_clock::now();
       // LOG("rendering frame");

       ProcessInput();

       DrawFrame(_camera->_position, _camera->_direction, nullptr, 0);
   }
}
*/

void Renderer::SetSpheres(const Sphere *objs, size_t count)
{
    _spheres = objs;
    _sphere_count = count;
}

void Renderer::SetLights(const LightSource *lights, size_t count)
{
    _lights = lights;
    _light_count = count;
}

void Renderer::RenderSpheres()
{
    size_t mem_size = _camera->_image_width * _camera->_image_height * sizeof(vec3) * _sphere_count;
    color *host_colors;
    cudaError(cudaMallocHost(&host_colors, mem_size));
    color *device_colors;
    // cudaError(
    //     cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&device_colors), nullptr,
    //     _cuda_vbo_resource));
    cudaError(cudaMalloc(reinterpret_cast<void **>(&device_colors), mem_size));

    Sphere *device_objs;
    cudaError(cudaMalloc(reinterpret_cast<void **>(&device_objs), _sphere_count * sizeof(Sphere)));

    cudaMemcpy(device_objs, _spheres, _sphere_count * sizeof(Sphere), cudaMemcpyHostToDevice);

    vec3 pixel_du = _camera->_viewport.u / _camera->_image_width; // длина пикселя по u
    vec3 pixel_dv = _camera->_viewport.v / _camera->_image_height;
    point3 viewport_upper_left = _camera->_camera_center - vec3(0, 0, _camera->_focal_length) -
                                 _camera->_viewport.u / 2 - _camera->_viewport.v / 2;
    point3 pixel_start = viewport_upper_left + 0.5 * (pixel_du + pixel_dv);

    launchCalculate(device_colors, _spheres, _sphere_count, _lights, _light_count, _camera->_camera_center,
                    _camera->_image_width, _camera->_image_height, pixel_du, pixel_dv, pixel_start);

    // glBindBuffer(GL_ARRAY_BUFFER, _vbo);

    cudaMemcpy(host_colors, device_colors, mem_size, cudaMemcpyDeviceToHost);

    std::cout << "P3\n" << _camera->_image_width << ' ' << _camera->_image_height << "\n255\n";

    for (int y = 0; y < _camera->_image_height; ++y)
        for (int x = 0; x < _camera->_image_width; ++x)
        {
            color pixel_color = host_colors[y * _camera->_image_width + x];
            int ir = static_cast<int>(255.99 * pixel_color.x());
            int ig = static_cast<int>(255.99 * pixel_color.y());
            int ib = static_cast<int>(255.99 * pixel_color.z());
            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }

    // glSwapBuffersSDL(_window);
    cudaFree(device_objs);
    cudaFree(device_colors);
    cudaFreeHost(host_colors);
}
/*
bool Renderer::initSDL()
{
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        std::cerr << "Could not initialize SDL: " << SDL_GetError() << std::endl;
        return false;
    }
    SDL_Window *window = nullptr;
    SDL_Renderer *renderer = nullptr;
    if (!SDL_CreateWindowAndRenderer("Path Tracer", _camera->_image_width, _camera->_image_height, 0, &_window,
                                     &_renderer))
    {
        std::cerr << "Could not create window and renderer: " << SDL_GetError() << std::endl;
        return false;
    }
    return true;
}

void Renderer::ProcessInput()
{
    if (_handler.getKeyState(SDL_SCANCODE_W))
    {
        _camera->SetPosition(_camera->getPosition() +
                             unit_vector(_camera->getDirection()) * static_cast<double>(_camera->_speed));
    }

    if (_handler.getKeyState(SDL_SCANCODE_S))
    {
        _camera->SetPosition(_camera->getPosition() -
                             unit_vector(_camera->getDirection()) * static_cast<double>(_camera->_speed));
    }

    if (_handler.getKeyState(SDL_SCANCODE_A))
    {
        _camera->SetPosition(_camera->getPosition() - unit_vector(cross(_camera->getDirection(), vec3{0.0, 1.0, 0.0})) *
                                                          static_cast<double>(_camera->_speed));
    }

    if (_handler.getKeyState(SDL_SCANCODE_D))
    {
        _camera->SetPosition(_camera->getPosition() + unit_vector(cross(_camera->getDirection(), vec3{0.0, 1.0, 0.0})) *
                                                          static_cast<double>(_camera->_speed));
    }

    if (_handler.getKeyState(SDL_SCANCODE_ESCAPE))
    {
        SDL_Event quit_event;
        quit_event.type = SDL_EVENT_QUIT;
        SDL_PushEvent(&quit_event);
    }

    if (abs(_handler.getMouseDelta().first) > 0)
    {
        // Gimbal Lock possible
        _camera->SetDirection(rotateX(_camera->getDirection(), _handler.getMouseDelta().first * MOUSE_SENSETIVITY));
        _camera->SetDirection(rotateY(_camera->getDirection(), _handler.getMouseDelta().second * MOUSE_SENSETIVITY));
    }
}
*/

Renderer::~Renderer()
{
    // if (_renderer)
    //{
    //     SDL_DestroyRenderer(_renderer);
    //     _renderer = nullptr;
    // }
    // if (_window)
    //{
    //     SDL_DestroyWindow(_window);
    //     _window = nullptr;
    // }
    SDL_Quit();
}