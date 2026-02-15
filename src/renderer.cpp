#include "renderer.h"
#include "calculate_kernel.cuh"
#include "camera.h"
#include "cuda_check.h"
#include "light.h"
#include "sphere.h"
#include "vec3.h"

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_pixels.h>
#include <SDL3/SDL_render.h>
#include <cstdint>
#include <cstdio>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

static const real_t MOUSE_SENSETIVITY = 0.02;
static const int SCREEN_FPS = 60;
static const int SCREEN_TICKS_PER_FRAME = 1000 / SCREEN_FPS;

Renderer::Renderer()
{
    setCamera(std::make_shared<Camera>(vec3{0.0, 0.0, 1.0}, point3{0.0, 0.0, 0.0}));
    if (!InitSDL())
        std::cerr << "Failed to initialize SDL context." << std::endl;

    setColorBuffer();
};

Renderer::Renderer(std::shared_ptr<Camera> camera) : _camera(camera)
{
    setCamera(camera);
    if (!InitSDL())
        std::cerr << "Failed to initialize SDL context." << std::endl;

    setColorBuffer();
};

void Renderer::setColorBuffer()
{
    if (_camera)
    {
        _color_size = _camera->_image_width * _camera->_image_height * sizeof(vec3);
        if (_colours)
            cudaFreeHost(_colours);
        CUDA_CHECK(cudaMallocHost(&_colours, _color_size));
        if (_device_colours)
            cudaFree(_device_colours);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&_device_colours), _color_size));
    }
    else
        std::cerr << "Failed to allocate colors - no camera found." << std::endl;
    size_t pixel_count = _camera->_image_width * _camera->_image_height;
    _texture_buffer = new uint32_t[pixel_count];
    if (_texture_buffer == nullptr)
    {
        std::cerr << "Failed to allocate texture buffer." << std::endl;
        return;
    }
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

void Renderer::StartMainLoop()
{
    SDL_SetWindowMouseGrab(_window, true);
    SDL_SetWindowRelativeMouseMode(_window, true);

    signed short int mouseDX;
    SDL_Event event;

    std::cout << "Entering main loop" << std::endl;
    while (_quit == false)
    {
        while (SDL_PollEvent(&event))
        {
            _handler.handleInput(event);
        }
        // auto start = std::chrono::high_resolution_clock::now();
        // LOG("rendering frame");
        // printf("Rendering Frame\n");
        ProcessInput();
        RenderSpheres();
    }
}

void Renderer::SetSpheres(Sphere *objs, size_t count)
{
    if (objs == nullptr || count == 0)
    {
        std::cerr << "Failed to allocate spheres - nothing to allocate" << std::endl;
        return;
    }
    _spheres = objs;
    _sphere_count = count;

    if (_device_spheres)
        cudaFree(_device_spheres);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&_device_spheres), _sphere_count * sizeof(Sphere)));
    CUDA_CHECK(cudaMemcpy(_device_spheres, _spheres, _sphere_count * sizeof(Sphere), cudaMemcpyHostToDevice));
}

void Renderer::SetLights(LightSource *lights, size_t count)
{
    if (lights == nullptr || count == 0)
    {
        std::cout << "Warning: allocate lights - nothing to allocate" << std::endl;
        return;
    }

    _lights = lights;
    _light_count = count;

    if (_device_lights)
        cudaFree(_device_lights);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&_device_lights), _light_count * sizeof(LightSource)));
    CUDA_CHECK(cudaMemcpy(_device_lights, _lights, _light_count * sizeof(LightSource), cudaMemcpyHostToDevice));
}

void Renderer::RenderSpheres()
{
    launchCalculate(_device_colours, _device_spheres, _sphere_count, _device_lights, _light_count, *_camera);

    cudaMemcpy(_colours, _device_colours, _color_size, cudaMemcpyDeviceToHost);

    // std::cout << "P3\n" << _camera->_image_width << ' ' << _camera->_image_height << "\n255\n";

    for (int y = 0; y < _camera->_image_height; ++y)
    {
        for (int x = 0; x < _camera->_image_width; ++x)
        {
            color pixel_color = _colours[y * _camera->_image_width + x];

            int ir = static_cast<int>(255.99 * pixel_color.x());
            int ig = static_cast<int>(255.99 * pixel_color.y());
            int ib = static_cast<int>(255.99 * pixel_color.z());
            // std::cout << ir << ' ' << ig << ' ' << ib << '\n';
            // SDL_SetRenderDrawColor(_renderer, 255, 0, 0, 255);

            _texture_buffer[y * _camera->_image_width + x] = (255 << 24) | (ir << 16) | (ig << 8) | ib;
            // SDL_SetRenderDrawColor(_renderer, ir, ig, ib, 255);
            // SDL_RenderPoint(_renderer, x, y);
        }
    }
    SDL_UpdateTexture(_texture, nullptr, _texture_buffer, _camera->_image_width * sizeof(uint32_t));
    SDL_RenderClear(_renderer);
    SDL_RenderTexture(_renderer, _texture, nullptr, nullptr);
    SDL_RenderPresent(_renderer);
}

bool Renderer::InitSDL()
{
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        std::cerr << "Could not initialize SDL: " << SDL_GetError() << std::endl;
        return false;
    }
    if (!SDL_CreateWindowAndRenderer("Path Tracer", _camera->_image_width, _camera->_image_height, 0, &_window,
                                     &_renderer))
    {
        std::cerr << "Could not create window and renderer: " << SDL_GetError() << std::endl;
        return false;
    }

    _texture = SDL_CreateTexture(_renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING,
                                 _camera->_image_width, _camera->_image_height);
    if (_texture == nullptr)
    {
        std::cerr << "Could not create texture: " << SDL_GetError() << std::endl;
    }
    return true;
}

void Renderer::ProcessInput()
{
    vec3 forward = _camera->getDirection();
    vec3 right = unit_vector(cross(forward, _camera->_dir_up)); // Вектор ВПРАВО
    point3 pos = _camera->getPosition();

    if (_handler.getKeyState(SDL_SCANCODE_W))
        _camera->SetPosition(pos + forward * _camera->_speed);
    if (_handler.getKeyState(SDL_SCANCODE_S))
        _camera->SetPosition(pos - forward * _camera->_speed);
    if (_handler.getKeyState(SDL_SCANCODE_D))
        _camera->SetPosition(pos - right * _camera->_speed);
    if (_handler.getKeyState(SDL_SCANCODE_A))
        _camera->SetPosition(pos + right * _camera->_speed);

    if (_handler.getKeyState(SDL_SCANCODE_ESCAPE))
    {
        _quit = true;
    }

    if (abs(_handler.getMouseDxDy().first) > 1 || abs(_handler.getMouseDxDy().second) > 1)
    {
        // Gimbal Lock possible
        real_t pitch = _handler.getMouseDxDy().first * MOUSE_SENSETIVITY;
        real_t yaw = _handler.getMouseDxDy().second * MOUSE_SENSETIVITY;
        vec3 currentDir = _camera->getDirection();

        // 1. Поворот ВЛЕВО-ВПРАВО: вращаем вокруг мирового Up
        // Отрицательный dx, чтобы мышь вправо крутила камеру вправо
        currentDir = rotateAroundAxis(currentDir, _camera->_dir_up, pitch);

        // 2. Поворот ВВЕРХ-ВНИЗ: вращаем вокруг ЛОКАЛЬНОЙ оси Right
        // Ось Right всегда перпендикулярна взгляду и мировому Up
        vec3 pitch_axis = unit_vector(cross(currentDir, _camera->_dir_up));

        // ВАЖНО: ограничиваем угол (pitch), чтобы камера не перевернулась.
        // В векторном виде это сложнее, чем в Эйлере, но для начала просто повернем:
        currentDir = rotateAroundAxis(currentDir, pitch_axis, yaw);

        // 3. Обновляем вектор в камере (обязательно нормализуем!)
        _camera->SetDirection(unit_vector(currentDir));

        _camera->RecalculateCamera();

        _handler.flushMouse();
    }
}

Renderer::~Renderer()
{
    cudaFreeHost(_colours);
    cudaFree(_device_colours);
    cudaFree(_device_spheres);
    cudaFree(_device_lights);
    if (_renderer)
    {
        SDL_DestroyRenderer(_renderer);
        _renderer = nullptr;
    }
    if (_window)
    {
        SDL_DestroyWindow(_window);
        _window = nullptr;
    }
    SDL_Quit();
}