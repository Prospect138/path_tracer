#include "renderer.h"

static vec3 k_direction{0.0, 0.0, 0.0};
static point3 k_position{0.0, 0.0, 0.0};

int main()
{
    Sphere sphere0{{0.0, 0.0, 7.0}, {255.0, 100.0, 100.0}, 1.0, 0.6};
    Sphere sphere1{{1.45, 2.0, 12.0}, {10.0, 200.0, 234.0}, 3.5, 0.6};
    Sphere sphere2{{-2.45, 5.0, 18.0}, {180.0, 150.0, 0.0}, 3.25, 0.8};
    const Sphere spheres[] = {sphere0, sphere1, sphere2};
    Renderer renderer{};
    renderer.DrawFrame(k_position, k_direction, spheres, 3);
    // renderer.StartMainLoop();
}
