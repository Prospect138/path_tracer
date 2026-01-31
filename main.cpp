#include "renderer.h"

static vec3 k_direction{0.0, 0.0, 0.0};
static point3 k_position{0.0, 0.0, 0.0};

int main()
{
    Sphere sphere0{{0.0, 0.0, 7.0}, {255.0, 100.0, 100.0}, 2.0};
    Sphere sphere1{{1.45, 2.0, 12.0}, {10.0, 200.0, 234.0}, 3.5};
    const Sphere spheres[] = {sphere0, sphere1};
    Renderer renderer{};
    renderer.DrawFrame(k_position, k_direction, spheres, 2);
}
