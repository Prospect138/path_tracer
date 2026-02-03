#include "light.h"
#include "renderer.h"

static vec3 k_direction{0.0, 0.0, 0.0};
static point3 k_position{0.0, 0.0, 0.0};

int main()
{
    Sphere sphere0{{0.0, 0.0, 7.0}, {1.0, 0.32, 0.3}, 1.0, 0.6};
    Sphere sphere1{{1.45, 2.0, 12.0}, {0.04, 0.75, 0.9}, 3.5, 0.6};
    Sphere sphere2{{-2.45, 5.0, 18.0}, {0.7, 0.72, 0.0}, 3.25, 0.8};
    LightSource light0{{-2.0, 5.0, -10.0}};
    LightSource light1{{2.0, -2.0, 10.0}};

    const LightSource lights[] = {light0, light1};
    const Sphere spheres[] = {sphere0, sphere1, sphere2};

    Renderer renderer{};
    renderer.SetSpheres(spheres, 3);
    renderer.SetLights(lights, 2);
    renderer.DrawFrame(k_position, k_direction);
    // renderer.StartMainLoop();
}
