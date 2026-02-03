#include "light.h"
#include "renderer.h"

static vec3 k_direction{0.0, 0.0, 0.0};
static point3 k_position{0.0, 0.0, 0.0};

int main()
{
    Sphere sphere0{{10.0, 2.0, 12.0}, {0.00, 0.0, 0.0}, 3.5, 1.0};
    Sphere sphere1{{1.45, 2.0, 12.0}, {0.04, 0.75, 0.1}, 3.5, 0.9};
    Sphere sphere2{{-8.4, 2.0, 12.0}, {0.04, 0.75, 0.1}, 3.5, 0.9};
    Sphere sphere3{{-2.45, 1008.0, 0.0}, {0.2, 0.72, 0.1}, 1000.0, 0.0};
    LightSource light0{{-2.0, 5.0, -10.0}};
    LightSource light1{{2.0, -2.0, 10.0}};

    const LightSource lights[] = {light0, light1};
    const Sphere spheres[] = {sphere0, sphere1, sphere2, sphere3};

    Renderer renderer{};
    renderer.SetSpheres(spheres, 4);
    renderer.SetLights(lights, 2);
    renderer.DrawFrame(k_position, k_direction);
    // renderer.StartMainLoop();
}
