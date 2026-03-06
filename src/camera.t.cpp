#include <gtest/gtest.h>
#include "camera.h"
#include "common/vec3.h"

TEST(CAMERA, basics)
{
    Camera camera;
    static const point3 pos0(0.0, 0.0, 0.0);
    static const  point3 pos(0.0, 12141.0, 2.0);
    static const  vec3 dir(1.0, 0.0, 0.0);
    static const vec3 dir2(0.0, 1.0, 0.0);

    EXPECT_EQ(camera.getPosition(), pos0);
    EXPECT_EQ(camera.getDirection(), dir);

    camera.SetPosition({0.0, 12141.0, 2.0});
    EXPECT_EQ(camera.getPosition(), pos);

    camera.SetDirection(dir2 * 123.1278653);
    EXPECT_EQ(camera.getDirection(), dir2);
}
TEST(CAMERA, viewport)
{
    {
        Camera camera({1.0, 0.0, 0.0}, point3{12.0, 0.0, 0.0});
        camera.RecalculateCamera();
        vec3 u = camera._viewport.u;
        u = unit_vector(u);
        vec3 vc{0.0, 0.0, -1.0};
        std::cerr << vc << " \n" << u << "\n";
        EXPECT_TRUE(u == vc);
    }
    {
        Camera camera({0.0, 0.0, 1.0}, point3{12.0, 0.0, 0.0});
        camera.RecalculateCamera();
        vec3 u = camera._viewport.u;
        u = unit_vector(u);
        vec3 vc{1.0, 0.0, 0.0};
        std::cerr << vc << " \n" << u << "\n";
        EXPECT_TRUE(u == vc);
    }
}