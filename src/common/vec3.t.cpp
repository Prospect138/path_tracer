#include <gtest/gtest.h>
#include "vec3.h"

static const vec3 a0{0.0, 0.0, 0.0};
static const vec3 b0{0.0, 0.0, 0.0};

static const vec3 a1{0.0, 0.0, 1.0};
static const vec3 b1{0.0, 1.0, 0.0};

static const vec3 a2{0.0, 2.5, 1.0};
static const vec3 b2{1.0, 1.0, 1.0};

static const vec3 a3{2.0, 2.5, 1.0};
static const vec3 b3{1.0, 5.0, 1.4};

TEST(VEC3, OPERATIONS)
{
    {
        vec3 c = a1 + b1;
        vec3 result{0.0, 1.0, 1.0};
        EXPECT_EQ(c, result);
    }
    {
        vec3 c = a2 + b1;
        vec3 result{0.0, 3.5, 1.0};
        EXPECT_EQ(c, result);
    }
    {
        vec3 c = a0 + b1;
        EXPECT_EQ(c, b1);
    }
    {
        vec3 c = a3 * 3;
        vec3 result{6.0, 7.5, 3.0};
        EXPECT_EQ(c, result);
    }
    {
        EXPECT_EQ(a3.x(), 2.0);
        EXPECT_EQ(a3.y(), 2.5);
        EXPECT_EQ(a3.z(), 1.0);
    }
}

TEST(VEC3, DOT)
{
    {
        real_t c = dot_product(a2, b3);
        real_t result{13.9f};
        EXPECT_EQ(c, result);
    }
    {
        real_t c = dot_product(a3, b0);
        real_t result{0.0f};
        EXPECT_EQ(c, result);
    }
}

TEST(VEC3, CROSS)
{
    {
        vec3 c = cross(a1, b1);
        vec3 result = {-1.0, 0.0, 0.0};
        EXPECT_EQ(c, result);

        c = cross(b1, a1);
        result = {1.0, 0.0, 0.0};
        EXPECT_EQ(c, result);
    }
    {
        vec3 c = cross(a2, b2);
        vec3 result = {1.5, 1.0, -2.5};
        EXPECT_EQ(c, result);
    }
    {
        vec3 c = cross(a0, b0);
        vec3 result = {0.0, 0.0, 0.0};
        EXPECT_EQ(c, result);
    }
}
