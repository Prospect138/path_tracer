#pragma once

#include "vec3.h"

struct LightSource {
    LightSource() = default;
    LightSource(point3 coordinate) : _coordinate(coordinate) {};
    ~LightSource() = default;
    point3 _coordinate;
};