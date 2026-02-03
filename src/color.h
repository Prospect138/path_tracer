#pragma once

#include "vec3.h"

#include <iostream>

using color = vec3;

inline void write_color(std::ostream& out, const color& pixel_color) {
    double r = pixel_color.x();
    double g = pixel_color.y();
    double b = pixel_color.z();

    int ir = static_cast<int>(255.99 * r);
    int ig = static_cast<int>(255.99 * g);
    int ib = static_cast<int>(255.99 * b);
    
    out << ir << " " << ig << " " << ib << "\n";
}
