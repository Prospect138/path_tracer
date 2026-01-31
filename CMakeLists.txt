cmake_minimum_required(VERSION 3.20)
project(rayctracer LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL3 REQUIRED sdl3)

set(CMAKE_CUDA_ARCHITECTURES 86 87)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

set(SOURCE
    main.cpp
    #sphere.cpp
    sphere_kernel.cu
    calculate_kernel.cu
    camera.cpp
    renderer.cpp)

add_executable(raytracer ${SOURCE})

target_compile_options(raytracer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_features(raytracer PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(raytracer PRIVATE
    ${SDL3_LIBRARIES}
    CUDA::cudart 
    CUDA::cuda_driver
)

target_include_directories(raytracer PRIVATE ${SDL3_INCLUDE_DIRS})

set_target_properties(raytracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)