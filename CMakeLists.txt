cmake_minimum_required(VERSION 3.20)
project(raytracer LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

enable_testing()

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(GTest REQUIRED)
pkg_check_modules(SDL3 REQUIRED sdl3)

set(CMAKE_CUDA_ARCHITECTURES 86 87)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

set(SRC src/)
set(SOURCE
    ${SRC}main.cpp
    ${SRC}calculate_kernel.cu
    ${SRC}camera.cpp
    ${SRC}renderer.cpp
    ${SRC}keyboard_handler.cpp)

add_executable(raytracer ${SOURCE})

target_compile_options(raytracer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo--extended-lambda>)
target_compile_features(raytracer PRIVATE cxx_std_20 cuda_std_20)
#target_compile_options(raytracer PRIVATE -fsanitize=address)
#target_link_options(raytracer PRIVATE -fsanitize=address)

target_link_libraries(raytracer PRIVATE
    ${SDL3_LIBRARIES}
    CUDA::cudart 
    CUDA::cuda_driver
)
target_include_directories(raytracer PRIVATE
    ${SDL3_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set(UNIT_TESTS
    ${SRC}common/vec3.t.cpp)

include(GoogleTest)

add_executable(tst ${UNIT_TESTS})

set_target_properties(tst PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(tst PRIVATE GTest::gtest_main CUDA::cudart)

gtest_discover_tests(tst)

set_target_properties(raytracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)