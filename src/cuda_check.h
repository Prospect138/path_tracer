#pragma once
#include <cstdio>
#include <cuda_runtime.h>
#include <exception>
#include <iostream>

#define CUDA_CHECK(fnc)                                                                                                \
    {                                                                                                                  \
        gpuAssert((fnc), __FILE__, __LINE__);                                                                          \
    }
inline void gpuAssert(cudaError_t err, const char *file, int line, bool abort = true)
{
    if (err != cudaError::cudaSuccess)
    {
        fprintf(stderr, "Error: CUDA fail: %s in %s at line %i.\n", cudaGetErrorString(err), file, line);
        if (abort)
        {
            printf("Abort.");
            std::terminate();
        }
    }
}