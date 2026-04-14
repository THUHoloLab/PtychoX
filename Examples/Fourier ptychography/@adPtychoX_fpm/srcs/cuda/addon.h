#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdint>

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)

using creal32_t = creal32_T;
using real32_t = real32_T;
using mxGPUArray_t = mxGPUArray;


template <typename T>
const T* getGPUDataRO(const mxGPUArray* gpu_array) {
    return (const T* __restrict__) mxGPUGetDataReadOnly(gpu_array);
}

template <typename T>
T* uint64ToPtr(const mxArray *arr) {
    return reinterpret_cast<T*>(
        static_cast<uintptr_t>(*reinterpret_cast<uint64_t*>(mxGetData(arr)))
    );
}
