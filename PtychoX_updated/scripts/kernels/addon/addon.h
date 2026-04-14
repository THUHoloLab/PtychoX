#pragma once
#include <cuda_runtime.h>
#include <cufft.h>
#include <string>
#include <stdexcept>
#include <vector>

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)
#define CHECK_THROW(x) \
	do { if (!(x)) throw std::runtime_error(std::string(FILE_LINE " check failed " #x)); } while(0)

#define CST_GPU_PTR const mxGPUArray_t *

// using creal32_t = creal32_T;
using real32_t = real32_T;
using mxGPUArray_t = mxGPUArray;

typedef struct __align__(8) complex32 {
    float re;
    float im;
} creal32_t;

template <typename T>
const T* getGPUDataRO(const mxGPUArray* gpu_array) {
    return (const T* __restrict__) mxGPUGetDataReadOnly(gpu_array);
}

static dim3 size2dim3( const mxGPUArray * in){
    const mwSize *sz = mxGPUGetDimensions(in);
    const int dim = (int) mxGPUGetNumberOfDimensions(in);
    dim3 imgSz = {(unsigned) sz[1], (unsigned) sz[0], 1};
    if (dim > 2){
        imgSz.z = (unsigned) sz[2];
    }
    return imgSz;
}

template <typename T>
T* uint64ToPtr(const mxArray *arr) {
    return reinterpret_cast<T*>(
        static_cast<uintptr_t>(*(uint64_t*)mxGetData(arr))
    );
}

