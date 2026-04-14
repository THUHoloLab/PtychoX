#pragma once

#include "addon/addon.h"

#define PI 3.141592654f
#define TwoPI 6.28318531f

using cplx32_t = creal32_t;

__host__ __device__ inline cplx32_t make_complex(float a, float b){
    cplx32_t out;
    out.re = a;
    out.im = b;
    return out;
}
