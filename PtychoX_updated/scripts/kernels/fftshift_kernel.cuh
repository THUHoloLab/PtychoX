#include <cuda_runtime.h>
#include "addon/addon.h"

__global__ void cufftShift_2D_kernel(
    creal32_t* __restrict__ data,
    int N
){
    unsigned xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned zIndex = blockIdx.z;

    const bool inside = (xIndex < (unsigned) N) && (yIndex < (unsigned) N);

    if(inside){
        int sSlice = N * N;
        int sEq1 = (sSlice + N) / 2;
        int sEq2 = (sSlice - N) / 2;

        int index = (int)(yIndex * N) + (int)xIndex + sSlice * (int)zIndex;
        creal32_t regTemp;
        if (xIndex < (unsigned)(N / 2)){
            if (yIndex < (unsigned)(N / 2)){
                regTemp = data[index];
                data[index] = data[index + sEq1];
                data[index + sEq1] = regTemp;
            }
        }else{
            if (yIndex < (unsigned)(N / 2)){
                regTemp = data[index];
                data[index] = data[index + sEq2];
                data[index + sEq2] = regTemp;
            }
        }
    }
}
