#include "addon.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>

namespace cg = cooperative_groups;

__global__ void probeSubArea(
    const float2 * __restrict__ probe,
    const float2 * __restrict__ sample,
    const int2 * __restrict__ position,
    const dim3 imgLSz,
    const dim3 imgHSz,
    // output
    float2 * __restrict__ latentW,
    float2 * __restrict__ latentX
){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z;

    const bool inside = (idx < imgLSz.x) && (idy < imgLSz.y);
    if (!inside)
        return;

    const int pixId  = idx * imgLSz.y + idy;
    const int pageId = idz * (imgLSz.x * imgLSz.y);
    const int2 pos   = position[idz];

    const int pixId_large = (pos.x + idx - 1) * imgHSz.y + (pos.y + idy - 1);

    const float2 this_P = probe[pixId];
    const float2 this_W = sample[pixId_large];

    float2 this_O = make_float2(
        this_W.x * this_P.x - this_W.y * this_P.y,
        this_W.y * this_P.x + this_W.x * this_P.y
    );
    latentW[pixId + pageId] = this_W;
    latentX[pixId + pageId] = this_O;
}

// __global__ void intensityDetect(
//     const dim3 imgSz,
//     float2* __restrict__ forward, 
//     float* __restrict__ observes
// ){
//     unsigned xIndex = blockIdx.x * blockDim.x + threadIdx.x; 
//     unsigned yIndex = blockIdx.y * blockDim.y + threadIdx.y;
//     unsigned zIndex = blockIdx.z;  

//     const bool inside = (xIndex < imgSz.x) && (yIndex < imgSz.y);

//     if(inside){
//         // 2D Slice & 1D Line
//         unsigned sSlice = imgSz.x * imgSz.y;
//         unsigned index = (xIndex * imgSz.y) + yIndex + sSlice * zIndex;
//         float2 temp = forward[index];
//         observes[index] = sqrtf(temp.x * temp.x + temp.y * temp.y);
//     }
// }

__global__ void intensityDetect(
    const dim3 imgSz,
    float2* __restrict__ forward, 
    float* __restrict__ observes
){
    unsigned xIndex = blockIdx.x * blockDim.x + threadIdx.x; 
    unsigned yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned zIndex = blockIdx.z;  

    const bool inside = (xIndex < imgSz.x) && (yIndex < imgSz.y);

    if(inside){
        // 2D Slice & 1D Line
        int sSlice = imgSz.x * imgSz.y;
        // Transformations Equations
        int sEq1 = (sSlice + imgSz.x) / 2;
        int sEq2 = (sSlice - imgSz.y) / 2;

        // Thread Index Converted into 1D Index
        int index = (yIndex * imgSz.x) + xIndex + sSlice * zIndex;
        float2 regTemp;
        float2 temp;
        float this_I; 

        if (xIndex < imgSz.x / 2){
            if (yIndex < imgSz.y / 2){
                regTemp = forward[index];
                temp = forward[index + sEq1];
                // First Quad
                forward[index] = temp;
                // Third Quad
                forward[index + sEq1] = regTemp;

                this_I = sqrtf(temp.x * temp.x + temp.y * temp.y);
                observes[index] = this_I;
                this_I = sqrtf(regTemp.x * regTemp.x + regTemp.y * regTemp.y);
                observes[index + sEq1] = this_I;
            }
        }else{
            if (yIndex < imgSz.y / 2){
                regTemp = forward[index];
                temp = forward[index + sEq2];
                // Second Quad
                forward[index] = temp;
                // Fourth Quad
                forward[index + sEq2] = regTemp;

                this_I = sqrtf(temp.x * temp.x + temp.y * temp.y);
                observes[index] = this_I;
                this_I = sqrtf(regTemp.x * regTemp.x + regTemp.y * regTemp.y);
                observes[index + sEq2] = this_I;
            }
        }
    }
}

__global__ void setConstraints(
    const float * __restrict__ dldout,
    const dim3 imgSz,
    float2 * __restrict__ forward
){
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned idz = blockIdx.z;

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y);
    if (!inside)
        return;

    unsigned pixId = idx * imgSz.y + idy;
    unsigned pageId = idz * imgSz.x * imgSz.y;

    float2 this_F = forward[pixId + pageId];
    float this_I = dldout[pixId + pageId] / ((float) imgSz.x * imgSz.y);

    float ang = atan2f(this_F.y,this_F.x);
    this_F = make_float2(
        __cosf(ang) * this_I, 
        __sinf(ang) * this_I
    );
    forward[pixId + pageId] = this_F;
}

__global__ void reducedSum(
    float2 * __restrict__ dldw1,
    float2 * __restrict__ dldw2,
    const float2 * __restrict__ probe,
    const float2 * __restrict__ forward,
    const float2 * __restrict__ latentW,
    const int2 * __restrict__ position,
    const dim3 imgSz,
    const dim3 imgHsz
){
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned idz = blockIdx.z;

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y);
    if (!inside)
        return;

    unsigned pixId = idx * imgSz.y + idy;
    unsigned pageId = idz * imgSz.x * imgSz.y;

    float2 this_W = latentW[pixId + pageId];
    float2 this_P = probe[pixId];
    float2 this_X = forward[pixId + pageId];

    float2 temp = make_float2(
        this_X.x * this_W.x + this_X.y * this_W.y,
        this_X.x * this_W.y - this_X.y * this_W.x
    );

    // temp.x = this_X.x * this_W.x + this_X.y * this_W.y;
    // temp.y = this_X.x * this_W.y - this_X.y * this_W.x; // conj
    float * temp_w = (float *)dldw2;
    atomicAdd(temp_w + 2 * pixId    , temp.x);
    atomicAdd(temp_w + 2 * pixId + 1, temp.y);

    const int2 pos = position[idz];
    const int pixId_large = (pos.x + idx - 1) * imgHsz.y + (pos.y + idy - 1);

    temp = make_float2(
        this_X.x * this_P.x + this_X.y * this_P.y,
        this_X.x * this_P.y - this_X.y * this_P.x
    );
    // temp.x = this_X.x * this_P.x + this_X.y * this_P.y;
    // temp.y = this_X.x * this_P.y - this_X.y * this_P.x; // conj
    float * temp_s = (float *)dldw1;
    atomicAdd(temp_s + 2 * pixId_large    , temp.x);
    atomicAdd(temp_s + 2 * pixId_large + 1, temp.y);
}   


__global__ void fullyfused_ConsShift(
    const float * __restrict__ dldout,
    const dim3 imgSz,
    float2 * __restrict__ forward
){
    unsigned xIndex = blockIdx.x * blockDim.x + threadIdx.x; 
    unsigned yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned zIndex = blockIdx.z;  

    const bool inside = (xIndex < imgSz.x) && (yIndex < imgSz.y);

    if(!inside)
        return;

    // 2D Slice & 1D Line
    int sSlice = imgSz.x * imgSz.y;
    // Transformations Equations
    int sEq1 = (sSlice + imgSz.x) / 2;
    int sEq2 = (sSlice - imgSz.y) / 2;

    // Thread Index Converted into 1D Index
    int index = (yIndex * imgSz.x) + xIndex + sSlice * zIndex;

    float2 reg_F;
    float2 tem_F;

    float reg_I;
    float tem_I;
    float this_I; 
    float ang;
    float ratio = 1.0f / ((float) (imgSz.x * imgSz.y));

    if (xIndex < imgSz.x / 2){
        if (yIndex < imgSz.y / 2){
            reg_F = forward[index];
            tem_F = forward[index + sEq1];

            reg_I = dldout[index] * ratio;
            tem_I = dldout[index + sEq1] * ratio;

            // First Quad
            ang = atan2f(tem_F.y,tem_F.x);
            // tem_F.x = __cosf(ang) * tem_I;
            // tem_F.y = __sinf(ang) * tem_I;
            tem_F = make_float2( __cosf(ang) * tem_I, __sinf(ang) * tem_I);
            forward[index] = tem_F;
            // Third Quad
            ang = atan2f(reg_F.y,reg_F.x);
            // reg_F.x = __cosf(ang) * reg_I;
            // reg_F.y = __sinf(ang) * reg_I;
            reg_F = make_float2( __cosf(ang) * reg_I, __sinf(ang) * reg_I);
            forward[index + sEq1] = reg_F;
        }
    }else{
        if (yIndex < imgSz.y / 2){
            reg_F = forward[index];
            tem_F = forward[index + sEq2];

            reg_I = dldout[index] * ratio;
            tem_I = dldout[index + sEq2] * ratio;

            // Second Quad
            ang = atan2f(tem_F.y,tem_F.x);
            tem_F = make_float2( __cosf(ang) * tem_I, __sinf(ang) * tem_I);
            forward[index] = tem_F;
            // Fourth Quad
            ang = atan2f(reg_F.y,reg_F.x);
            reg_F = make_float2( __cosf(ang) * reg_I, __sinf(ang) * reg_I);
            forward[index + sEq2] = reg_F;
        }
    }
}