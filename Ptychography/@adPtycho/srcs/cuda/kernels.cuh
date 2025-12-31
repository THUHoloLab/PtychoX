 /* =========================================================================
 kernels.cuh

 Author: Shuhe Zhang
 Affiliation: Tsinghua University
 Email: shuhe-zhang@tsinghua.edu.cn

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 =========================================================================*/

#include "addon.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>

namespace cg = cooperative_groups;

/**
 * probeSubArea
 * ------------
 * For each scan position (idz), crop a sub-area from the large sample (W)
 * and multiply it with the probe (P) to form the exit-wave (O = W .* P).
 *
 * Data layout assumptions:
 * - probe:      [imgLSz.x, imgLSz.y] (single probe, shared by all positions)
 * - sample:     [imgHSz.x, imgHSz.y] (large object / sample)
 * - position:   [nViews] each is int2(pos.x, pos.y), giving the top-left-ish
 *              coordinate of the sub-area in the large sample
 * - latentW:    [imgLSz.x, imgLSz.y, nViews] cropped sample patch per view
 * - latentX:    [imgLSz.x, imgLSz.y, nViews] exit wave per view (O)
 *
 * Indexing convention:
 * - local pixel index: pixId = idx * imgLSz.y + idy  (column-major like (x,y)->x*H+y)
 * - pageId = idz * (imgLSz.x*imgLSz.y)
 *
 * Note:
 * - `pos.x + idx - 1` and `pos.y + idy - 1` implies `position` likely uses
 *   1-based indexing (MATLAB-style). If position is already 0-based, remove -1.
 */
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

/**
 * intensityDetect
 * ---------------
 * Compute intensity magnitude from complex field:
 *   I = |F| = sqrt(Re^2 + Im^2)
 *
 * In addition, this version performs an in-place quadrant swap ("fftshift"-like)
 * on `forward` *while computing intensity*, to avoid launching a separate kernel.
 *
 * IMPORTANT:
 * - This kernel only processes half of the image (x < Nx/2, y < Ny/2) and
 *   swaps with the corresponding quadrant partner. That ensures each pair
 *   is swapped exactly once (no double swapping).
 *
 * Indexing:
 * - forward / observes are treated as batched 2D arrays with batch = zIndex
 * - flattened index uses row-major style: index = y * Nx + x + slice*z
 *   (NOTE: this differs from probeSubArea's x*Ny+y convention!)
 *
 * Safety:
 * - This assumes imgSz.x and imgSz.y are even. If odd, boundary handling is needed.
 */
__global__ void intensityDetect(
    const dim3 imgSz,                 // (Nx, Ny, nViews)
    float2* __restrict__ forward,     // complex field to be shifted and measured
    float* __restrict__ observes      // output intensity magnitude (same layout)
){
    unsigned xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned zIndex = blockIdx.z;

    const bool inside = (xIndex < imgSz.x) && (yIndex < imgSz.y);
    if (!inside) return;

    // Per-view slice size
    int sSlice = imgSz.x * imgSz.y;

    // Offsets that map quadrant pairs for an (Nx, Ny) array
    // These are derived for a specific flattening order (y * Nx + x).
    // sEq1 maps Q1 <-> Q3, sEq2 maps Q2 <-> Q4.
    int sEq1 = (sSlice + imgSz.x) / 2;
    int sEq2 = (sSlice - imgSz.y) / 2;

    // Flatten index (row-major)
    int index = (yIndex * imgSz.x) + xIndex + sSlice * zIndex;

    float2 regTemp; // original value at current index
    float2 temp;    // value at paired quadrant index
    float this_I;

    // Only perform swaps for y < Ny/2 (top half), and split x for left/right half.
    // This prevents performing the same swap twice.
    if (xIndex < imgSz.x / 2) {
        if (yIndex < imgSz.y / 2) {
            // Q1 <-> Q3
            regTemp = forward[index];
            temp    = forward[index + sEq1];

            forward[index]        = temp;
            forward[index + sEq1] = regTemp;

            // Intensity for both swapped locations
            this_I = sqrtf(temp.x * temp.x + temp.y * temp.y);
            observes[index] = this_I;

            this_I = sqrtf(regTemp.x * regTemp.x + regTemp.y * regTemp.y);
            observes[index + sEq1] = this_I;
        }
    } else {
        if (yIndex < imgSz.y / 2) {
            // Q2 <-> Q4
            regTemp = forward[index];
            temp    = forward[index + sEq2];

            forward[index]        = temp;
            forward[index + sEq2] = regTemp;

            this_I = sqrtf(temp.x * temp.x + temp.y * temp.y);
            observes[index] = this_I;

            this_I = sqrtf(regTemp.x * regTemp.x + regTemp.y * regTemp.y);
            observes[index + sEq2] = this_I;
        }
    }
}

/**
 * setConstraints
 * --------------
 * Apply measured-intensity constraint in Fourier domain:
 *   forward := (forward / |forward|) * I_target
 *
 * Here I_target is provided by dldout (naming suggests gradient from MATLAB),
 * and is normalized by (Nx*Ny). The kernel keeps the current phase of forward
 * but enforces a new magnitude.
 *
 * Note:
 * - This version does NOT do any quadrant shift.
 * - It uses an x*Ny+y indexing (different from intensityDetect!).
 *   Make sure the calling code matches this convention.
 */
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

/**
 * reducedSum
 * ----------
 * Accumulate gradients for sample (W) and probe (P) using atomic adds.
 *
 * Given forward field X (likely the Fourier-domain residual propagated back),
 * and cached latentW (cropped sample patch), probe P:
 *
 * - Gradient wrt probe at patch pixel:
 *     dL/dP += conj(W) * X   (or a closely related form depending on convention)
 *
 * - Gradient wrt sample at corresponding large-sample pixel:
 *     dL/dW += conj(P) * X
 *
 * Because multiple scan positions overlap on the large sample, the update to
 * dldw1 (sample gradient) requires atomicAdd.
 *
 * dldw1: gradient buffer over the large sample (size imgHsz.x * imgHsz.y)
 * dldw2: gradient buffer over the probe patch (size imgSz.x * imgSz.y)
 *
 * NOTE:
 * - dldw1/dldw2 are float2*, but atomicAdd works on float. So the code
 *   reinterprets them as float* and does atomicAdd on real/imag separately.
 */
__global__ void reducedSum(
    float2 * __restrict__ dldw1,        // [imgHsz.x, imgHsz.y] gradient for sample W (complex)
    float2 * __restrict__ dldw2,        // [imgSz.x,  imgSz.y ] gradient for probe  P (complex)
    const float2 * __restrict__ probe,  // probe P in patch coordinates
    const float2 * __restrict__ forward,// X (backpropagated field) in patch coords, per view
    const float2 * __restrict__ latentW,// cached W patch per view
    const int2 * __restrict__ position, // scan positions
    const dim3 imgSz,                   // patch size
    const dim3 imgHsz                   // large sample size
){
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned idz = blockIdx.z;

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y);
    if (!inside) return;

    unsigned pixId  = idx * imgSz.y + idy;
    unsigned pageId = idz * imgSz.x * imgSz.y;

    // Cached patch sample W and probe P at current patch pixel
    float2 this_W = latentW[pixId + pageId];
    float2 this_P = probe[pixId];

    // Backprop field X at current patch pixel
    float2 this_X = forward[pixId + pageId];

    // ---- Accumulate gradient for probe (dldw2) ----
    // temp = conj(W) * X  (depending on sign convention)
    float2 temp = make_float2(
        this_X.x * this_W.x + this_X.y * this_W.y,
        this_X.x * this_W.y - this_X.y * this_W.x
    );

    float * temp_w = (float *)dldw2;
    atomicAdd(temp_w + 2 * pixId    , temp.x);
    atomicAdd(temp_w + 2 * pixId + 1, temp.y);

    // ---- Accumulate gradient for sample (dldw1) ----
    const int2 pos = position[idz];
    const int pixId_large = (pos.x + idx - 1) * imgHsz.y + (pos.y + idy - 1);

    // temp = conj(P) * X  (depending on sign convention)
    temp = make_float2(
        this_X.x * this_P.x + this_X.y * this_P.y,
        this_X.x * this_P.y - this_X.y * this_P.x
    );

    float * temp_s = (float *)dldw1;
    atomicAdd(temp_s + 2 * pixId_large    , temp.x);
    atomicAdd(temp_s + 2 * pixId_large + 1, temp.y);
}

/**
 * fullyfused_ConsShift
 * --------------------
 * Fully fused kernel: (1) quadrant swap (fftshift-like) + (2) magnitude constraint.
 *
 * It does the same "swap once" strategy as intensityDetect:
 * - Only threads in the top half (y < Ny/2) participate.
 * - Left half swaps Q1<->Q3, right half swaps Q2<->Q4.
 *
 * Then for each swapped element, it enforces:
 *   forward := exp(i*angle(forward)) * (dldout / (Nx*Ny))
 *
 * Inputs:
 * - dldout: target magnitudes (or measured amplitude)
 * - forward: complex field, updated in-place with shift+constraint
 *
 * Notes:
 * - Like intensityDetect, the flattening order is index = y*Nx + x + slice*z.
 * - imgSz.x and imgSz.y are assumed even.
 */
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
