 /* =========================================================================
 cuPthcho_Bwd.cu

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

#include "addon/mex.h"
#include "addon/mxGPUArray.h"
#include "addon/addon.h"

#include "cuda/kernels.cuh"
#include "cuda/fftshift_kernel.cuh"

/**
 * mexFunction (MATLAB entry)
 * --------------------------
 * This MEX function runs the backward/gradient path for a ptychography-like
 * pipeline on GPU:
 *
 * High-level flow (in this code):
 * 1) Read inputs (all are gpuArray):
 *    - dldout   : target magnitudes (or upstream gradient) in Fourier domain
 *    - sample   : large object/sample W (complex)
 *    - probe    : probe P (complex), patch-sized
 *    - position : scan positions (int2), one per view
 *    - latentW  : cached sample patch per view (complex), used for gradients
 *    - latentZ  : complex field to be constrained and then inverse FFT'ed
 *    - (prhs[6]) a cuFFT plan handle passed from MATLAB (uint64)
 *
 * 2) Allocate output gradients:
 *    - dldw1 : gradient wrt sample (same size as sample)
 *    - dldw2 : gradient wrt probe  (same size as probe)
 *
 * 3) Apply magnitude constraint + fftshift-like quadrant swap in Fourier domain:
 *    fullyfused_ConsShift(dldout, latentZ)
 *
 * 4) Inverse FFT (C2C) on latentZ using the provided cuFFT plan handle
 *
 * 5) Accumulate gradients with atomicAdd:
 *    reducedSum(dldw1, dldw2, probe, latentZ, latentW, position, ...)
 *
 * Outputs:
 * - plhs[0] = dldw1 (gpuArray complex single)
 * - plhs[1] = dldw2 (gpuArray complex single)
 */
void mexFunction(
    int nlhs, mxArray *plhs[], 
    int nrhs, mxArray const *  __restrict__ prhs[]
){
    // input parames
    mxInitGPU();

    CHECK_THROW(mxIsGPUArray(prhs[0])); // dldout
    CHECK_THROW(mxIsGPUArray(prhs[1])); // wavefront1
    CHECK_THROW(mxIsGPUArray(prhs[2])); // probe
    CHECK_THROW(mxIsGPUArray(prhs[3])); // position

    CHECK_THROW(mxIsGPUArray(prhs[4])); // latentW
    CHECK_THROW(mxIsGPUArray(prhs[5])); // latentZ

    const mxGPUArray_t * dldout   = mxGPUCreateFromMxArray(prhs[0]);
    const mxGPUArray_t * sample   = mxGPUCreateFromMxArray(prhs[1]);
    const mxGPUArray_t * probe    = mxGPUCreateFromMxArray(prhs[2]);
    const mxGPUArray_t * position = mxGPUCreateFromMxArray(prhs[3]);
    const mxGPUArray_t * latentW  = mxGPUCreateFromMxArray(prhs[5]);

    mxGPUArray_t * latentZ = mxGPUCopyFromMxArray(prhs[4]);

    // ------------------------------------------------------------
    // 2) Dimensions
    // ------------------------------------------------------------
    // imLs_bc: low-size batch cube (Nx, Ny, nViews), inferred from dldout
    // imHs_sz: high-size sample (Hx, Hy, 1 or ???), inferred from sample
    dim3 imLs_bc = size2dim3(dldout);   
    dim3 imHs_sz = size2dim3(sample);   

    // ------------------------------------------------------------
    // 3) Allocate output gradient buffers on GPU
    // ------------------------------------------------------------
    // dldw1: gradient w.r.t. sample (same size/type as sample)
    // dldw2: gradient w.r.t. probe  (same size/type as probe)
    //
    // MX_GPU_INITIALIZE_VALUES ensures output is zero-initialized. This is critical
    // because reducedSum uses atomicAdd accumulation.
    mxGPUArray_t * dldw1 = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(sample), 
        mxGPUGetDimensions(sample),                   
        mxSINGLE_CLASS, mxCOMPLEX, 
        MX_GPU_INITIALIZE_VALUES //MX_GPU_INITIALIZE_VALUES
    );
    mxGPUArray_t * dldw2 = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(probe), 
        mxGPUGetDimensions(probe),                   
        mxSINGLE_CLASS, mxCOMPLEX, 
        MX_GPU_INITIALIZE_VALUES //MX_GPU_INITIALIZE_VALUES
    );

    dim3 N_THREADS = {BLOCK_X,BLOCK_Y,1};
    dim3 N_BLOCKS = {
        (unsigned) ((imLs_bc.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imLs_bc.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) imLs_bc.z
    };

    const float2 * v_latentW = getGPUDataRO<float2>(latentW);
    const float2 * v_probe   = getGPUDataRO<float2>(probe);
    const int2 * v_position = getGPUDataRO<int2>(position);
    const float * v_dldout  = getGPUDataRO<float>(dldout);
   
    float2 * v_dldw1   = (float2 * ) mxGPUGetData(dldw1);
    float2 * v_dldw2   = (float2 * ) mxGPUGetData(dldw2);
    float2 * v_latentZ = (float2 * ) mxGPUGetData(latentZ);
    
    fullyfused_ConsShift<<<N_BLOCKS, N_THREADS>>>(
        v_dldout,
        imLs_bc,
        v_latentZ
    );
    // setConstraints<<<N_BLOCKS, N_THREADS>>>(
    //     v_dldout,
    //     imLs_bc,
    //     v_latentZ
    // );

    // cufftShift_2D_kernel<<<N_BLOCKS, N_THREADS>>>(
    //     v_latentZ,
    //     (int) imLs_bc.x
    // );

    // cufftHandle many_plan;
    int inembed[2];
    inembed[0] = (int) imLs_bc.x;
    inembed[1] = (int) imLs_bc.x;
    // cufftPlanMany(&many_plan, 2, 
    //     &inembed[0], &inembed[0], 1,  imLs_bc.x * imLs_bc.y, // idist
    //     &inembed[0], 1, imLs_bc.x * imLs_bc.y, // idist
    //     CUFFT_C2C, imLs_bc.z
    // );
    // uint64_t handle = *static_cast<uint64_t*>(mxGetData(prhs[6]));
    // cufftHandle many_plan = reinterpret_cast<cufftHandle>(handle);
    cufftHandle many_plan = static_cast<cufftHandle>(*(uint64_t*)mxGetData(prhs[6]));
    cufftExecC2C(many_plan, (cufftComplex *)&v_latentZ[0], (cufftComplex *)&v_latentZ[0], 
        CUFFT_INVERSE
    );
    cufftDestroy(many_plan);

    reducedSum<<<N_BLOCKS, N_THREADS>>>(
        v_dldw1, v_dldw2,
        v_probe, v_latentZ, v_latentW, v_position,
        imLs_bc, imHs_sz
    );

    plhs[0] = mxGPUCreateMxArrayOnGPU(dldw1);
    plhs[1] = mxGPUCreateMxArrayOnGPU(dldw2);


    mxGPUDestroyGPUArray(latentZ);
    mxGPUDestroyGPUArray(dldw1);
    mxGPUDestroyGPUArray(dldw2);

    mxGPUDestroyGPUArray(dldout);
    mxGPUDestroyGPUArray(probe);
    mxGPUDestroyGPUArray(sample);
    mxGPUDestroyGPUArray(position);
    mxGPUDestroyGPUArray(latentW);
}