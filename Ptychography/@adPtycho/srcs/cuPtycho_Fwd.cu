 /* =========================================================================
 cuPthcho_Fwd.cu

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
void mexFunction(
    int nlhs, mxArray *plhs[], 
    int nrhs, mxArray const *  __restrict__ prhs[]
){
    mxInitGPU();

    CHECK_THROW(mxIsGPUArray(prhs[0]));
    CHECK_THROW(mxIsGPUArray(prhs[1]));
    CHECK_THROW(mxIsGPUArray(prhs[2]));

    CST_GPU_PTR sampWave = mxGPUCreateFromMxArray(prhs[0]); 
    CST_GPU_PTR propBeam = mxGPUCreateFromMxArray(prhs[1]); 
    CST_GPU_PTR scansPos = mxGPUCreateFromMxArray(prhs[2]); 


    dim3 imLs = size2dim3(propBeam);
    dim3 imLs_sz = size2dim3(scansPos);

    dim3 imHs_sz = size2dim3(sampWave);
    dim3 imLs_bc = {imLs.x, imLs.y, imLs_sz.x};

    const mwSize Hsz[3] = {imLs_bc.x,imLs_bc.y,imLs_bc.z};

    // outputs of forward function
    mxGPUArray_t * latentW = mxGPUCreateGPUArray(
        3, Hsz,                   
        mxSINGLE_CLASS, mxCOMPLEX, 
        MX_GPU_DO_NOT_INITIALIZE //MX_GPU_INITIALIZE_VALUES
    );
    mxGPUArray_t * latentZ = mxGPUCreateGPUArray(
        3, Hsz,                   
        mxSINGLE_CLASS, mxCOMPLEX, 
        MX_GPU_DO_NOT_INITIALIZE //MX_GPU_INITIALIZE_VALUES
    );
    mxGPUArray_t * observeI = mxGPUCreateGPUArray(
        3, Hsz,                   
        mxSINGLE_CLASS, mxREAL, 
        MX_GPU_DO_NOT_INITIALIZE 
    );

    const float2 * v_sampWave = getGPUDataRO<float2>(sampWave);
    const float2 * v_propBeam = getGPUDataRO<float2>(propBeam);
    const int2 * v_shiftspox   = getGPUDataRO<int2>(scansPos);
          
    float2 * v_latentW = (float2 *) mxGPUGetData(latentW);
    float2 * v_latentZ = (float2 *) mxGPUGetData(latentZ);
    float * v_observeI   = (float *) mxGPUGetData(observeI);


    dim3 N_THREADS = {BLOCK_X,BLOCK_Y,1};
    dim3 N_BLOCKS = {
        (unsigned) ((imLs_bc.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imLs_bc.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) imLs_bc.z
    };

    cufftHandle many_plan;
    int inembed[2];
    inembed[0] = (int) imLs_bc.x;
    inembed[1] = (int) imLs_bc.x;
    cufftPlanMany(
        &many_plan, 2, 
        &inembed[0], &inembed[0], 1,  imLs_bc.x * imLs_bc.y, // idist
        &inembed[0], 1, imLs_bc.x * imLs_bc.y, // idist
        CUFFT_C2C, imLs_bc.z
    );

    probeSubArea<<<N_BLOCKS, N_THREADS>>>(
        v_propBeam, v_sampWave, v_shiftspox,
        imLs_bc, imHs_sz,
        v_latentW,
        v_latentZ
    );

    cufftExecC2C(many_plan, 
        (cufftComplex *)&v_latentZ[0], 
        (cufftComplex *)&v_latentZ[0], 
        CUFFT_FORWARD
    );
    //cufftDestroy(many_plan);

    // cufftShift_2D_kernel<<<N_BLOCKS, N_THREADS>>>(
    //     v_latentZ,
    //     (int) imLs_bc.x
    // );

    intensityDetect<<<N_BLOCKS, N_THREADS>>>(
        imLs_bc,
        v_latentZ,
        v_observeI
    );

    plhs[0] = mxGPUCreateMxArrayOnGPU(latentW);
    plhs[1] = mxGPUCreateMxArrayOnGPU(latentZ);
    plhs[2] = mxGPUCreateMxArrayOnGPU(observeI);

    plhs[3] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *(uint64_t*)mxGetData(plhs[3]) = static_cast<uint64_t>(many_plan);   
    // delete output parameters
    mxGPUDestroyGPUArray(latentW);
    mxGPUDestroyGPUArray(latentZ);
    mxGPUDestroyGPUArray(observeI);
    // delete input parameters
    mxGPUDestroyGPUArray(sampWave);
    mxGPUDestroyGPUArray(propBeam);
    mxGPUDestroyGPUArray(scansPos);
}
