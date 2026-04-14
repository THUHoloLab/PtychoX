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

#include "mex/mex.h"
#include "mex/mxGPUArray.h"
#include "kernels/addon/addon.h"

#include "kernels/kernels_base.cuh"

void mexFunction(
    int nlhs, mxArray *plhs[],
    int nrhs, mxArray const * __restrict__ prhs[]
){
    mxInitGPU();

    CHECK_THROW(nrhs == 6);
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

    const mwSize Hsz[3] = {imLs_bc.x, imLs_bc.y, imLs_bc.z};
    mxGPUArray_t * observeI = mxGPUCreateGPUArray(
        3, Hsz,
        mxSINGLE_CLASS, mxREAL,
        MX_GPU_DO_NOT_INITIALIZE
    );

    const float2 * v_sampWave = getGPUDataRO<float2>(sampWave);
    const float2 * v_propBeam = getGPUDataRO<float2>(propBeam);
    const int2 * v_shiftspox = getGPUDataRO<int2>(scansPos);

    cufftHandle many_plan = static_cast<cufftHandle>(*(uint64_t*)mxGetData(prhs[3]));
    float2 * v_latentW = reinterpret_cast<float2*>(uint64ToPtr<creal32_t>(prhs[4]));
    float2 * v_latentZ = reinterpret_cast<float2*>(uint64ToPtr<creal32_t>(prhs[5]));
    float * v_observeI = (float *) mxGPUGetData(observeI);

    dim3 N_THREADS = {BLOCK_X, BLOCK_Y, 1};
    dim3 N_BLOCKS = {
        (unsigned) ((imLs_bc.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imLs_bc.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) imLs_bc.z
    };

    probeSubArea<<<N_BLOCKS, N_THREADS>>>(
        v_propBeam, v_sampWave, v_shiftspox,
        imLs_bc, imHs_sz,
        v_latentW,
        v_latentZ
    );

    cufftExecC2C(
        many_plan,
        (cufftComplex *)&v_latentZ[0],
        (cufftComplex *)&v_latentZ[0],
        CUFFT_FORWARD
    );

    intensityDetect<<<N_BLOCKS, N_THREADS>>>(
        imLs_bc,
        v_latentZ,
        v_observeI
    );

    plhs[0] = mxGPUCreateMxArrayOnGPU(observeI);

    mxGPUDestroyGPUArray(observeI);
    mxGPUDestroyGPUArray(sampWave);
    mxGPUDestroyGPUArray(propBeam);
    mxGPUDestroyGPUArray(scansPos);
}

