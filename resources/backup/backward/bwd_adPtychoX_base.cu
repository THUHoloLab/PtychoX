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

#include "mex/mex.h"
#include "mex/mxGPUArray.h"
#include "kernels/addon/addon.h"

#include "kernels/kernels_base.cuh"

void mexFunction(
    int nlhs, mxArray *plhs[],
    int nrhs, mxArray const * __restrict__ prhs[]
){
    mxInitGPU();

    CHECK_THROW(nrhs == 7);
    CHECK_THROW(mxIsGPUArray(prhs[0]));
    CHECK_THROW(mxIsGPUArray(prhs[1]));
    CHECK_THROW(mxIsGPUArray(prhs[2]));
    CHECK_THROW(mxIsGPUArray(prhs[3]));

    const mxGPUArray_t * dldout = mxGPUCreateFromMxArray(prhs[0]);
    const mxGPUArray_t * sample = mxGPUCreateFromMxArray(prhs[1]);
    const mxGPUArray_t * probe = mxGPUCreateFromMxArray(prhs[2]);
    const mxGPUArray_t * position = mxGPUCreateFromMxArray(prhs[3]);

    dim3 imLs_bc = size2dim3(dldout);
    dim3 imHs_sz = size2dim3(sample);

    mxGPUArray_t * dldw1 = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(sample),
        mxGPUGetDimensions(sample),
        mxSINGLE_CLASS, mxCOMPLEX,
        MX_GPU_INITIALIZE_VALUES
    );
    mxGPUArray_t * dldw2 = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(probe),
        mxGPUGetDimensions(probe),
        mxSINGLE_CLASS, mxCOMPLEX,
        MX_GPU_INITIALIZE_VALUES
    );

    dim3 N_THREADS = {BLOCK_X, BLOCK_Y, 1};
    dim3 N_BLOCKS = {
        (unsigned) ((imLs_bc.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imLs_bc.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) imLs_bc.z
    };

    const float2 * v_probe = getGPUDataRO<float2>(probe);
    const int2 * v_position = getGPUDataRO<int2>(position);
    const float * v_dldout = getGPUDataRO<float>(dldout);

    cufftHandle many_plan = static_cast<cufftHandle>(*(uint64_t*)mxGetData(prhs[4]));
    const float2 * v_latentW = reinterpret_cast<const float2*>(uint64ToPtr<creal32_t>(prhs[5]));
    float2 * v_latentZ = reinterpret_cast<float2*>(uint64ToPtr<creal32_t>(prhs[6]));

    float2 * v_dldw1 = (float2 *) mxGPUGetData(dldw1);
    float2 * v_dldw2 = (float2 *) mxGPUGetData(dldw2);

    fullyfused_ConsShift<<<N_BLOCKS, N_THREADS>>>(
        v_dldout,
        imLs_bc,
        v_latentZ
    );

    cufftExecC2C(
        many_plan,
        (cufftComplex *)&v_latentZ[0],
        (cufftComplex *)&v_latentZ[0],
        CUFFT_INVERSE
    );

    N_BLOCKS.z = 1;
    reducedSum<<<N_BLOCKS, N_THREADS>>>(
        v_dldw1, v_dldw2,
        v_probe, v_latentZ, v_latentW, v_position,
        imLs_bc, imHs_sz
    );

    plhs[0] = mxGPUCreateMxArrayOnGPU(dldw1);
    plhs[1] = mxGPUCreateMxArrayOnGPU(dldw2);

    mxGPUDestroyGPUArray(dldw1);
    mxGPUDestroyGPUArray(dldw2);
    mxGPUDestroyGPUArray(dldout);
    mxGPUDestroyGPUArray(probe);
    mxGPUDestroyGPUArray(sample);
    mxGPUDestroyGPUArray(position);
}

