#include "mex/mex.h"
#include "mex/mxGPUArray.h"
#include "kernels/addon/addon.h"

#include "kernels/cplx_number.cuh"
#include "kernels/kernels_nearf.cuh"

void mexFunction(
    int nlhs, mxArray *plhs[],
    int nrhs, mxArray const *  __restrict__ prhs[]
){
    mxInitGPU();

    CHECK_THROW(mxIsGPUArray(prhs[0]));
    CHECK_THROW(mxIsGPUArray(prhs[1]));
    CHECK_THROW(mxIsGPUArray(prhs[2]));
    CHECK_THROW(mxIsGPUArray(prhs[3]));

    CST_GPU_PTR wavefront1 = mxGPUCreateFromMxArray(prhs[0]);
    CST_GPU_PTR wavefront2 = mxGPUCreateFromMxArray(prhs[1]);
    CST_GPU_PTR diffracH2 = mxGPUCreateFromMxArray(prhs[2]);
    CST_GPU_PTR shiftspox = mxGPUCreateFromMxArray(prhs[3]);

    cufftHandle plan = static_cast<cufftHandle>(*(uint64_t*)mxGetData(prhs[4]));
    cufftHandle many_plan = static_cast<cufftHandle>(*(uint64_t*)mxGetData(prhs[5]));

    creal32_t * __restrict__ v_wavefront1 = uint64ToPtr<creal32_t>(prhs[6]);
    creal32_t * __restrict__ v_X_record = uint64ToPtr<creal32_t>(prhs[7]);
    creal32_t * __restrict__ v_X_forward = uint64ToPtr<creal32_t>(prhs[8]);
    int downSam_ratio = (int) mxGetScalar(prhs[9]);

    dim3 imLs_sz = size2dim3(shiftspox);
    dim3 imHs_sz = size2dim3(wavefront1);
    dim3 imHs_bc = {imHs_sz.x, imHs_sz.y, imLs_sz.x};
    dim3 imLs_bc = {imHs_sz.x / downSam_ratio,
                    imHs_sz.y / downSam_ratio,
                    imLs_sz.x};

    const mwSize Lsz[3] = {imLs_bc.x, imLs_bc.y, imLs_sz.x};
    mxGPUArray_t * observeds = mxGPUCreateGPUArray(
        3, Lsz,
        mxSINGLE_CLASS, mxREAL,
        MX_GPU_DO_NOT_INITIALIZE
    );

    const creal32_t * __restrict__ v_wavefront1_in  = getGPUDataRO<creal32_t>(wavefront1);
    const creal32_t * __restrict__ v_wavefront2     = getGPUDataRO<creal32_t>(wavefront2);
    const creal32_t * __restrict__ v_diffracH2      = getGPUDataRO<creal32_t>(diffracH2);
    const float2 * __restrict__ v_shiftspox         = getGPUDataRO<float2>(shiftspox);
    float * __restrict__ v_observeds = (float *) mxGPUGetData(observeds);

    dim3 N_THREADS = {BLOCK_X, BLOCK_Y, 1};
    dim3 N_BLOCKS = {
        (unsigned) ((imHs_sz.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imHs_sz.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) imHs_bc.z
    };
    cudaMemcpy(
        v_wavefront1, v_wavefront1_in,
        (size_t) imHs_sz.x * (size_t) imHs_sz.y * sizeof(creal32_t),
        cudaMemcpyDeviceToDevice
    );
    cufftExecC2C(
        plan,
        (cufftComplex *)&v_wavefront1[0],
        (cufftComplex *)&v_wavefront1[0],
        CUFFT_FORWARD
    );

    fullyfused_shiftNprop<<<N_BLOCKS, N_THREADS>>>(
        v_wavefront1,
        v_shiftspox,
        imHs_bc,
        v_X_record
    );

    cufftExecC2C(
        many_plan,
        (cufftComplex *)&v_X_record[0],
        (cufftComplex *)&v_X_record[0],
        CUFFT_INVERSE
    );

    fused_pixelwiseProduct<<<N_BLOCKS, N_THREADS>>>(
        v_X_record,
        v_wavefront2,
        imHs_bc,
        v_X_forward
    );

    // do ASM propagation
    cufftExecC2C(
        many_plan,
        (cufftComplex *)&v_X_forward[0],
        (cufftComplex *)&v_X_forward[0],
        CUFFT_FORWARD
    );
    fused_pixelwiseProduct_inplace<<<N_BLOCKS, N_THREADS>>>(
        v_diffracH2,
        imHs_bc,
        v_X_forward
    );
    cufftExecC2C(
        many_plan,
        (cufftComplex *)&v_X_forward[0],
        (cufftComplex *)&v_X_forward[0],
        CUFFT_INVERSE
    );
    fullyfused_DownSample_Fwd<<<N_BLOCKS, N_THREADS>>>(
        v_X_forward,
        (int) downSam_ratio,
        imLs_bc,
        v_observeds
    );

    plhs[0] = mxGPUCreateMxArrayOnGPU(observeds);

    mxGPUDestroyGPUArray(observeds);
    mxGPUDestroyGPUArray(wavefront1);
    mxGPUDestroyGPUArray(wavefront2);
    mxGPUDestroyGPUArray(diffracH2);
    mxGPUDestroyGPUArray(shiftspox);
}

