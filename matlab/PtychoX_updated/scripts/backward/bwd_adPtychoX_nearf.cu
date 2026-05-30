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

    // This MEX expects:
    // 0 dldout, 1 observ, 2 codedSurf, 3 diffracH2, 4 shiftspox,
    // 5 plan, 6 many_plan, 7 v_forward, 8 v_X_record, 9 downSam_ratio.
    CHECK_THROW(nrhs >= 10);
    CHECK_THROW(nlhs <= 2);

    CHECK_THROW(mxIsGPUArray(prhs[0]));
    CHECK_THROW(mxIsGPUArray(prhs[1]));
    CHECK_THROW(mxIsGPUArray(prhs[2]));
    CHECK_THROW(mxIsGPUArray(prhs[3]));
    CHECK_THROW(mxIsGPUArray(prhs[4]));

    CST_GPU_PTR dldout = mxGPUCreateFromMxArray(prhs[0]);
    CST_GPU_PTR observ = mxGPUCreateFromMxArray(prhs[1]);
    CST_GPU_PTR codedSurf = mxGPUCreateFromMxArray(prhs[2]);
    CST_GPU_PTR diffracH2 = mxGPUCreateFromMxArray(prhs[3]);
    CST_GPU_PTR shiftspox = mxGPUCreateFromMxArray(prhs[4]);

    cufftHandle plan = static_cast<cufftHandle>(*(uint64_t*)mxGetData(prhs[5]));
    cufftHandle many_plan = static_cast<cufftHandle>(*(uint64_t*)mxGetData(prhs[6]));
    creal32_t * __restrict__ v_forward = uint64ToPtr<creal32_t>(prhs[7]);
    creal32_t * __restrict__ v_X_record = uint64ToPtr<creal32_t>(prhs[8]);
    int downSam_ratio = (int) mxGetScalar(prhs[9]);

    // Low-resolution detector size comes from the observation tensor,
    // while the high-resolution field size follows codedSurf.
    dim3 codedSurf_sz = size2dim3(codedSurf);
    dim3 imLs_bc = size2dim3(observ);
    dim3 imHs_bc = {codedSurf_sz.x, codedSurf_sz.y, imLs_bc.z};

    mwSize Hsz[2] = {imHs_bc.x, imHs_bc.y};

    mxGPUArray_t * dldw1 = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(diffracH2),
        Hsz,
        mxSINGLE_CLASS, mxCOMPLEX,
        MX_GPU_DO_NOT_INITIALIZE
    );
    mxGPUArray_t * dldw2 = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(diffracH2),
        Hsz,
        mxSINGLE_CLASS, mxCOMPLEX,
        MX_GPU_DO_NOT_INITIALIZE
    );

    const creal32_t * __restrict__ v_codedSurf = getGPUDataRO<creal32_t>(codedSurf);
    const creal32_t * __restrict__ v_diffracH2 = getGPUDataRO<creal32_t>(diffracH2);
    const float2 * __restrict__ v_shiftspox = getGPUDataRO<float2>(shiftspox);
    const float * __restrict__ v_dldout = getGPUDataRO<float>(dldout);
    const float * __restrict__ v_observ = getGPUDataRO<float>(observ);

    creal32_t * __restrict__ v_dldw1 = (creal32_t *) mxGPUGetData(dldw1);
    creal32_t * __restrict__ v_dldw2 = (creal32_t *) mxGPUGetData(dldw2);

    dim3 N_THREADS = {BLOCK_X, BLOCK_Y, 1};
    dim3 N_BLOCKS = {
        (unsigned) ((imHs_bc.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imHs_bc.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) imHs_bc.z
    };
    dim3 N_BLOCKS_S = {
        (unsigned) ((imLs_bc.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imLs_bc.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) imLs_bc.z
    };

    // Backpropagate through the detector downsampling stage into v_forward.
    fullyfused_DownSample_Bwd<<<N_BLOCKS_S, N_THREADS>>>(
        v_dldout,
        v_observ,
        downSam_ratio,
        imLs_bc,
        v_forward
    );

    // Move the gradient to the frequency domain and apply the adjoint
    // propagation kernel conj(diffracH2).
    cufftExecC2C(
        many_plan,
        (cufftComplex *)&v_forward[0],
        (cufftComplex *)&v_forward[0],
        CUFFT_FORWARD
    );

    fused_pixelwiseProduct_inplace_conj<<<N_BLOCKS, N_THREADS>>>(
        v_diffracH2,
        imHs_bc,
        v_forward
    );

    cufftExecC2C(
        many_plan,
        (cufftComplex *)&v_forward[0],
        (cufftComplex *)&v_forward[0],
        CUFFT_INVERSE
    );

    // Split the gradient into:
    // 1) dldw2: coded surface term.
    // 2) v_forward: gradient w.r.t. the shifted high-resolution field.
    N_BLOCKS.z = 1;
    deconvCodedSurf<<<N_BLOCKS, N_THREADS>>>(
        v_codedSurf,
        v_X_record,
        imHs_bc,
        // output
        v_forward,
        v_dldw2
    );

    // Accumulate scan-wise contributions back to a single high-resolution
    // wavefront and transform it to match the expected output domain.
    cufftExecC2C(
        many_plan,
        (cufftComplex *)&v_forward[0],
        (cufftComplex *)&v_forward[0],
        CUFFT_FORWARD
    );
    fullyfused_ReducedSum<<<N_BLOCKS, N_THREADS>>>(
        v_forward,
        // v_diffracH1,
        v_shiftspox,
        imHs_bc,
        v_dldw1
    );
    cufftExecC2C(
        plan,
        (cufftComplex *)&v_dldw1[0],
        (cufftComplex *)&v_dldw1[0],
        CUFFT_FORWARD
    );

    // Return gradients for the two learnable branches.
    plhs[0] = mxGPUCreateMxArrayOnGPU(dldw1);
    plhs[1] = mxGPUCreateMxArrayOnGPU(dldw2);

    mxGPUDestroyGPUArray(dldw1);
    mxGPUDestroyGPUArray(dldw2);
    mxGPUDestroyGPUArray(dldout);
    mxGPUDestroyGPUArray(observ);
    mxGPUDestroyGPUArray(codedSurf);
    mxGPUDestroyGPUArray(diffracH2);
    mxGPUDestroyGPUArray(shiftspox);
}

