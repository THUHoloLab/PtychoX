#include "addon/mex.h"
#include "addon/mxGPUArray.h"
#include "addon/addon.h"

#include "cuda/cplx_number.cuh"
#include "cuda/kernels.cuh"
#include "cuda/cufftexes.cuh"

void mexFunction(
    int nlhs, mxArray *plhs[], 
    int nrhs, mxArray const *  __restrict__ prhs[]
){
    // input parames
    mxInitGPU();
    CHECK_THROW(mxIsGPUArray(prhs[0])); // dldo 
    CHECK_THROW(mxIsGPUArray(prhs[1]));
    CHECK_THROW(mxIsGPUArray(prhs[2]));
    CHECK_THROW(mxIsGPUArray(prhs[3]));
    CHECK_THROW(mxIsGPUArray(prhs[4]));
    CHECK_THROW(mxIsGPUArray(prhs[5]));
    CHECK_THROW(mxIsGPUArray(prhs[6]));
    CHECK_THROW(mxIsGPUArray(prhs[7]));

    // CST_GPU_PTR dldvout = mxGPUCreateFromMxArray(prhs[0]);
    CST_GPU_PTR dldvout = mxGPUCreateFromMxArray(prhs[0]);
    CST_GPU_PTR intensity = mxGPUCreateFromMxArray(prhs[1]);

    mxGPUArray_t * forward = mxGPUCopyFromMxArray(prhs[2]);

    CST_GPU_PTR codedSurf = mxGPUCreateFromMxArray(prhs[3]); 
    CST_GPU_PTR X_record  = mxGPUCreateFromMxArray(prhs[4]); 
    CST_GPU_PTR diffracH1 = mxGPUCreateFromMxArray(prhs[5]); 
    CST_GPU_PTR diffracH2 = mxGPUCreateFromMxArray(prhs[6]); 
    CST_GPU_PTR shiftspox = mxGPUCreateFromMxArray(prhs[7]);

    int pratio = (int) mxGetPr(prhs[8])[0];
    
    dim3 imHs_bc = size2dim3(forward);
    dim3 imLs_bc = size2dim3(dldvout);

    mwSize Hsz[2] = {imHs_bc.x,imHs_bc.y};

    mxGPUArray_t * dldw1 = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(diffracH1), 
        Hsz,                   
        mxSINGLE_CLASS, mxCOMPLEX, 
        MX_GPU_INITIALIZE_VALUES //MX_GPU_INITIALIZE_VALUES
    );

    mxGPUArray_t * dldw2 = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(diffracH1), 
        Hsz,                   
        mxSINGLE_CLASS, mxCOMPLEX, 
        MX_GPU_INITIALIZE_VALUES //MX_GPU_INITIALIZE_VALUES
    );

    // const creal32_t * v_dldvout   = getGPUDataRO<creal32_t>(dldvout);
    const float * v_dldvout   = getGPUDataRO<float>(dldvout);
    const float * v_intensity = getGPUDataRO<float>(intensity);

    const creal32_t * v_codedSurf = getGPUDataRO<creal32_t>(codedSurf);
    const creal32_t * v_X_record  = getGPUDataRO<creal32_t>(X_record);
    const creal32_t * v_diffracH1 = getGPUDataRO<creal32_t>(diffracH1);
    const creal32_t * v_diffracH2 = getGPUDataRO<creal32_t>(diffracH2);

    const int2 * v_shiftspox = getGPUDataRO<int2>(shiftspox);

    creal32_t * v_forward = (creal32_t *) mxGPUGetData(forward);
    creal32_t * v_dldw1   = (creal32_t *) mxGPUGetData(dldw1);
    creal32_t * v_dldw2   = (creal32_t *) mxGPUGetData(dldw2);
    
    dim3 N_THREADS = {BLOCK_X,BLOCK_Y,1};
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

    fullyfused_DownSample_Bwd<<<N_BLOCKS_S, N_THREADS>>>(
        v_dldvout,      // const float * __restrict__ dldout,
        v_intensity,    // const float * __restrict__ out,
        pratio,         // const int ds,
        imLs_bc,        // const dim3 imgSz,
        // output
        v_forward       // creal32_t * __restrict__ forward,
    );

    int inembed[2];
    inembed[0] = (int) imHs_bc.x;
    inembed[1] = (int) imHs_bc.x;

    cufftHandle many_plan;
    cufftPlanMany(
        &many_plan, 2, 
        &inembed[0], &inembed[0], 1,  imHs_bc.x * imHs_bc.y, // idist
        &inembed[0], 1, imHs_bc.x * imHs_bc.y, // idist
        CUFFT_C2C, imHs_bc.z
    );

    {// do ASP propagation
        cufftExecC2C(many_plan, 
            (cufftComplex *)&v_forward[0], 
            (cufftComplex *)&v_forward[0], 
            CUFFT_FORWARD
        );
        // done!
        fused_pixelwiseProduct_inplace_conj<<<N_BLOCKS, N_THREADS>>>(
            v_diffracH2, // propagation
            imHs_bc,
            v_forward
        );
        // ifft2d_many(imHs_bc, v_X_forward);
        cufftExecC2C(many_plan, 
            (cufftComplex *)&v_forward[0], 
            (cufftComplex *)&v_forward[0], 
            CUFFT_INVERSE
        );
    };cufftDestroy(many_plan);


    N_BLOCKS.z = 1;
    // fullyfused_AtomicShiftProduct_Bwd<<<N_BLOCKS, N_THREADS>>>(
    fullyfused_ShiftProduct_Bwd<<<N_BLOCKS, N_THREADS>>>(
        v_forward,
        v_codedSurf,
        v_X_record,
        v_shiftspox,
        imHs_bc,
        // output
        v_dldw1,
        v_dldw2
    );

    plhs[0] = mxGPUCreateMxArrayOnGPU(dldw1);
    plhs[1] = mxGPUCreateMxArrayOnGPU(dldw2);

    // cudaFree(latentW);

    mxGPUDestroyGPUArray(dldw1);
    mxGPUDestroyGPUArray(dldw2);

    mxGPUDestroyGPUArray(dldvout);
    mxGPUDestroyGPUArray(intensity);
    mxGPUDestroyGPUArray(forward);
    mxGPUDestroyGPUArray(codedSurf);
    mxGPUDestroyGPUArray(X_record);
    mxGPUDestroyGPUArray(diffracH1);
    mxGPUDestroyGPUArray(diffracH2);
    mxGPUDestroyGPUArray(shiftspox);
}