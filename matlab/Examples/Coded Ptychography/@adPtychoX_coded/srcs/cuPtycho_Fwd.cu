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
    mxInitGPU();

    CHECK_THROW(mxIsGPUArray(prhs[0]));
    CHECK_THROW(mxIsGPUArray(prhs[1]));
    CHECK_THROW(mxIsGPUArray(prhs[2]));
    CHECK_THROW(mxIsGPUArray(prhs[3]));
    CHECK_THROW(mxIsGPUArray(prhs[4]));
    // CHECK_THROW(mxIsGPUArray(prhs[5]));

    CST_GPU_PTR wavefront2 = mxGPUCreateFromMxArray(prhs[1]); 
    CST_GPU_PTR diffracH1  = mxGPUCreateFromMxArray(prhs[2]); 
    CST_GPU_PTR diffracH2  = mxGPUCreateFromMxArray(prhs[3]); 
    CST_GPU_PTR shiftspox  = mxGPUCreateFromMxArray(prhs[4]);
    // CST_GPU_PTR obseY      = mxGPUCreateFromMxArray(prhs[5]);

    int pratio = (int) mxGetPr(prhs[5])[0];

    mxGPUArray_t * wavefront1  = mxGPUCopyFromMxArray(prhs[0]); 

    dim3 imLs_sz = size2dim3(shiftspox);
    dim3 imHs_sz = size2dim3(wavefront1);
    dim3 imHs_bc = {imHs_sz.x,imHs_sz.y,imLs_sz.x};

    const mwSize Hsz[3] = {imHs_bc.x,imHs_bc.y,imHs_bc.z};
    const mwSize Lsz[3] = {imHs_bc.x/pratio,imHs_bc.y/pratio,imHs_bc.z};
    // outputs of forward function
    mxGPUArray_t * latentX = mxGPUCreateGPUArray(
        3, Hsz, mxSINGLE_CLASS, mxCOMPLEX, 
        MX_GPU_DO_NOT_INITIALIZE //MX_GPU_INITIALIZE_VALUES
    );
    mxGPUArray_t * latentY = mxGPUCreateGPUArray(
        3, Hsz, mxSINGLE_CLASS, mxCOMPLEX, 
        MX_GPU_DO_NOT_INITIALIZE 
    );
    mxGPUArray_t * intensity = mxGPUCreateGPUArray(
        3, Lsz, mxSINGLE_CLASS, mxREAL, 
        MX_GPU_DO_NOT_INITIALIZE 
    );
    imLs_sz = {(unsigned) Lsz[0],(unsigned) Lsz[1], (unsigned) Lsz[2]};

    const creal32_t * v_wavefront2 = getGPUDataRO<creal32_t>(wavefront2);
    const creal32_t * v_diffracH1  = getGPUDataRO<creal32_t>(diffracH1);
    const creal32_t * v_diffracH2  = getGPUDataRO<creal32_t>(diffracH2);
    const int2 * v_shiftspox   = getGPUDataRO<int2>(shiftspox);
          
    creal32_t * v_wavefront1 = (creal32_t *) mxGPUGetData(wavefront1);
    creal32_t * v_latentX    = (creal32_t *) mxGPUGetData(latentX);
    creal32_t * v_latentY    = (creal32_t *) mxGPUGetData(latentY);
    float * v_intensity      = (float *) mxGPUGetData(intensity);

    dim3 N_THREADS = {BLOCK_X,BLOCK_Y,1};
    dim3 N_BLOCKS = {
        (unsigned) ((imHs_bc.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imHs_bc.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) imHs_bc.z
    };

    cufftHandle many_plan;
    int inembed[2];
    inembed[0] = (int) imHs_bc.x;
    inembed[1] = (int) imHs_bc.x;

    cufftPlanMany(
        &many_plan, 2, 
        &inembed[0], &inembed[0], 1,  imHs_bc.x * imHs_bc.y, // idist
        &inembed[0], 1, imHs_bc.x * imHs_bc.y, // idist
        CUFFT_C2C, imHs_bc.z
    );

    fullyfused_ShiftProduct_Fwd<<<N_BLOCKS, N_THREADS>>>(
        v_wavefront1, v_wavefront2,
        v_shiftspox,
        imHs_bc,
        // output
        v_latentX, v_latentY
    );

    {// do ASP propagation
        cufftExecC2C(many_plan, 
            (cufftComplex *)&v_latentY[0], 
            (cufftComplex *)&v_latentY[0], 
            CUFFT_FORWARD
        );
        // done!
        fused_pixelwiseProduct_inplace<<<N_BLOCKS, N_THREADS>>>(
            v_diffracH2, // propagation
            imHs_bc,
            v_latentY
        );
        // ifft2d_many(imHs_bc, v_latentY);
        cufftExecC2C(many_plan, 
            (cufftComplex *)&v_latentY[0], 
            (cufftComplex *)&v_latentY[0], 
            CUFFT_INVERSE
        );
    };cufftDestroy(many_plan);

    N_BLOCKS = {
        (unsigned) ((imLs_sz.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imLs_sz.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) imLs_sz.z
    };

    fullyfused_DownSample_Fwd<<<N_BLOCKS, N_THREADS>>>(
        v_latentY,  // const creal32_t * __restrict__ input,
        pratio,     // const int ds,
        imLs_sz,    // const dim3 imgSz,
        v_intensity // float * __restrict__ output
    );

    // ifftCorrection_many<<<N_BLOCKS, N_THREADS>>>(v_latentY,imHs_bc);
    plhs[0] = mxGPUCreateMxArrayOnGPU(latentX);
    plhs[1] = mxGPUCreateMxArrayOnGPU(latentY);
    plhs[2] = mxGPUCreateMxArrayOnGPU(intensity);
    // delete output parameters
    mxGPUDestroyGPUArray(latentX);
    mxGPUDestroyGPUArray(latentY);
    mxGPUDestroyGPUArray(intensity);
    // delete input parameters
    mxGPUDestroyGPUArray(wavefront1);
    mxGPUDestroyGPUArray(wavefront2);
    mxGPUDestroyGPUArray(diffracH1);
    mxGPUDestroyGPUArray(diffracH2);
    mxGPUDestroyGPUArray(shiftspox);
    // mxGPUDestroyGPUArray(obseY);
}