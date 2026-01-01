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
    // CHECK_THROW(mxIsGPUArray(prhs[4]));
    // CHECK_THROW(mxIsGPUArray(prhs[5]));

    CST_GPU_PTR wavefront2 = mxGPUCreateFromMxArray(prhs[1]); 
    // CST_GPU_PTR diffracH1  = mxGPUCreateFromMxArray(prhs[2]); 
    CST_GPU_PTR diffracH2  = mxGPUCreateFromMxArray(prhs[2]); 
    CST_GPU_PTR shiftspox  = mxGPUCreateFromMxArray(prhs[3]);

    mxGPUArray_t * wavefront1  = mxGPUCopyFromMxArray(prhs[0]); 

    int downSam_ratio  = (int) mxGetScalar(prhs[4]);

    dim3 imLs_sz = size2dim3(shiftspox);
    dim3 imHs_sz = size2dim3(wavefront1);
    dim3 imHs_bc = {imHs_sz.x,imHs_sz.y,imLs_sz.x};
    dim3 imLs_bc = {imHs_sz.x/downSam_ratio,
                    imHs_sz.y/downSam_ratio,
                    imLs_sz.x};

    const mwSize Hsz[3] = {imHs_sz.x,imHs_sz.y,imLs_sz.x};
    const mwSize Lsz[3] = {imHs_sz.x/downSam_ratio,
                           imHs_sz.y/downSam_ratio,
                           imLs_sz.x};
    // outputs of forward function
    mxGPUArray_t * X_record = mxGPUCreateGPUArray(
        3, Hsz,                   
        mxSINGLE_CLASS, mxCOMPLEX, 
        MX_GPU_DO_NOT_INITIALIZE //MX_GPU_INITIALIZE_VALUES
    );
    mxGPUArray_t * X_forward = mxGPUCreateGPUArray(
        3, Hsz,                   
        mxSINGLE_CLASS, mxCOMPLEX, 
        MX_GPU_DO_NOT_INITIALIZE 
    );
    mxGPUArray_t * observeds = mxGPUCreateGPUArray(
        3, Lsz,                   
        mxSINGLE_CLASS, mxREAL, 
        MX_GPU_DO_NOT_INITIALIZE 
    );


    const creal32_t * v_wavefront2 = getGPUDataRO<creal32_t>(wavefront2);
    // const creal32_t * v_diffracH1  = getGPUDataRO<creal32_t>(diffracH1);
    const creal32_t * v_diffracH2  = getGPUDataRO<creal32_t>(diffracH2);
    const float2 * v_shiftspox   = getGPUDataRO<float2>(shiftspox);
          
    creal32_t * v_wavefront1 = (creal32_t *) mxGPUGetData(wavefront1);
    creal32_t * v_X_record   = (creal32_t *) mxGPUGetData(X_record);
    creal32_t * v_X_forward  = (creal32_t *) mxGPUGetData(X_forward);
    float * v_observeds = (float *) mxGPUGetData(observeds);

    dim3 N_THREADS = {BLOCK_X,BLOCK_Y,1};

    dim3 N_BLOCKS = {
        (unsigned) ((imHs_sz.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imHs_sz.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) imHs_bc.z
    };

    cufftHandle plan;
    cufftHandle many_plan;
    cufftPlan2d(&plan, imHs_sz.x, imHs_sz.y, CUFFT_C2C);
    
    int inembed[2];
    inembed[0] = (int) imHs_bc.x;
    inembed[1] = (int) imHs_bc.x;

    cufftPlanMany(
        &many_plan, 2, 
        &inembed[0], &inembed[0], 1,  imHs_bc.x * imHs_bc.y, // idist
        &inembed[0], 1, imHs_bc.x * imHs_bc.y, // idist
        CUFFT_C2C, imHs_bc.z
    );
    // running executions
    cufftExecC2C(plan, 
        (cufftComplex *)&v_wavefront1[0], 
        (cufftComplex *)&v_wavefront1[0],
        CUFFT_FORWARD
    );cufftDestroy(plan);
    // cufftShift_2D_kernel<<<N_BLOCKS, N_THREADS>>>(v_wavefront1,imHs_sz.x);

    fullyfused_shiftNprop<<<N_BLOCKS, N_THREADS>>>(
        v_wavefront1,   // const creal32_t * __restrict__ wavefront1,
        // v_diffracH1,    // const creal32_t * __restrict__ prop,
        v_shiftspox,    // const float2 * __restrict__ scanning_pos,
        imHs_bc,        // const dim3 imgSzH, 
        // outputs 
        v_X_record      // creal32_t * __restrict__ x_forward
    );
    // begin ifft2d
    cufftExecC2C(many_plan, 
        (cufftComplex *)&v_X_record[0], 
        (cufftComplex *)&v_X_record[0], 
        CUFFT_INVERSE
    );
    // done!
    fused_pixelwiseProduct<<<N_BLOCKS, N_THREADS>>>(
        v_X_record, v_wavefront2,
        imHs_bc,
        v_X_forward
    );
    {// do ASP propagation
        cufftExecC2C(many_plan, 
            (cufftComplex *)&v_X_forward[0], 
            (cufftComplex *)&v_X_forward[0], 
            CUFFT_FORWARD
        );
        // done!
        fused_pixelwiseProduct_inplace<<<N_BLOCKS, N_THREADS>>>(
            v_diffracH2, // propagation
            imHs_bc,
            v_X_forward
        );
        // ifft2d_many(imHs_bc, v_X_forward);
        cufftExecC2C(many_plan, 
            (cufftComplex *)&v_X_forward[0], 
            (cufftComplex *)&v_X_forward[0], 
            CUFFT_INVERSE
        );
    };cufftDestroy(many_plan);


    fullyfused_DownSample_Fwd<<<N_BLOCKS, N_THREADS>>>(
        v_X_forward,
        (int) downSam_ratio,
        imLs_bc,
        v_observeds
    );

    // ifftCorrection_many<<<N_BLOCKS, N_THREADS>>>(v_X_forward,imHs_bc);
    plhs[0] = mxGPUCreateMxArrayOnGPU(X_record);
    plhs[1] = mxGPUCreateMxArrayOnGPU(X_forward);
    plhs[2] = mxGPUCreateMxArrayOnGPU(observeds);
    // delete output parameters
    mxGPUDestroyGPUArray(X_record);
    mxGPUDestroyGPUArray(X_forward);
    mxGPUDestroyGPUArray(observeds);
    // delete input parameters
    mxGPUDestroyGPUArray(wavefront1);
    mxGPUDestroyGPUArray(wavefront2);
    // mxGPUDestroyGPUArray(diffracH1);
    mxGPUDestroyGPUArray(diffracH2);
    mxGPUDestroyGPUArray(shiftspox);
    // mxGPUDestroyGPUArray(obseY);
}