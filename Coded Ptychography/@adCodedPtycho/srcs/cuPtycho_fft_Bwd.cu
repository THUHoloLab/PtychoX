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
    // CHECK_THROW(mxIsGPUArray(prhs[8]));

    CST_GPU_PTR dldout = mxGPUCreateFromMxArray(prhs[0]); 
    CST_GPU_PTR observ = mxGPUCreateFromMxArray(prhs[1]); 

    CST_GPU_PTR codedSurf = mxGPUCreateFromMxArray(prhs[3]); 
    CST_GPU_PTR X_record  = mxGPUCreateFromMxArray(prhs[4]); 
    CST_GPU_PTR diffracH1 = mxGPUCreateFromMxArray(prhs[5]); 
    CST_GPU_PTR diffracH2 = mxGPUCreateFromMxArray(prhs[6]); 
    CST_GPU_PTR shiftspox = mxGPUCreateFromMxArray(prhs[7]);

    mxGPUArray_t * forward = mxGPUCopyFromMxArray(prhs[2]);
    dim3 imHs_bc = size2dim3(forward);
    dim3 imLs_bc = size2dim3(observ);

    mwSize Hsz[2] = {imHs_bc.x,imHs_bc.y};

    // creal32_t * __restrict__ latentZ;
    // cudaMalloc((creal32_t**)&latentZ, (2U * (imHs_bc.x * imHs_bc.y * imHs_bc.z)) * sizeof(float));

    mxGPUArray_t * dldw1 = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(diffracH1), 
        Hsz,                   
        mxSINGLE_CLASS, mxCOMPLEX, 
        MX_GPU_DO_NOT_INITIALIZE //MX_GPU_INITIALIZE_VALUES
    );

    mxGPUArray_t * dldw2 = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(diffracH1), 
        Hsz,                   
        mxSINGLE_CLASS, mxCOMPLEX, 
        MX_GPU_DO_NOT_INITIALIZE //MX_GPU_INITIALIZE_VALUES
    );

    // const creal32_t * v_dldvout   = getGPUDataRO<creal32_t>(dldvout);
    const creal32_t * v_codedSurf = getGPUDataRO<creal32_t>(codedSurf);
    const creal32_t * v_X_record  = getGPUDataRO<creal32_t>(X_record);
    const creal32_t * v_diffracH1 = getGPUDataRO<creal32_t>(diffracH1);
    const creal32_t * v_diffracH2 = getGPUDataRO<creal32_t>(diffracH2);

    const float2 * v_shiftspox = getGPUDataRO<float2>(shiftspox);
    const float * v_dldout = getGPUDataRO<float>(dldout);
    const float * v_observ = getGPUDataRO<float>(observ);

    creal32_t * v_forward = (creal32_t *) mxGPUGetData(forward);
    creal32_t * v_dldw1 = (creal32_t *) mxGPUGetData(dldw1);
    creal32_t * v_dldw2 = (creal32_t *) mxGPUGetData(dldw2);
    
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
        v_dldout,
        v_observ,
        (int) 4,
        imLs_bc,
        v_forward
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
    };

    deconvCodedSurf<<<N_BLOCKS, N_THREADS>>>(
        v_codedSurf,
        v_X_record,
        imHs_bc,
        // output
        v_forward,
        v_dldw2
    );
    // pixelwiseProduct_conj<<<N_BLOCKS, N_THREADS>>>(
    //     v_forward,// const creal32_t * __restrict__ X,
    //     v_codedSurf,// const creal32_t * __restrict__ Y,
    //     imHs_bc,// const dim3 imgSz,
    //     latentZ// creal32_t * __restrict__ Z
    // );

    cufftExecC2C(many_plan, 
        (cufftComplex *)&v_forward[0], 
        (cufftComplex *)&v_forward[0], 
        CUFFT_FORWARD
    );

    N_BLOCKS.z = 1;
    fullyfused_ReducedSum<<<N_BLOCKS, N_THREADS>>>(
        v_forward,
        v_diffracH1,
        v_shiftspox,
        imHs_bc,
        v_dldw1
    );
    cufftDestroy(many_plan);

    // fullyfused_shiftNprop_Bwd<<<N_BLOCKS, N_THREADS>>>(
    //     v_diffracH1,    // const creal32_t * __restrict__ prop,
    //     v_shiftspox,    // const float2 * __restrict__ scanning_pos,
    //     imHs_bc,        // const dim3 imgSzH, 
    //     // outputs 
    //     v_forward      // creal32_t * __restrict__ x_forward
    // );
    // // // cufftExecC2C(many_plan, 
    // // //     (cufftComplex *)&v_forward[0], 
    // // //     (cufftComplex *)&v_forward[0], 
    // // //     CUFFT_INVERSE
    // // // );
    // cudaDeviceSynchronize();
    // N_BLOCKS.z = 1;
    // reducedSum_Bwd_simple<<<N_BLOCKS, N_THREADS>>>(
    //     v_forward,
    //     imHs_bc,
    //     v_dldw1
    // );
    // // cudaDeviceSynchronize();
    cufftHandle plan;
    cufftPlan2d(&plan, imHs_bc.x, imHs_bc.y, CUFFT_C2C);
    cufftExecC2C(plan, 
        (cufftComplex *)&v_dldw1[0], 
        (cufftComplex *)&v_dldw1[0],
        CUFFT_FORWARD
    );cufftDestroy(plan);
    // reducedSum_Bwd<<<N_BLOCKS, N_THREADS>>>(
    //     latentZ,        // const creal32_t * __restrict__ forward,
    //     v_forward,      // const creal32_t * __restrict__ bwd_record,
    //     v_X_record,     // const creal32_t * __restrict__ fwd_record,
    //     imHs_bc,        // const dim3 imgSz,
    //     // outputs
    //     v_dldw1,        // const creal32_t * __restrict__ dldw1,
    //     v_dldw2         // const creal32_t * __restrict__ dldw2
    // );

    plhs[0] = mxGPUCreateMxArrayOnGPU(dldw1);
    plhs[1] = mxGPUCreateMxArrayOnGPU(dldw2);

    // cudaFree(latentZ);
    mxGPUDestroyGPUArray(dldw1);
    mxGPUDestroyGPUArray(dldw2);

    mxGPUDestroyGPUArray(dldout);
    mxGPUDestroyGPUArray(observ);
    mxGPUDestroyGPUArray(forward);
    mxGPUDestroyGPUArray(codedSurf);
    mxGPUDestroyGPUArray(X_record);
    mxGPUDestroyGPUArray(diffracH1);
    mxGPUDestroyGPUArray(diffracH2);
    mxGPUDestroyGPUArray(shiftspox);
}