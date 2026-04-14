#include "mex/mex.h"
#include "mex/mxGPUArray.h"
#include "kernels/addon/addon.h"
#include <cstdint>

#include "kernels/cplx_number.cuh"
#include "kernels/kernels_coded.cuh"
#include "kernels/cufftexes.cuh"

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
    // CHECK_THROW(mxIsGPUArray(prhs[8]));

    CST_GPU_PTR dldout = mxGPUCreateFromMxArray(prhs[0]); 
    CST_GPU_PTR observ = mxGPUCreateFromMxArray(prhs[1]); 
    CST_GPU_PTR codedSurf = mxGPUCreateFromMxArray(prhs[2]); 
    CST_GPU_PTR diffracH1 = mxGPUCreateFromMxArray(prhs[3]); 
    CST_GPU_PTR diffracH2 = mxGPUCreateFromMxArray(prhs[4]); 
    CST_GPU_PTR shiftspox = mxGPUCreateFromMxArray(prhs[5]);

    cufftHandle plan      = static_cast<cufftHandle>(*(uint64_t*)mxGetData(prhs[6]));
    cufftHandle many_plan = static_cast<cufftHandle>(*(uint64_t*)mxGetData(prhs[7]));
    creal32_t* v_X_forward = reinterpret_cast<creal32_t*>(
        static_cast<uintptr_t>(*reinterpret_cast<uint64_t*>(mxGetData(prhs[8])))
    );
    const creal32_t* v_X_record = reinterpret_cast<const creal32_t*>(
        static_cast<uintptr_t>(*reinterpret_cast<uint64_t*>(mxGetData(prhs[9])))
    );

    int downSam_ratio  = (int) mxGetScalar(prhs[10]);
    dim3 imHs_bc = size2dim3(codedSurf);
    dim3 imLs_bc = size2dim3(observ);
    imHs_bc.z = imLs_bc.z;

    const mwSize Hsz[2] = {imHs_bc.x,imHs_bc.y};

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
    const creal32_t * v_diffracH1 = getGPUDataRO<creal32_t>(diffracH1);
    const creal32_t * v_diffracH2 = getGPUDataRO<creal32_t>(diffracH2);

    const float2 * v_shiftspox = getGPUDataRO<float2>(shiftspox);
    const float * v_dldout = getGPUDataRO<float>(dldout);
    const float * v_observ = getGPUDataRO<float>(observ);

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
        (int) downSam_ratio,
        imLs_bc,
        v_X_forward
    );

    {// do ASM propagation
        cufftExecC2C(many_plan, 
            (cufftComplex *)&v_X_forward[0], 
            (cufftComplex *)&v_X_forward[0], 
            CUFFT_FORWARD
        );
        // done!
        fused_pixelwiseProduct_inplace_conj<<<N_BLOCKS, N_THREADS>>>(
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
    };
    N_BLOCKS.z = 1;
    deconvCodedSurf<<<N_BLOCKS, N_THREADS>>>(
        v_codedSurf,
        v_X_record,
        imHs_bc,
        // output
        v_X_forward,
        v_dldw2
    );
    cufftExecC2C(many_plan, 
        (cufftComplex *)&v_X_forward[0], 
        (cufftComplex *)&v_X_forward[0], 
        CUFFT_FORWARD
    );
    fullyfused_ReducedSum<<<N_BLOCKS, N_THREADS>>>(
        v_X_forward,
        v_diffracH1,
        v_shiftspox,
        imHs_bc,
        v_dldw1
    );

    cufftExecC2C(plan, 
        (cufftComplex *)&v_dldw1[0], 
        (cufftComplex *)&v_dldw1[0],
        CUFFT_FORWARD
    );
    
    plhs[0] = mxGPUCreateMxArrayOnGPU(dldw1);
    plhs[1] = mxGPUCreateMxArrayOnGPU(dldw2);

    mxGPUDestroyGPUArray(dldw1);
    mxGPUDestroyGPUArray(dldw2);

    mxGPUDestroyGPUArray(dldout);
    mxGPUDestroyGPUArray(observ);
    mxGPUDestroyGPUArray(codedSurf);
    mxGPUDestroyGPUArray(diffracH1);
    mxGPUDestroyGPUArray(diffracH2);
    mxGPUDestroyGPUArray(shiftspox);
}

