#include "mex.h"
#include "cuda/mxGPUArray.h"
#include "cuda/kernels.cuh"
#include <cuda_runtime.h>

dim3 size2dim3( const mxGPUArray * in){
    const mwSize *sz = mxGPUGetDimensions(in);
    const int  dim = (int) mxGPUGetNumberOfDimensions(in);
    dim3 imgSz;
    imgSz = {(unsigned) sz[1], (unsigned) sz[0], 1};
    if (dim > 2){
        imgSz.z = (unsigned) sz[2];
    }
    return imgSz;
}

void mexFunction(
    int nlhs, mxArray *plhs[], 
    int nrhs, mxArray const *  __restrict__ prhs[]
){
    mxInitGPU();

    const mxGPUArray_t * dl_doutput = mxGPUCreateFromMxArray(prhs[0]); // sample wavefront
    const mxGPUArray_t * wavefront1 = mxGPUCreateFromMxArray(prhs[1]); // sample wavefront
    const mxGPUArray_t * wavefront2 = mxGPUCreateFromMxArray(prhs[2]); // pupil wavefront
    const mxGPUArray_t * ledIdx     = mxGPUCreateFromMxArray(prhs[3]); // LED position in pixel
    const mxGPUArray_t * obseY      = mxGPUCreateFromMxArray(prhs[8]); 

    const real32_t * d_dl_doutput  = getGPUDataRO<real32_t>(dl_doutput);
    const creal32_t * d_wavefront2 = getGPUDataRO<creal32_t>(wavefront2);
    const int2 * d_ledIdx      = getGPUDataRO<int2>(ledIdx);

    cufftHandle plan = static_cast<cufftHandle>(*(uint64_t*)mxGetData(prhs[4]));
    cufftHandle plan_many = static_cast<cufftHandle>(*(uint64_t*)mxGetData(prhs[5]));
    creal32_t * __restrict__ v_X_record = uint64ToPtr<creal32_t>(prhs[6]);
    creal32_t * __restrict__ v_X_forward = uint64ToPtr<creal32_t>(prhs[7]);
    
    mxGPUArray_t * dldw1 = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(wavefront1), 
        mxGPUGetDimensions(wavefront1),
        mxSINGLE_CLASS, mxCOMPLEX, 
        MX_GPU_INITIALIZE_VALUES
    );

    mxGPUArray_t * dldw2 = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(wavefront2), 
        mxGPUGetDimensions(wavefront2),
        mxSINGLE_CLASS, mxCOMPLEX, 
        MX_GPU_INITIALIZE_VALUES
    );

    creal32_t * __restrict__ d_dldw1 = (creal32_t * __restrict__ ) mxGPUGetData(dldw1);
    creal32_t * __restrict__ d_dldw2 = (creal32_t * __restrict__ ) mxGPUGetData(dldw2);

    // get image size                            
    dim3 imLs_sz = size2dim3(obseY);
    dim3 imHs_sz = size2dim3(wavefront1);

    dim3 N_THREADS = {BLOCK_X,BLOCK_Y,1};
    dim3 N_BLOCKS = {
        (unsigned) (imLs_sz.x + BLOCK_X - 1) / BLOCK_X,
        (unsigned) (imLs_sz.y + BLOCK_Y - 1) / BLOCK_Y,
        (unsigned) imLs_sz.z
    };

    backward_Z<<<N_BLOCKS, N_THREADS>>>(
        imLs_sz, 
        d_dl_doutput, 
        v_X_record
    );
    cufftExecC2C(
        plan_many, 
        (cufftComplex *)v_X_record, 
        (cufftComplex *)v_X_record, 
        CUFFT_FORWARD
    );
    fused_deconvPIE_shifted<<<N_BLOCKS, N_THREADS>>>(
        v_X_record,
        d_wavefront2,
        v_X_forward,
        d_ledIdx,
        imLs_sz, imHs_sz, 1.0f / (float) (imHs_sz.x * imHs_sz.y),
        d_dldw1, d_dldw2
    ); 
    cufftExecC2C(
        plan, 
        (cufftComplex *)d_dldw1, 
        (cufftComplex *)d_dldw1,
        CUFFT_FORWARD
    );
    plhs[0] = mxGPUCreateMxArrayOnGPU(dldw1);
    plhs[1] = mxGPUCreateMxArrayOnGPU(dldw2);

    mxGPUDestroyGPUArray(dldw1);
    mxGPUDestroyGPUArray(dldw2);

    mxGPUDestroyGPUArray(dl_doutput);
    mxGPUDestroyGPUArray(wavefront1);
    mxGPUDestroyGPUArray(wavefront2);
    mxGPUDestroyGPUArray(obseY);
    mxGPUDestroyGPUArray(ledIdx);
}
