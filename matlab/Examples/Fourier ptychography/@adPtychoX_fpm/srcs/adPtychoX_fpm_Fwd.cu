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

    mxGPUArray_t * wavefront1   = mxGPUCopyFromMxArray(prhs[0]); // sample wavefront
    const mxGPUArray_t * pupil  = mxGPUCreateFromMxArray(prhs[1]); // pupil wavefront
    const mxGPUArray_t * obseY  = mxGPUCreateFromMxArray(prhs[2]); // observed intensity
    const mxGPUArray_t * ledIdx = mxGPUCreateFromMxArray(prhs[3]); // LED position in pixel

    cufftHandle plan = static_cast<cufftHandle>(*(uint64_t*)mxGetData(prhs[4]));
    cufftHandle plan_many = static_cast<cufftHandle>(*(uint64_t*)mxGetData(prhs[5]));
    creal32_t * __restrict__ v_X_record = uint64ToPtr<creal32_t>(prhs[6]);
    creal32_t * __restrict__ v_X_forward = uint64ToPtr<creal32_t>(prhs[7]);

    const creal32_t * v_pupil  = getGPUDataRO<creal32_t>(pupil);
    const int2 * v_ledIdx = getGPUDataRO<int2>(ledIdx);

    mxGPUArray_t * predY = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(obseY), 
        mxGPUGetDimensions(obseY),                   
        mxSINGLE_CLASS, mxREAL, 
        MX_GPU_DO_NOT_INITIALIZE
    );

    creal32_t * v_wavefront1 = (creal32_t * __restrict__ ) mxGPUGetData(wavefront1);
    float * v_predY = (float * __restrict__ ) mxGPUGetData(predY);

    // get image size                            
    dim3 imLs_sz = size2dim3(obseY);
    dim3 imHs_sz = size2dim3(wavefront1);
    dim3 N_THREADS = {BLOCK_X, BLOCK_Y,1};

    dim3 N_BLOCKS_S = {
        (unsigned) ((imLs_sz.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imLs_sz.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) imLs_sz.z
    };
    cufftExecC2C(
        plan, 
        (cufftComplex *)&v_wavefront1[0], 
        (cufftComplex *)&v_wavefront1[0],
        CUFFT_FORWARD
    );
    fused_getSubpupil_shift<<<N_BLOCKS_S, N_THREADS>>>(
        v_wavefront1, 
        v_pupil, v_ledIdx, 
        imLs_sz, imHs_sz, 
        v_X_forward, 
        v_X_record
    );
    cufftExecC2C(
        plan_many, 
        (cufftComplex *)&v_X_record[0], 
        (cufftComplex *)&v_X_record[0], 
        CUFFT_INVERSE
    );
    ifftCorrection_sub<<<N_BLOCKS_S, N_THREADS>>>(
        v_X_record, v_predY, 
        imLs_sz, 1.0f / (float) (imLs_sz.x * imLs_sz.y)
    );

    plhs[0] = mxGPUCreateMxArrayOnGPU(predY);

    mxGPUDestroyGPUArray(predY);
    mxGPUDestroyGPUArray(wavefront1);
    mxGPUDestroyGPUArray(pupil);
    mxGPUDestroyGPUArray(obseY);
    mxGPUDestroyGPUArray(ledIdx);
}
