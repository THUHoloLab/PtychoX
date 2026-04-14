#include "addon/mex.h"
#include "addon/mxGPUArray.h"
#include "addon/addon.h"
#include <cstdint>

void mexFunction(
    int nlhs, mxArray *plhs[], 
    int nrhs, mxArray const *  __restrict__ prhs[]
){
    mxInitGPU();
    
    int large_image_size  = (int) mxGetScalar(prhs[0]);
    int small_image_size  = (int) mxGetScalar(prhs[1]);
    int batch_size  = (int) mxGetScalar(prhs[2]);

    int inembed[2];
    inembed[0] = (int) large_image_size;
    inembed[1] = (int) large_image_size;

    cufftHandle cuFFT_manyplan;
    cufftHandle cuFFT_singplan;

    cufftPlanMany(
        &cuFFT_manyplan, 2, 
        &inembed[0], &inembed[0], 1,  large_image_size * large_image_size, // idist
        &inembed[0], 1, large_image_size * large_image_size, // idist
        CUFFT_C2C, batch_size
    );

    cufftPlan2d(
        &cuFFT_singplan, 
        large_image_size, 
        large_image_size, 
        CUFFT_C2C
    );

    creal32_t * __restrict__ wavefront1_buffer;
    creal32_t * __restrict__ X_forward;
    creal32_t * __restrict__ X_record;
    uint64_t arraysize = large_image_size * large_image_size * batch_size;
    uint64_t planesize = large_image_size * large_image_size;

    cudaMalloc((creal32_t**)&wavefront1_buffer, (2 * planesize) * sizeof(float));
    cudaMalloc((creal32_t**)&X_forward, (2 * arraysize) * sizeof(float));
    cudaMalloc((creal32_t**)&X_record,  (2 * arraysize) * sizeof(float));

    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    plhs[1] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    plhs[2] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    plhs[3] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    plhs[4] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);

    *(uint64_t*)mxGetData(plhs[0]) = static_cast<uint64_t>(cuFFT_singplan);
    *(uint64_t*)mxGetData(plhs[1]) = static_cast<uint64_t>(cuFFT_manyplan);
    *(uint64_t*)mxGetData(plhs[2]) = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(wavefront1_buffer));
    *(uint64_t*)mxGetData(plhs[3]) = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(X_forward));
    *(uint64_t*)mxGetData(plhs[4]) = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(X_record));
}
