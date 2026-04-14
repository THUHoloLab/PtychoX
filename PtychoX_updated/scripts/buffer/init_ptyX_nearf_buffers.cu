#include "mex/mex.h"
#include "mex/mxGPUArray.h"
#include "kernels/addon/addon.h"
#include <cstdint>

void mexFunction(
    int nlhs, mxArray *plhs[],
    int nrhs, mxArray const *  __restrict__ prhs[]
){
    mxInitGPU();
    
    int large_image_height = (int) mxGetScalar(prhs[0]);
    int large_image_width = (int) mxGetScalar(prhs[1]);
    int batch_size = (int) mxGetScalar(prhs[2]);

    int inembed[2] = {large_image_width, large_image_width};

    cufftHandle cuFFT_singplan;
    cufftHandle cuFFT_manyplan;

    cufftPlan2d(
        &cuFFT_singplan, 
        large_image_width, 
        large_image_height, 
        CUFFT_C2C
    );
    
    cufftPlanMany(
        &cuFFT_manyplan, 2,
        &inembed[0], &inembed[0], 
        1, large_image_width * large_image_height,
        &inembed[0], 1, large_image_width * large_image_height,
        CUFFT_C2C, batch_size
    );

    creal32_t * __restrict__ wave_buffer;
    creal32_t * __restrict__ Xfwd_buffer;
    creal32_t * __restrict__ Xrcd_buffer;

    uint64_t plane_size = (uint64_t) large_image_height * (uint64_t) large_image_width;
    uint64_t stack_size = plane_size * (uint64_t) batch_size;

    cudaMalloc((creal32_t**)&wave_buffer, (2 * plane_size) * sizeof(float));
    cudaMalloc((creal32_t**)&Xfwd_buffer, (2 * stack_size) * sizeof(float));
    cudaMalloc((creal32_t**)&Xrcd_buffer, (2 * stack_size) * sizeof(float));

    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    plhs[1] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    plhs[2] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    plhs[3] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    plhs[4] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);

    *(uint64_t*)mxGetData(plhs[0]) = static_cast<uint64_t>(cuFFT_singplan);
    *(uint64_t*)mxGetData(plhs[1]) = static_cast<uint64_t>(cuFFT_manyplan);
    *(uint64_t*)mxGetData(plhs[2]) = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(wave_buffer));
    *(uint64_t*)mxGetData(plhs[3]) = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Xfwd_buffer));
    *(uint64_t*)mxGetData(plhs[4]) = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Xrcd_buffer));
}

