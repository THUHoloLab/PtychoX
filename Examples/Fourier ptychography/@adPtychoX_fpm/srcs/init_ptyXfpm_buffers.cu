#include "mex.h"
#include "cuda/mxGPUArray.h"
#include "cuda/addon.h"

void mexFunction(
    int nlhs, mxArray *plhs[],
    int nrhs, mxArray const *  __restrict__ prhs[]
){
    mxInitGPU();

    int large_image_height = (int) mxGetScalar(prhs[0]);
    int large_image_width = (int) mxGetScalar(prhs[1]);
    int batch_size = (int) mxGetScalar(prhs[2]);
    int small_image_width = (int) mxGetScalar(prhs[3]);

    int inembed[2] = {small_image_width, small_image_width};

    cufftHandle cuFFThandle_single;
    cufftHandle cuFFThandle_many;

    creal32_t * __restrict__ Xfwd_buffer;
    creal32_t * __restrict__ Xrcd_buffer;

    cufftPlan2d(
        &cuFFThandle_single,
        large_image_width,
        large_image_height,
        CUFFT_C2C
    );

    cufftPlanMany(
        &cuFFThandle_many,
        2,
        &inembed[0],
        &inembed[0],
        1,
        small_image_width * small_image_width,
        &inembed[0],
        1,
        small_image_width * small_image_width,
        CUFFT_C2C,
        batch_size
    );

    uint64_t plane_size = (uint64_t) small_image_width * (uint64_t) small_image_width;
    uint64_t stack_size = plane_size * (uint64_t) batch_size;

    cudaMalloc((creal32_t**)&Xfwd_buffer, (2 * stack_size) * sizeof(float));
    cudaMalloc((creal32_t**)&Xrcd_buffer, (2 * stack_size) * sizeof(float));

    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    plhs[1] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    plhs[2] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    plhs[3] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);

    *(uint64_t*)mxGetData(plhs[0]) = static_cast<uint64_t>(cuFFThandle_single);
    *(uint64_t*)mxGetData(plhs[1]) = static_cast<uint64_t>(cuFFThandle_many);
    *(uint64_t*)mxGetData(plhs[2]) = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Xfwd_buffer));
    *(uint64_t*)mxGetData(plhs[3]) = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Xrcd_buffer));
}
