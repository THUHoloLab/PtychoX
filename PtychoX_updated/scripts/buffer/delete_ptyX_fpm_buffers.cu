#include "mex/mex.h"
#include "mex/mxGPUArray.h"
#include "kernels/addon/addon.h"
#include <cstdint>

void mexFunction(
    int nlhs, mxArray *plhs[],
    int nrhs, mxArray const *  __restrict__ prhs[]
){
    mxInitGPU();

    cufftHandle cuFFThandle_single;
    cufftHandle cuFFThandle_many;

    cuFFThandle_single = static_cast<cufftHandle>(*(uint64_t*)mxGetData(prhs[0]));
    cuFFThandle_many = static_cast<cufftHandle>(*(uint64_t*)mxGetData(prhs[1]));

    creal32_t* Xfwd_buffer = uint64ToPtr<creal32_t>(prhs[2]);
    creal32_t* Xrcd_buffer = uint64ToPtr<creal32_t>(prhs[3]);

    cudaFree(Xfwd_buffer);
    cudaFree(Xrcd_buffer);
    cufftDestroy(cuFFThandle_single);
    cufftDestroy(cuFFThandle_many);
}
