#include "addon/mex.h"
#include "addon/mxGPUArray.h"
#include "addon/addon.h"

void mexFunction(
    int nlhs, mxArray *plhs[],
    int nrhs, mxArray const *  __restrict__ prhs[]
){
    mxInitGPU();
    
    cufftHandle cuFFT_singplan;
    cufftHandle cuFFT_manyplan;

    cuFFT_singplan = static_cast<cufftHandle>(*(uint64_t*)mxGetData(prhs[0]));
    cuFFT_manyplan = static_cast<cufftHandle>(*(uint64_t*)mxGetData(prhs[1]));

    creal32_t* wave_buffer = uint64ToPtr<creal32_t>(prhs[2]);
    creal32_t* Xfwd_buffer = uint64ToPtr<creal32_t>(prhs[3]);
    creal32_t* Xrcd_buffer = uint64ToPtr<creal32_t>(prhs[4]);

    cufftDestroy(cuFFT_singplan);
    cufftDestroy(cuFFT_manyplan);
    cudaFree(wave_buffer);
    cudaFree(Xfwd_buffer);
    cudaFree(Xrcd_buffer);
}
