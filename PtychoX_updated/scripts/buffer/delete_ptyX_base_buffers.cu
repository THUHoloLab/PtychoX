#include "mex/mex.h"
#include "mex/mxGPUArray.h"
#include "kernels/addon/addon.h"

void mexFunction(
    int nlhs, mxArray *plhs[],
    int nrhs, mxArray const * __restrict__ prhs[]
){
    mxInitGPU();

    CHECK_THROW(nrhs == 3);

    cufftHandle cuFFT_manyplan = static_cast<cufftHandle>(*(uint64_t*)mxGetData(prhs[0]));
    creal32_t* latentW_buffer = uint64ToPtr<creal32_t>(prhs[1]);
    creal32_t* latentZ_buffer = uint64ToPtr<creal32_t>(prhs[2]);

    cufftDestroy(cuFFT_manyplan);
    cudaFree(latentW_buffer);
    cudaFree(latentZ_buffer);
}

