#include "addon/mex.h"
#include "addon/mxGPUArray.h"
#include "addon/addon.h"
#include <cstdint>

void mexFunction(
    int nlhs, mxArray *plhs[], 
    int nrhs, mxArray const *  __restrict__ prhs[]
){
    mxInitGPU();
    
    cufftHandle cuFFT_manyplan;
    cufftHandle cuFFT_singplan;

    cuFFT_singplan = static_cast<cufftHandle>(*(uint64_t*)mxGetData(prhs[0]));
    cuFFT_manyplan = static_cast<cufftHandle>(*(uint64_t*)mxGetData(prhs[1]));

    creal32_t* wavefront1_buffer = reinterpret_cast<creal32_t*>(
        static_cast<uintptr_t>(*reinterpret_cast<uint64_t*>(mxGetData(prhs[2])))
    );
    creal32_t* X_forward = reinterpret_cast<creal32_t*>(
        static_cast<uintptr_t>(*reinterpret_cast<uint64_t*>(mxGetData(prhs[3])))
    );
    creal32_t* X_record = reinterpret_cast<creal32_t*>(
        static_cast<uintptr_t>(*reinterpret_cast<uint64_t*>(mxGetData(prhs[4])))
    );

    cufftDestroy(cuFFT_singplan);
    cufftDestroy(cuFFT_manyplan);
    cudaFree(wavefront1_buffer);
    cudaFree(X_forward);
    cudaFree(X_record);
}
