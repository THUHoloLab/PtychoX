#include "mex/mex.h"
#include "mex/mxGPUArray.h"
#include "kernels/addon/addon.h"
#include <cstdint>

void mexFunction(
    int nlhs, mxArray *plhs[],
    int nrhs, mxArray const * __restrict__ prhs[]
){
    mxInitGPU();

    CHECK_THROW(nrhs == 5);

    int sample_height = (int) mxGetScalar(prhs[0]);
    int sample_width = (int) mxGetScalar(prhs[1]);
    int probe_height = (int) mxGetScalar(prhs[2]);
    int probe_width = (int) mxGetScalar(prhs[3]);
    int batch_size = (int) mxGetScalar(prhs[4]);

    CHECK_THROW(probe_height > 0);
    CHECK_THROW(probe_width > 0);
    CHECK_THROW(batch_size > 0);
    CHECK_THROW(sample_height >= probe_height);
    CHECK_THROW(sample_width >= probe_width);

    int inembed[2] = {probe_width, probe_width};
    cufftHandle cuFFT_manyplan;

    cufftPlanMany(
        &cuFFT_manyplan, 2,
        &inembed[0], &inembed[0], 1, probe_width * probe_height,
        &inembed[0], 1, probe_width * probe_height,
        CUFFT_C2C, batch_size
    );

    creal32_t * latentW_buffer;
    creal32_t * latentZ_buffer;

    uint64_t stack_size = (uint64_t) probe_height * (uint64_t) probe_width * (uint64_t) batch_size;

    cudaMalloc((creal32_t**)&latentW_buffer, stack_size * sizeof(creal32_t));
    cudaMalloc((creal32_t**)&latentZ_buffer, stack_size * sizeof(creal32_t));

    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    plhs[1] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    plhs[2] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);

    *(uint64_t*)mxGetData(plhs[0]) = static_cast<uint64_t>(cuFFT_manyplan);
    *(uint64_t*)mxGetData(plhs[1]) = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(latentW_buffer));
    *(uint64_t*)mxGetData(plhs[2]) = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(latentZ_buffer));
}

