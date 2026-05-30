#include "addon/mex.h"
#include "addon/mxGPUArray.h"
#include "addon/addon.h"

#include "cuda/kernels.cuh"
#include "cuda/fftshift_kernel.cuh"

void mexFunction(
    int nlhs, mxArray *plhs[], 
    int nrhs, mxArray const *  __restrict__ prhs[]
){
    int large_image_size  = (int) mxGetScalar(prhs[0]);
    int small_image_size  = (int) mxGetScalar(prhs[1]);

    int batch_size  = (int) mxGetScalar(prhs[2]);

    int inembed[2];
    inembed[0] = (int) large_image_size;
    inembed[1] = (int) large_image_size;

    cufftHandle many_plan;
    cufftHandle plan;

    cufftPlanMany(
        &many_plan, 2, 
        &inembed[0], &inembed[0], 1,  large_image_size * large_image_size, // idist
        &inembed[0], 1, large_image_size * large_image_size, // idist
        CUFFT_C2C, batch_size
    );
    cufftPlan2d(&plan, large_image_size, large_image_size, CUFFT_C2C);

    
}