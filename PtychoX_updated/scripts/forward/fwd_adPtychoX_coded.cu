#include "mex/mex.h"
#include "mex/mxGPUArray.h"
#include "kernels/addon/addon.h"
#include <cstdint>

#include "kernels/cplx_number.cuh"
#include "kernels/kernels_coded.cuh"
#include "kernels/cufftexes.cuh"

// __global__ void mybreaks(){
//     const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
//     const unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;
//     const unsigned idz = blockIdx.z;
// }

void mexFunction(
    int nlhs, mxArray *plhs[], 
    int nrhs, mxArray const *  __restrict__ prhs[]
){
    mxInitGPU();

    CHECK_THROW(mxIsGPUArray(prhs[0]));
    CHECK_THROW(mxIsGPUArray(prhs[1]));
    CHECK_THROW(mxIsGPUArray(prhs[2]));
    CHECK_THROW(mxIsGPUArray(prhs[3]));
    CHECK_THROW(mxIsGPUArray(prhs[4]));
    // CHECK_THROW(mxIsGPUArray(prhs[5]));
    CST_GPU_PTR wavefront1_in = mxGPUCreateFromMxArray(prhs[0]); 
    CST_GPU_PTR wavefront2 = mxGPUCreateFromMxArray(prhs[1]); 
    CST_GPU_PTR diffracH1  = mxGPUCreateFromMxArray(prhs[2]); 
    CST_GPU_PTR diffracH2  = mxGPUCreateFromMxArray(prhs[3]); 
    CST_GPU_PTR shiftspox  = mxGPUCreateFromMxArray(prhs[4]);
    cufftHandle plan      = static_cast<cufftHandle>(*(uint64_t*)mxGetData(prhs[5]));
    cufftHandle many_plan = static_cast<cufftHandle>(*(uint64_t*)mxGetData(prhs[6]));
    creal32_t * v_wavefront1 = reinterpret_cast<creal32_t*>(
        static_cast<uintptr_t>(*reinterpret_cast<uint64_t*>(mxGetData(prhs[7])))
    );
    creal32_t * v_X_record = reinterpret_cast<creal32_t*>(
        static_cast<uintptr_t>(*reinterpret_cast<uint64_t*>(mxGetData(prhs[8])))
    );
    creal32_t * v_X_forward = reinterpret_cast<creal32_t*>(
        static_cast<uintptr_t>(*reinterpret_cast<uint64_t*>(mxGetData(prhs[9])))
    );

    int downSam_ratio  = (int) mxGetScalar(prhs[10]);
    dim3 imLs_sz = size2dim3(shiftspox);
    dim3 imHs_sz = size2dim3(wavefront1_in);
    dim3 imHs_bc = {imHs_sz.x,imHs_sz.y,imLs_sz.x};
    dim3 imLs_bc = {imHs_sz.x/downSam_ratio,imHs_sz.y/downSam_ratio,imLs_sz.x};
    const mwSize Lsz[3] = {imLs_bc.x,imLs_bc.y,imLs_sz.x};
    
    mxGPUArray_t * observeds = mxGPUCreateGPUArray(
        3, Lsz,                   
        mxSINGLE_CLASS, mxREAL, 
        MX_GPU_DO_NOT_INITIALIZE 
    );


    const creal32_t * v_wavefront1_in = getGPUDataRO<creal32_t>(wavefront1_in);
    const creal32_t * v_wavefront2 = getGPUDataRO<creal32_t>(wavefront2);
    const creal32_t * v_diffracH1  = getGPUDataRO<creal32_t>(diffracH1);
    const creal32_t * v_diffracH2  = getGPUDataRO<creal32_t>(diffracH2);
    const float2 * v_shiftspox   = getGPUDataRO<float2>(shiftspox);
    float * v_observeds = (float *) mxGPUGetData(observeds);

    dim3 N_THREADS = {BLOCK_X,BLOCK_Y,1};
    dim3 N_BLOCKS = {
        (unsigned) ((imHs_sz.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imHs_sz.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) imHs_bc.z
    };

    // running executions
    cudaMemcpy(
        v_wavefront1,
        v_wavefront1_in,
        static_cast<size_t>(imHs_sz.x) * static_cast<size_t>(imHs_sz.y) * sizeof(creal32_t),
        cudaMemcpyDeviceToDevice
    );
    cufftExecC2C(plan, 
        (cufftComplex *)&v_wavefront1[0], 
        (cufftComplex *)&v_wavefront1[0],
        CUFFT_FORWARD
    );
    // cufftShift_2D_kernel<<<N_BLOCKS, N_THREADS>>>(v_wavefront1,imHs_sz.x);
    fullyfused_shiftNprop<<<N_BLOCKS, N_THREADS>>>(
        v_wavefront1,   // const creal32_t * __restrict__ wavefront1,
        v_diffracH1,    // const creal32_t * __restrict__ prop,
        v_shiftspox,    // const float2 * __restrict__ scanning_pos,
        imHs_bc,        // const dim3 imgSzH, 
        // outputs 
        v_X_record      // creal32_t * __restrict__ x_forward
    );
    // begin ifft2d
    cufftExecC2C(many_plan, 
        (cufftComplex *)&v_X_record[0], 
        (cufftComplex *)&v_X_record[0], 
        CUFFT_INVERSE
    );
    // done!
    fused_pixelwiseProduct<<<N_BLOCKS, N_THREADS>>>(
        v_X_record, v_wavefront2,
        imHs_bc,
        v_X_forward
    );
    {// do ASP propagation
        cufftExecC2C(many_plan, 
            (cufftComplex *)&v_X_forward[0], 
            (cufftComplex *)&v_X_forward[0], 
            CUFFT_FORWARD
        );
        // done!
        fused_pixelwiseProduct_inplace<<<N_BLOCKS, N_THREADS>>>(
            v_diffracH2, // propagation
            imHs_bc,
            v_X_forward
        );
        // ifft2d_many(imHs_bc, v_X_forward);
        cufftExecC2C(many_plan, 
            (cufftComplex *)&v_X_forward[0], 
            (cufftComplex *)&v_X_forward[0], 
            CUFFT_INVERSE
        );
    };

    fullyfused_DownSample_Fwd<<<N_BLOCKS, N_THREADS>>>(
        v_X_forward,
        (int) downSam_ratio,
        imLs_bc,
        v_observeds
    );

    // ifftCorrection_many<<<N_BLOCKS, N_THREADS>>>(v_X_forward,imHs_bc);
    plhs[0] = mxGPUCreateMxArrayOnGPU(observeds);
    // delete output parameters
    mxGPUDestroyGPUArray(observeds);
    // delete input parameters
    mxGPUDestroyGPUArray(wavefront1_in);
    mxGPUDestroyGPUArray(wavefront2);
    mxGPUDestroyGPUArray(diffracH1);
    mxGPUDestroyGPUArray(diffracH2);
    mxGPUDestroyGPUArray(shiftspox);
    // mxGPUDestroyGPUArray(obseY);
}

