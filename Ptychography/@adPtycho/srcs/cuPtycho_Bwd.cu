#include "addon/mex.h"
#include "addon/mxGPUArray.h"
#include "addon/addon.h"

#include "cuda/kernels.cuh"
#include "cuda/fftshift_kernel.cuh"

void mexFunction(
    int nlhs, mxArray *plhs[], 
    int nrhs, mxArray const *  __restrict__ prhs[]
){
    // input parames
    mxInitGPU();

    CHECK_THROW(mxIsGPUArray(prhs[0])); // dldout
    CHECK_THROW(mxIsGPUArray(prhs[1])); // wavefront1
    CHECK_THROW(mxIsGPUArray(prhs[2])); // probe
    CHECK_THROW(mxIsGPUArray(prhs[3])); // position

    CHECK_THROW(mxIsGPUArray(prhs[4])); // latentW
    CHECK_THROW(mxIsGPUArray(prhs[5])); // latentZ

    const mxGPUArray_t * dldout   = mxGPUCreateFromMxArray(prhs[0]);
    const mxGPUArray_t * sample   = mxGPUCreateFromMxArray(prhs[1]);
    const mxGPUArray_t * probe    = mxGPUCreateFromMxArray(prhs[2]);
    const mxGPUArray_t * position = mxGPUCreateFromMxArray(prhs[3]);
    const mxGPUArray_t * latentW  = mxGPUCreateFromMxArray(prhs[5]);

    mxGPUArray_t * latentZ = mxGPUCopyFromMxArray(prhs[4]);

    dim3 imLs_bc = size2dim3(dldout);   
    dim3 imHs_sz = size2dim3(sample);   

    mxGPUArray_t * dldw1 = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(sample), 
        mxGPUGetDimensions(sample),                   
        mxSINGLE_CLASS, mxCOMPLEX, 
        MX_GPU_INITIALIZE_VALUES //MX_GPU_INITIALIZE_VALUES
    );
    mxGPUArray_t * dldw2 = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(probe), 
        mxGPUGetDimensions(probe),                   
        mxSINGLE_CLASS, mxCOMPLEX, 
        MX_GPU_INITIALIZE_VALUES //MX_GPU_INITIALIZE_VALUES
    );

    dim3 N_THREADS = {BLOCK_X,BLOCK_Y,1};
    dim3 N_BLOCKS = {
        (unsigned) ((imLs_bc.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imLs_bc.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) imLs_bc.z
    };

    const float2 * v_latentW = getGPUDataRO<float2>(latentW);
    const float2 * v_probe   = getGPUDataRO<float2>(probe);
    const int2 * v_position = getGPUDataRO<int2>(position);
    const float * v_dldout  = getGPUDataRO<float>(dldout);
   
    float2 * v_dldw1   = (float2 * ) mxGPUGetData(dldw1);
    float2 * v_dldw2   = (float2 * ) mxGPUGetData(dldw2);
    float2 * v_latentZ = (float2 * ) mxGPUGetData(latentZ);
    
    fullyfused_ConsShift<<<N_BLOCKS, N_THREADS>>>(
        v_dldout,
        imLs_bc,
        v_latentZ
    );
    // setConstraints<<<N_BLOCKS, N_THREADS>>>(
    //     v_dldout,
    //     imLs_bc,
    //     v_latentZ
    // );

    // cufftShift_2D_kernel<<<N_BLOCKS, N_THREADS>>>(
    //     v_latentZ,
    //     (int) imLs_bc.x
    // );

    // cufftHandle many_plan;
    int inembed[2];
    inembed[0] = (int) imLs_bc.x;
    inembed[1] = (int) imLs_bc.x;
    // cufftPlanMany(&many_plan, 2, 
    //     &inembed[0], &inembed[0], 1,  imLs_bc.x * imLs_bc.y, // idist
    //     &inembed[0], 1, imLs_bc.x * imLs_bc.y, // idist
    //     CUFFT_C2C, imLs_bc.z
    // );
    // uint64_t handle = *static_cast<uint64_t*>(mxGetData(prhs[6]));
    // cufftHandle many_plan = reinterpret_cast<cufftHandle>(handle);
    cufftHandle many_plan = static_cast<cufftHandle>(*(uint64_t*)mxGetData(prhs[6]));
    cufftExecC2C(many_plan, (cufftComplex *)&v_latentZ[0], (cufftComplex *)&v_latentZ[0], 
        CUFFT_INVERSE
    );
    cufftDestroy(many_plan);

    reducedSum<<<N_BLOCKS, N_THREADS>>>(
        v_dldw1, v_dldw2,
        v_probe, v_latentZ, v_latentW, v_position,
        imLs_bc, imHs_sz
    );

    plhs[0] = mxGPUCreateMxArrayOnGPU(dldw1);
    plhs[1] = mxGPUCreateMxArrayOnGPU(dldw2);


    mxGPUDestroyGPUArray(latentZ);
    mxGPUDestroyGPUArray(dldw1);
    mxGPUDestroyGPUArray(dldw2);

    mxGPUDestroyGPUArray(dldout);
    mxGPUDestroyGPUArray(probe);
    mxGPUDestroyGPUArray(sample);
    mxGPUDestroyGPUArray(position);
    mxGPUDestroyGPUArray(latentW);
}