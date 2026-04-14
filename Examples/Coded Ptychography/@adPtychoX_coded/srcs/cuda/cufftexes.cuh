#include "fftshift_kernel.cuh"

// forward fft2d with fftshift
void fft2d(
    const dim3 imgSz,
    creal32_t* __restrict__ x
){
    dim3 N_THREADS(BLOCK_X,BLOCK_Y,1);
    dim3 N_BLOCKS = {
        (unsigned) ((imgSz.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imgSz.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) 1
    };
    cufftHandle plan;
    cufftPlan2d(&plan, imgSz.x, imgSz.y, CUFFT_C2C);
    cufftExecC2C(plan, (cufftComplex *)&x[0], (cufftComplex *)&x[0],CUFFT_FORWARD);
    cufftDestroy(plan);
    cufftShift_2D_kernel<<<N_BLOCKS, N_THREADS>>>(x,imgSz.x);
}

// forward ifft2d with fftshift
void ifft2d(
    const dim3 imgSz,
    creal32_t* __restrict__ y
){
    dim3 N_THREADS(BLOCK_X,BLOCK_Y,1);
    dim3 N_BLOCKS = {
        (unsigned) ((imgSz.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imgSz.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) 1
    };
    cufftShift_2D_kernel<<<N_BLOCKS, N_THREADS>>>(y,(int) imgSz.x);

    cufftHandle plan;
    cufftPlan2d(&plan, imgSz.x, imgSz.y, CUFFT_C2C);
    cufftExecC2C(plan, (cufftComplex *)&y[0], (cufftComplex *)&y[0],CUFFT_INVERSE);
    cufftDestroy(plan);

    ifftCorrection<<<N_BLOCKS, N_THREADS>>>(y,imgSz);
}


void fft2d_many(
    const dim3 imLs_sz,
    creal32_t* __restrict__ output
){
    int inembed[2];
    for (int i{0}; i < 2; i++) {
        inembed[i] = (int) imLs_sz.x;
    }

    dim3 N_BLOCKS = {
        (unsigned) ((imLs_sz.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imLs_sz.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) imLs_sz.z
    };
    dim3 N_THREADS(BLOCK_X, BLOCK_Y, 1);

    cufftHandle plan;
    cufftPlanMany(
        &plan, 
        2, 
        &inembed[0], // n
        &inembed[0], // inembed
        1,           // istride
        (int) imLs_sz.x * (int) imLs_sz.y, // idist
        &inembed[0], // inembed
        1,           // istride
        (int) imLs_sz.x * (int) imLs_sz.y, // idist
        CUFFT_C2C, 
        (int) imLs_sz.z
    );
    cufftExecC2C(plan, (cufftComplex *)&output[0], (cufftComplex *)&output[0], CUFFT_FORWARD);
    cufftDestroy(plan);

    cufftShift_2D_kernel<<<N_BLOCKS, N_THREADS>>>(output, (unsigned) imLs_sz.x);
}

void ifft2d_many(
    const dim3 imLs_sz,
    creal32_t* __restrict__ output
){
    
    dim3 N_THREADS(BLOCK_X,BLOCK_Y,1);

    dim3 N_BLOCKS = {
        (unsigned) ((imLs_sz.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imLs_sz.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) imLs_sz.z
    };

    int inembed[2];
    for (int i{0}; i < 2; i++) {
        inembed[i] = (int) imLs_sz.x;
    }

    cufftShift_2D_kernel<<<N_BLOCKS, N_THREADS>>>(output, (unsigned) imLs_sz.x);

    cufftHandle plan;
    cufftPlanMany(
        &plan, 
        2, 
        &inembed[0], // n
        &inembed[0], // inembed
        1,           // istride
        (int) imLs_sz.x * (int) imLs_sz.y, // idist
        &inembed[0], // inembed
        1,           // istride
        (int) imLs_sz.x * (int) imLs_sz.y, // idist
        CUFFT_C2C, 
        (int) imLs_sz.z
    );
    cufftExecC2C(plan, (cufftComplex *)&output[0], (cufftComplex *)&output[0], CUFFT_INVERSE);
    cufftDestroy(plan);

    ifftCorrection_many<<<N_BLOCKS, N_THREADS>>>(output,imLs_sz);
}
