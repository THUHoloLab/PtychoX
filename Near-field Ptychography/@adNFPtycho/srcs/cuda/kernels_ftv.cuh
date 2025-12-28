#include "addon.h"
#include <cooperative_groups.h>
#include <iostream>

namespace cg = cooperative_groups;

__global__ void fullyfused_shiftNprop(
    const creal32_t * __restrict__ wavefront1,
    const creal32_t * __restrict__ prop,
    const float2 * __restrict__ scanning_pos,
    const dim3 imgSzH, 
    creal32_t * __restrict__ x_forward
){
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned idz = blockIdx.z;  
    
    const bool inside = (idx < imgSzH.x) && (idy < imgSzH.y) && (idz < imgSzH.z);
    if (!inside)
        return;

    const unsigned pix_id = idx * imgSzH.y + idy;
    const unsigned page_id = idz * (imgSzH.x * imgSzH.y);
    const float2 pos = scanning_pos[idz];
    // float freq_x = ((2 * idx) < imgSzH.x)? (float) idx : (float) ((int) idx - (int) imgSzH.x);
    // float freq_y = ((2 * idy) < imgSzH.y)? (float) idy : (float) ((int) idy - (int) imgSzH.y);
    float2 freq = make_float2(
        (float) ((int) idx - (int) ((2U * idx) >= imgSzH.x) * (int) imgSzH.x),
        (float) ((int) idy - (int) ((2U * idy) >= imgSzH.y) * (int) imgSzH.y)
    ); // not fftshifted
    float fqs   = TwoPI * (freq.x * pos.x + freq.y * pos.y);
    freq = make_float2(__cosf(fqs), __sinf(fqs));

    const creal32_t this_w1 = wavefront1[pix_id];
    creal32_t this_x;

    this_x.re = this_w1.re * freq.x - this_w1.im * freq.y;
    this_x.im = this_w1.re * freq.y + this_w1.im * freq.x;

    x_forward[pix_id + page_id] = this_x;
}

__global__ void fullyfused_shiftNprop_Bwd(
    const creal32_t * __restrict__ prop,
    const float2 * __restrict__ scanning_pos,
    const dim3 imgSzH, 
    // output
    creal32_t * __restrict__ outpus
){
    auto block = cg::this_thread_block();
    const unsigned idy = block.group_index().x * block.group_dim().x + block.thread_index().x;
    const unsigned idx = block.group_index().y * block.group_dim().y + block.thread_index().y; 
    const unsigned idz = block.group_index().z; 
    
    // __shared__ creal32_t this_Y[BLOCK_SIZE];

    const bool inside = (idx < imgSzH.x) && (idy < imgSzH.y) && (idz < imgSzH.z);
    if (!inside)
        return;

    const unsigned pageSz = imgSzH.x * imgSzH.y;
    const float ratio = 1.0f / (float) pageSz;

    const unsigned pix_id = idx * imgSzH.y + idy;
    const unsigned page_id = idz * pageSz;
    const float2 pos = scanning_pos[idz];
    // float freq_x = ((2 * idx) < imgSzH.x)? (float) idx : (float) ((int) idx - (int) imgSzH.x);
    // float freq_y = ((2 * idy) < imgSzH.y)? (float) idy : (float) ((int) idy - (int) imgSzH.y);
    float2 freq = make_float2(
        (float) ((int) idx - (int) ((2U * idx) >= imgSzH.x) * (int) imgSzH.x),
        (float) ((int) idy - (int) ((2U * idy) >= imgSzH.y) * (int) imgSzH.y)
    ); // not fftshifted

    float fqs   = TwoPI * (freq.x * pos.x + freq.y * pos.y);
    float fq_re = __cosf(fqs) * ratio;
    float fq_im = __sinf(fqs) * ratio;

    const creal32_t this_out = outpus[pix_id + page_id];
    creal32_t out;

    out.re = this_out.re * fq_re + this_out.im * fq_im;
    out.im = this_out.im * fq_re - this_out.re * fq_im;

    outpus[pix_id + page_id] = out;
}

__global__ void fused_pixelwiseProduct(
    creal32_t * __restrict__ X,
    const creal32_t * __restrict__ Y,
    const dim3 imgSz,
    creal32_t * __restrict__ Z
){
    auto block = cg::this_thread_block();
    const unsigned idy = block.group_index().x * block.group_dim().x + block.thread_index().x;
    const unsigned idx = block.group_index().y * block.group_dim().y + block.thread_index().y; 
    const unsigned idz = block.group_index().z; 
    
    __shared__ creal32_t shared_Y[BLOCK_SIZE];
    const unsigned pix_id = idx * imgSz.y + idy;
    const unsigned tr = block.thread_rank();
    shared_Y[tr] = Y[pix_id];
    block.sync();

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y);
    if (inside){
        const unsigned pageSz = imgSz.x * imgSz.y;
        const unsigned page_id = idz * pageSz;
     
        // unsigned pix_id = idx * imgSz.y + idy;
        unsigned pix_bc = pix_id + page_id;

        creal32_t this_X = X[pix_bc];
        const creal32_t this_Y = shared_Y[tr];
        creal32_t this_Z;

        const float ratio = 1.0f / (float) pageSz;
        this_X.re *= ratio;
        this_X.im *= ratio;

        this_Z.re = this_X.re * this_Y.re - this_X.im * this_Y.im;
        this_Z.im = this_X.im * this_Y.re + this_X.re * this_Y.im;

        Z[pix_bc] = this_Z;
        X[pix_bc] = this_X;
    }
}

__global__ void pixelwiseProduct_conj(
    const creal32_t * __restrict__ X,
    const creal32_t * __restrict__ Y,
    const dim3 imgSz,
    creal32_t * __restrict__ Z
){
    auto block = cg::this_thread_block();
    const unsigned idy = block.group_index().x * block.group_dim().x + block.thread_index().x;
    const unsigned idx = block.group_index().y * block.group_dim().y + block.thread_index().y; 
    const unsigned idz = block.group_index().z; 
    
    __shared__ creal32_t shared_Y[BLOCK_SIZE];
    const unsigned pix_id = idx * imgSz.y + idy;
    const unsigned tr = block.thread_rank();
    shared_Y[tr] = Y[pix_id];
    block.sync();

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y);

    if (inside) {
        const unsigned page_id = idz * (imgSz.x * imgSz.y);

        const creal32_t this_X = X[pix_id + page_id];
        const creal32_t this_Y = shared_Y[tr];
        creal32_t this_Z;    

        this_Z.re = this_X.re * this_Y.re + this_X.im * this_Y.im;
        this_Z.im = this_X.im * this_Y.re - this_X.re * this_Y.im;

        Z[pix_id + page_id] = this_Z;
    }
}

__global__ void fused_pixelwiseProduct_inplace(
    const creal32_t * __restrict__ Y,
    const dim3 imgSz,
    creal32_t * __restrict__ X
){
    // const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x; 
    // const unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;
    // const unsigned idz = blockIdx.z;  
    auto block = cg::this_thread_block();
    const unsigned idy = block.group_index().x * block.group_dim().x + block.thread_index().x;
    const unsigned idx = block.group_index().y * block.group_dim().y + block.thread_index().y; 
    const unsigned idz = block.group_index().z; 
    
    __shared__ creal32_t shared_Y[BLOCK_SIZE];
    const unsigned pix_id = idx * imgSz.y + idy;
    const unsigned tr = block.thread_rank();
    shared_Y[tr] = Y[pix_id];
    block.sync();

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y);
    if (inside){
        const unsigned page_id = idz * (imgSz.x * imgSz.y);
        
        // creal32_t temp = X[pix_id + page_id];
        // temp = temp * Y[pix_id];
        const creal32_t this_X = X[pix_id + page_id];
        const creal32_t this_Y = shared_Y[tr];
        creal32_t this_O;

        const float ratio = 1.0f / (float) (imgSz.x * imgSz.y);

        this_O.re = (this_X.re * this_Y.re - this_X.im * this_Y.im) * ratio;
        this_O.im = (this_X.im * this_Y.re + this_X.re * this_Y.im) * ratio;

        X[pix_id + page_id] = this_O;
    }
}

__global__ void fused_pixelwiseProduct_inplace_conj(
    const creal32_t * __restrict__ Y,
    const dim3 imgSz,
    creal32_t * __restrict__ X
){
    auto block = cg::this_thread_block();
    const unsigned idy = block.group_index().x * block.group_dim().x + block.thread_index().x;
    const unsigned idx = block.group_index().y * block.group_dim().y + block.thread_index().y; 
    const unsigned idz = block.group_index().z; 
    
    __shared__ creal32_t shared_Y[BLOCK_SIZE];
    const unsigned pix_id = idx * imgSz.y + idy;
    const unsigned tr = block.thread_rank();
    shared_Y[tr] = Y[pix_id];
    block.sync();

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y);

    if (!inside)
        return;

    const unsigned page_id = idz * (imgSz.x * imgSz.y);
    // unsigned pix_id = idx * imgSz.y + idy;
    // creal32_t temp = X[pix_id + page_id] * Y[pix_id].conj();

    const creal32_t this_X = X[pix_id + page_id];
    const creal32_t this_Y = shared_Y[tr];
    creal32_t this_O;

    const float ratio = 1.0f / (float) (imgSz.x * imgSz.y);

    this_O.re = (this_X.re * this_Y.re + this_X.im * this_Y.im) * ratio;
    this_O.im = (this_X.im * this_Y.re - this_X.re * this_Y.im) * ratio;

    X[pix_id + page_id] = this_O;
}

__global__ void ifftCorrection(
    creal32_t* __restrict__ spectrum,
    const dim3 imHs_sz
){
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x; 
    unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idz = blockIdx.z;  

    const bool inside = (idx < imHs_sz.x) && (idy < imHs_sz.y);    
    if (inside){
        unsigned pix_id = idx * imHs_sz.y + idy;

        float ratio = 1 / ((float) (imHs_sz.x * imHs_sz.y));
        float a = spectrum[pix_id].re;
        float b = spectrum[pix_id].im;
        spectrum[pix_id].re = a * ratio;
        spectrum[pix_id].im = b * ratio;
    }
}

__global__ void ifftCorrection_many(
    creal32_t * __restrict__ spectrum, 
    const dim3 imgSz
){
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x; 
    unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idz = blockIdx.z;  

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y) && (idz < imgSz.z);
    if(inside){
        unsigned page_id = idz * (imgSz.x * imgSz.y);
        unsigned pix_id = idx * imgSz.y + idy;

        creal32_t temp = spectrum[pix_id + page_id];
        float ratio = 1.0f / (float) (imgSz.x * imgSz.y);
        temp.re *= ratio;
        temp.im *= ratio;
        spectrum[pix_id + page_id] = temp;
    }
}

__global__ void reducedSum_Bwd(
    const creal32_t * __restrict__ forward,
    const creal32_t * __restrict__ bwd_record,
    const creal32_t * __restrict__ fwd_record,
    const dim3 imgSz,
    // output
    creal32_t * __restrict__ dldw1,
    creal32_t * __restrict__ dldw2
){
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y);

    if (!inside)
        return;

    creal32_t reduced_dldw1 = {0.0f, 0.0f};
    creal32_t reduced_dldw2 = {0.0f, 0.0f};
    const unsigned pix_id = idx * imgSz.y + idy;

    #pragma unroll
    for(unsigned idz = 0; idz < imgSz.z; ++idz){
        const unsigned page_id = idz * (imgSz.x * imgSz.y);
        const unsigned pix_bc = pix_id + page_id;

        const creal32_t this_fwd = forward[pix_bc];
        const creal32_t this_bwdrec = bwd_record[pix_bc];
        const creal32_t this_fwdrec = fwd_record[pix_bc];

        reduced_dldw1.re += this_fwd.re;
        reduced_dldw1.im -= this_fwd.im;

        reduced_dldw2.re += this_bwdrec.re * this_fwdrec.re + this_bwdrec.im * this_fwdrec.im;
        reduced_dldw2.im += this_bwdrec.re * this_fwdrec.im - this_bwdrec.im * this_fwdrec.re;

        // reduced_dldw2 = reduced_dldw2 + bwd_record[pix_bc] * fwd_record[pix_bc].conj();
    }

    dldw1[pix_id] = reduced_dldw1;
    dldw2[pix_id] = reduced_dldw2;
}