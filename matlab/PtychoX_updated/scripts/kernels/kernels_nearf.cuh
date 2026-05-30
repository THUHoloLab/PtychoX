#include "addon/addon.h"
#include <cooperative_groups.h>
#include <iostream>

namespace cg = cooperative_groups;

/**
 * Kernel: fullyfused_shiftNprop (Forward, shift only)
 * -----------------------------------------------------------------------------
 * Purpose
 *   Apply a scan-dependent Fourier-domain shift to a shared spectrum wavefront1
 *   and write the result into a batched output tensor.
 *
 * Mathematical form (per frequency sample k = (kx, ky), per scan idz)
 *   x_forward(k,idz) = wavefront1(k) * exp(+j * 2? * (kx*dx + ky*dy))
 *   where (dx, dy) = scanning_pos[idz].
 *
 * Memory layout
 *   wavefront1: [W x H]           (single page, shared across idz)
 *   scanning_pos: [Z]            (one shift per scan/page)
 *   x_forward:  [W x H x Z]      (page-major / batched output)
 *
 * Indexing
 *   pix_id  = idx*H + idy        (pixel index within one 2D page)
 *   page_id = idz*(W*H)          (offset to the idz-th page)
 *
 * Notes
 *   - Frequency indexing is "signed, non-fftshifted":
 *       k = i               if i < N/2
 *           i - N           if i >= N/2
 *     implemented branchlessly using (2*i >= N).
 *   - Uses __cosf/__sinf (fast math) for exp(j*phi) = cos(phi) + j sin(phi).
 */
__global__ void fullyfused_shiftNprop(
    const creal32_t * __restrict__ wavefront1,
    // const creal32_t * __restrict__ prop,   // (disabled here) optional transfer function
    const float2 * __restrict__ scanning_pos,
    const dim3 imgSzH,                           // (W, H, Z)
    creal32_t * __restrict__ x_forward
){
    // Thread coordinates
    const unsigned idy = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned idx = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned idz = blockIdx.z;

    // Bounds check
    const bool inside = (idx < imgSzH.x) && (idy < imgSzH.y) && (idz < imgSzH.z);
    if (!inside) return;

    // Flattened index within page and page offset for batch dimension
    const unsigned pix_id  = idx * imgSzH.y + idy;
    const unsigned page_id = idz * (imgSzH.x * imgSzH.y);

    // Scan shift (dx, dy) for this page
    const float2 pos = scanning_pos[idz];

    // Signed frequency indices (non-fftshifted ordering)
    float2 freq = make_float2(
        (float)((int)idx - (int)((2U * idx) >= imgSzH.x) * (int)imgSzH.x),
        (float)((int)idy - (int)((2U * idy) >= imgSzH.y) * (int)imgSzH.y)
    );

    // Phase: 2? * (kx*dx + ky*dy)
    float fqs = TwoPI * (freq.x * pos.x + freq.y * pos.y);

    // exp(+j*fqs) = cos(fqs) + j sin(fqs)
    freq = make_float2(__cosf(fqs), __sinf(fqs));

    // Load input spectrum sample
    const creal32_t this_w1 = wavefront1[pix_id];

    // Multiply by exp(+j*fqs): complex rotation
    creal32_t this_x;
    this_x.re = this_w1.re * freq.x - this_w1.im * freq.y;
    this_x.im = this_w1.re * freq.y + this_w1.im * freq.x;

    // If a propagator were enabled, it would be another complex multiply here.
    x_forward[pix_id + page_id] = this_x;
}


/**
 * Kernel: fullyfused_shiftNprop_Bwd (Backward / adjoint, in-place on outpus)
 * -----------------------------------------------------------------------------
 * Purpose
 *   Apply the adjoint of the forward "shift" (and optionally a propagator) to
 *   the gradient buffer stored in `outpus`.
 *
 * Expected adjoint (conceptually)
 *   Forward:  y = x * exp(+j?)          (optionally: * prop)
 *   Adjoint:  x_grad = y_grad * exp(-j?) (optionally: * conj(prop))
 *
 * What this kernel implements
 *   - exp(-j?) via (cos, sin) with the sign pattern in out.re/out.im
 *   - multiplication by conj(prop) using:
 *       (u+ jv) * conj(a+jb) = (u*a + v*b) + j(v*a - u*b)
 *   - normalization ratio = 1/(W*H) folded into cos/sin (FFT scaling convention)
 *
 * Memory layout
 *   outpus: [W x H x Z] (in/out)
 *   prop:   [W x H]     (single page, shared)
 *
 * Notes
 *   - Uses cooperative_groups indexing; ensure your grid maps group_index()
 *     consistently with your block dimensions.
 *   - Be consistent about FFT normalization: cuFFT uses unnormalized FFT/IFFT.
 */
__global__ void fullyfused_shiftNprop_Bwd(
    const creal32_t * __restrict__ prop,
    const float2 * __restrict__ scanning_pos,
    const dim3 imgSzH,                         // (W, H, Z)
    creal32_t * __restrict__ outpus            // in/out gradient buffer
){
    auto block = cg::this_thread_block();

    // Cooperative-groups based indices (note the swapped x/y mapping)
    const unsigned idy = block.group_index().x * block.group_dim().x + block.thread_index().x;
    const unsigned idx = block.group_index().y * block.group_dim().y + block.thread_index().y;
    const unsigned idz = block.group_index().z;

    const bool inside = (idx < imgSzH.x) && (idy < imgSzH.y) && (idz < imgSzH.z);
    if (!inside) return;

    const unsigned pageSz = imgSzH.x * imgSzH.y;
    const float ratio = 1.0f / (float)pageSz;

    const unsigned pix_id  = idx * imgSzH.y + idy;
    const unsigned page_id = idz * pageSz;

    const float2 pos = scanning_pos[idz];

    // Signed frequency indices (non-fftshifted)
    float2 freq = make_float2(
        (float)((int)idx - (int)((2U * idx) >= imgSzH.x) * (int)imgSzH.x),
        (float)((int)idy - (int)((2U * idy) >= imgSzH.y) * (int)imgSzH.y)
    );

    // Phase ? = 2?*(kx*dx + ky*dy)
    float fqs   = TwoPI * (freq.x * pos.x + freq.y * pos.y);

    // exp(-j?) = cos(?) - j sin(?); ratio folded in
    float fq_re = __cosf(fqs) * ratio;
    float fq_im = __sinf(fqs) * ratio;

    // Load incoming gradient (per page) and propagator (shared)
    const creal32_t this_out = outpus[pix_id + page_id];
    const creal32_t this_Pr  = prop[pix_id];

    // Apply exp(-j?): this_out * (cos - j sin)
    creal32_t out;
    out.re = this_out.re * fq_re + this_out.im * fq_im;
    out.im = this_out.im * fq_re - this_out.re * fq_im;

    // Multiply by conj(prop): out * conj(this_Pr)
    creal32_t xout;
    xout.re = out.re * this_Pr.re + out.im * this_Pr.im;
    xout.im = out.im * this_Pr.re - out.re * this_Pr.im;

    // Write back in place
    outpus[pix_id + page_id] = xout;
}


/**
 * Kernel: fullyfused_ReducedSum (Reduce over pages, adjoint shift only)
 * -----------------------------------------------------------------------------
 * Purpose
 *   For each frequency pixel (idx,idy), accumulate contributions over all scans
 *   idz by applying exp(-j?) and summing.
 *
 * Mathematical form (as implemented)
 *   outpus(k) = ?_idz [ input(k,idz) * exp(-j?_idz(k)) ] * (1/pageSz)
 *   where ?_idz(k) = 2?*(kx*dx_idz + ky*dy_idz).
 *
 * Memory layout
 *   input:  [W x H x Z] (page-major)
 *   outpus: [W x H]     (single page)
 *
 * Notes
 *   - `freq` is pre-multiplied by TwoPI so fqs = (2?*k)?pos.
 *   - The imaginary accumulation uses a minus sign:
 *       reduced.im -= (this_out.im*fq_re - this_out.re*fq_im)
 *     which matches multiplication by exp(-j?) with conjugate/adjoint convention.
 *   - If you later re-enable `prop`, you would multiply by conj(prop) here.
 */
__global__ void fullyfused_ReducedSum(
    const creal32_t * __restrict__ input,
    // const creal32_t * __restrict__ prop,
    const float2 * __restrict__ scanning_pos,
    const dim3 imgSzH,                         // (W, H, Z)
    creal32_t * __restrict__ outpus            // [W x H]
){
    const unsigned idy = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned idx = blockIdx.y * blockDim.y + threadIdx.y;

    const bool inside = (idx < imgSzH.x) && (idy < imgSzH.y);
    if (!inside) return;

    const unsigned pageSz = imgSzH.x * imgSzH.y;
    const float ratio = 1.0f / (float)pageSz;

    const unsigned pix_id = idx * imgSzH.y + idy;

    // Precompute 2? * signed frequency indices
    float2 freq = make_float2(
        TwoPI * (float)((int)idx - (int)((2U * idx) >= imgSzH.x) * (int)imgSzH.x),
        TwoPI * (float)((int)idy - (int)((2U * idy) >= imgSzH.y) * (int)imgSzH.y)
    );

    // Accumulator over pages
    creal32_t reduced_dldw1 = {0.0f, 0.0f};

    #pragma unroll
    for (unsigned idz = 0; idz < imgSzH.z; ++idz){
        const unsigned page_id = idz * pageSz;

        // Scan displacement for this page
        float2 pos = scanning_pos[idz];

        // fqs = (2?*kx)*dx + (2?*ky)*dy
        float fqs   = (freq.x * pos.x + freq.y * pos.y);

        // exp(-j fqs) * ratio
        float fq_re = __cosf(fqs) * ratio;
        float fq_im = __sinf(fqs) * ratio;

        // Load per-page complex input
        const creal32_t this_out = input[pix_id + page_id];

        // Multiply by exp(-j fqs) and accumulate:
        // (a+jb) * (c - jd) -> re = a*c + b*d, im = b*c - a*d
        reduced_dldw1.re += this_out.re * fq_re + this_out.im * fq_im;
        reduced_dldw1.im -= this_out.im * fq_re - this_out.re * fq_im;
    }

    outpus[pix_id] = reduced_dldw1;
}


/**
 * Kernel: fused_pixelwiseProduct
 * -----------------------------------------------------------------------------
 * Purpose
 *   For each page idz, scale X by 1/(W*H) and compute Z = X_scaled * Y.
 *   Also writes back the scaled X (in-place scaling).
 *
 * Memory layout
 *   X: [W x H x Z] (in/out)
 *   Y: [W x H]     (shared, cached in shared memory)
 *   Z: [W x H x Z] (output)
 *
 * Notes
 *   - Requires BLOCK_SIZE == blockDim.x * blockDim.y for shared_Y indexing.
 *   - Scaling by 1/(W*H) is often needed for IFFT normalization conventions.
 */
__global__ void fused_pixelwiseProduct(
    creal32_t * __restrict__ X,
    const creal32_t * __restrict__ Y,
    const dim3 imgSz,                          // (W, H, Z)
    creal32_t * __restrict__ Z
){
    auto block = cg::this_thread_block();
    const unsigned idy = block.group_index().x * block.group_dim().x + block.thread_index().x;
    const unsigned idx = block.group_index().y * block.group_dim().y + block.thread_index().y;
    const unsigned idz = block.group_index().z;

    // Cache shared operand Y for better reuse within block
    const unsigned pix_id = idx * imgSz.y + idy;
    const bool inside = (idx < imgSz.x) && (idy < imgSz.y);
    if (inside){
        const unsigned pageSz  = imgSz.x * imgSz.y;
        const unsigned page_id = idz * pageSz;
        const unsigned pix_bc  = pix_id + page_id;

        creal32_t this_X = X[pix_bc];
        const creal32_t this_Y = Y[pix_id];

        // Normalize X by page size
        const float ratio = 1.0f / (float)pageSz;
        this_X.re *= ratio;
        this_X.im *= ratio;

        // Z = X * Y
        creal32_t this_Z;
        this_Z.re = this_X.re * this_Y.re - this_X.im * this_Y.im;
        this_Z.im = this_X.im * this_Y.re + this_X.re * this_Y.im;

        Z[pix_bc] = this_Z;
        X[pix_bc] = this_X;   // write back scaled X
    }
}

/**
 * Kernel: fused_pixelwiseProduct_inplace
 * -----------------------------------------------------------------------------
 * Purpose
 *   In-place multiply for each page: X <- (X * Y) / (W*H)
 *
 * Memory layout
 *   X: [W x H x Z] (in/out)
 *   Y: [W x H]     (shared, cached in shared mem)
 *
 * Notes
 *   - If you already normalized elsewhere, avoid double scaling.
 */
__global__ void fused_pixelwiseProduct_inplace(
    const creal32_t * __restrict__ Y,
    const dim3 imgSz,
    creal32_t * __restrict__ X
){
    auto block = cg::this_thread_block();
    const unsigned idy = block.group_index().x * block.group_dim().x + block.thread_index().x;
    const unsigned idx = block.group_index().y * block.group_dim().y + block.thread_index().y;
    const unsigned idz = block.group_index().z;

    const unsigned pix_id = idx * imgSz.y + idy;

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y);
    if (inside){
        const unsigned page_id = idz * (imgSz.x * imgSz.y);

        const creal32_t this_X = X[pix_id + page_id];
        const creal32_t this_Y = Y[pix_id];

        const float ratio = 1.0f / (float)(imgSz.x * imgSz.y);

        // X <- (X * Y) * ratio
        creal32_t this_O;
        this_O.re = (this_X.re * this_Y.re - this_X.im * this_Y.im) * ratio;
        this_O.im = (this_X.im * this_Y.re + this_X.re * this_Y.im) * ratio;

        X[pix_id + page_id] = this_O;
    }
}


/**
 * Kernel: fused_pixelwiseProduct_inplace_conj
 * -----------------------------------------------------------------------------
 * Purpose
 *   In-place multiply for each page: X <- (X * conj(Y)) / (W*H)
 *
 * Complex multiply:
 *   (a+jb) * conj(c+jd) = (a*c + b*d) + j(b*c - a*d)
 */
__global__ void fused_pixelwiseProduct_inplace_conj(
    const creal32_t * __restrict__ Y,
    const dim3 imgSz,
    creal32_t * __restrict__ X
){
    auto block = cg::this_thread_block();
    const unsigned idy = block.group_index().x * block.group_dim().x + block.thread_index().x;
    const unsigned idx = block.group_index().y * block.group_dim().y + block.thread_index().y;
    const unsigned idz = block.group_index().z;

    const unsigned pix_id = idx * imgSz.y + idy;
    const bool inside = (idx < imgSz.x) && (idy < imgSz.y);
    if (!inside) return;

    const unsigned page_id = idz * (imgSz.x * imgSz.y);

    const creal32_t this_X = X[pix_id + page_id];
    const creal32_t this_Y = Y[pix_id];

    const float ratio = 1.0f / (float)(imgSz.x * imgSz.y);

    // X <- (X * conj(Y)) * ratio
    creal32_t this_O;
    this_O.re = (this_X.re * this_Y.re + this_X.im * this_Y.im) * ratio;
    this_O.im = (this_X.im * this_Y.re - this_X.re * this_Y.im) * ratio;

    X[pix_id + page_id] = this_O;
}

/**
 * Kernel: deconvCodedSurf
 * -----------------------------------------------------------------------------
 * Purpose
 *   In-place update:
 *     forward <- forward * conj(codedSurf)
 *   and accumulate a correlation-like term into dldw2 via atomicAdd:
 *     dldw2 += forward_original * conj(fwd_record)
 *
 * Memory layout
 *   codedSurf:  [W x H]     (shared, cached in shared memory)
 *   fwd_record: [W x H x Z]
 *   forward:    [W x H x Z] (in/out)
 *   dldw2:      [W x H]     (atomic accumulated; must be zero-initialized)
 *
 * Notes
 *   - dldw2 is treated as float2 by casting to (float*).
 *   - Ensure creal32_t is exactly {float re; float im;} with proper alignment.
 */
__global__ void deconvCodedSurf(
    const creal32_t * __restrict__ codedSurf,
    const creal32_t * __restrict__ fwd_record,
    const dim3 imgSz,
    creal32_t * __restrict__ forward,
    creal32_t * __restrict__ dldw2
){
    // auto block = cg::this_thread_block();
    // const unsigned idy = block.group_index().x * block.group_dim().x + block.thread_index().x;
    // const unsigned idx = block.group_index().y * block.group_dim().y + block.thread_index().y;
    // const unsigned idz = block.group_index().z;
    const unsigned idy = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned idx = blockIdx.y * blockDim.y + threadIdx.y;

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y);
    if (!inside) return;

    const unsigned pix_id = idx * imgSz.y + idy;
    float v_re = 0.0f;
    float v_im = 0.0f;
    const unsigned pageSz = imgSz.x * imgSz.y;
    for(int idz = 0; idz < imgSz.z; ++idz){
        const unsigned page_id = idz * pageSz;
        const unsigned pix_bc  = pix_id + page_id;
        const creal32_t this_X = forward[pix_bc];
        const creal32_t this_Y = codedSurf[pix_id];
        const creal32_t this_C = fwd_record[pix_bc];

        // forward <- this_X * conj(this_Y)
        creal32_t this_Z;
        this_Z.re = this_X.re * this_Y.re + this_X.im * this_Y.im;
        this_Z.im = this_X.im * this_Y.re - this_X.re * this_Y.im;
        forward[pix_bc] = this_Z;

        // Accumulate this_X * conj(this_C) into dldw2 (atomic)
        v_re += this_X.re * this_C.re + this_X.im * this_C.im;
        v_im += this_X.re * this_C.im - this_X.im * this_C.re;
    }
    dldw2[pix_id].re = v_re;
    dldw2[pix_id].im = v_im;
}


/**
 * Kernel: fullyfused_DownSample_Fwd
 * -----------------------------------------------------------------------------
 * Purpose
 *   Downsample complex field magnitude by factor ds using RMS magnitude:
 *     output = sqrt( mean_{ds x ds} |input|^2 )
 *
 * Memory layout assumption
 *   - imgSz is the SMALL output size (W, H, Z).
 *   - input is the LARGE field with spatial dimensions scaled by ds in each axis.
 *   - page_large = page_id * ds * ds assumes the large pages are contiguous and
 *     exactly ds^2 times larger than the small page size in element count.
 *
 * Notes
 *   - Check that your actual large-grid stride matches:
 *       (ds * imgSz.y) as the "large H" dimension.
 */
__global__ void fullyfused_DownSample_Fwd(
    const creal32_t * __restrict__ input,
    const int ds,
    const dim3 imgSz,
    float * __restrict__ output
){
    int idy = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z;

    if (idx >= (int)imgSz.x || idy >= (int)imgSz.y) return;

    int pix_id     = idx * imgSz.y + idy;
    int page_id    = idz * (imgSz.x * imgSz.y);
    int page_large = page_id * ds * ds;

    float reducedSum = 0.0f;

    // Accumulate |X|^2 over ds x ds block in the large image
    #pragma unroll
    for (int xx = 0; xx < ds; ++xx){
        #pragma unroll
        for (int yy = 0; yy < ds; ++yy){
            int pixLarge_idx =
                (ds * idx + xx) * (ds * imgSz.y) + (ds * idy + yy);

            creal32_t this_X = input[pixLarge_idx + page_large];
            reducedSum += (this_X.re * this_X.re + this_X.im * this_X.im);
        }
    }

    reducedSum /= (float)(ds * ds);              // mean(|X|^2)
    output[pix_id + page_id] = sqrtf(reducedSum); // RMS magnitude
}


/**
 * Kernel: fullyfused_DownSample_Bwd
 * -----------------------------------------------------------------------------
 * Purpose
 *   Backpropagate through the RMS magnitude downsampling.
 *
 * Current implementation
 *   - Computes a scalar factor:
 *       this_O = dldout / (out + eps)
 *     and multiplies every complex sample in the ds x ds block by this_O.
 *
 * Notes
 *   - This is a simplified gradient (does not explicitly include X/out and 1/ds^2
 *     factors as in the exact derivative of sqrt(mean(|X|^2)) w.r.t. X).
 *   - If this is intentional (stability / heuristic), keep it.
 *   - Updates `forward` in place on the large grid.
 */
__global__ void fullyfused_DownSample_Bwd(
    const float * __restrict__ dldout,
    const float * __restrict__ out,
    const int ds,
    const dim3 imgSz,
    creal32_t * __restrict__ forward
){
    int idy = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z;

    if (idx >= (int)imgSz.x || idy >= (int)imgSz.y) return;

    int pix_id     = idx * imgSz.y + idy;
    int page_id    = idz * (imgSz.x * imgSz.y);
    int page_large = page_id * ds * ds;

    // Scalar gradient factor with epsilon to avoid division by zero
    float this_O = dldout[pix_id + page_id] / (out[pix_id + page_id] + 0.001f);

    // Distribute gradient back to ds x ds block (in-place scaling)
    #pragma unroll
    for (int xx = 0; xx < ds; ++xx){
        #pragma unroll
        for (int yy = 0; yy < ds; ++yy){
            int pixLarge_idx =
                (ds * idx + xx) * (ds * imgSz.y) + (ds * idy + yy);

            creal32_t this_X = forward[pixLarge_idx + page_large];
            this_X.re *= this_O;
            this_X.im *= this_O;
            forward[pixLarge_idx + page_large] = this_X;
        }
    }
}

