#include "addon.h"
#include <cooperative_groups.h>
#include <iostream>

namespace cg = cooperative_groups;

/**
 * Kernel: fullyfused_shiftNprop (Forward)
 * -----------------------------------------------------------------------------
 * Purpose
 *   For each scan/frame (idz), apply a spatial shift in Fourier domain to the
 *   input wavefront spectrum and then multiply by the propagation transfer
 *   function.
 *
 * Mathematical form (per pixel k = (kx, ky))
 *   X(k, idz) = wavefront1(k) * exp(+j * 2π * (kx*dx + ky*dy)) * prop(k)
 *   where (dx, dy) = scanning_pos[idz].
 *
 * Memory layout
 *   wavefront1: [HxW] (single page, shared for all idz)
 *   prop:       [HxW] (single page, shared for all idz)
 *   x_forward:  [HxW x Z] (batched output, page-major)
 *     pix_id  = idx*H + idy
 *     page_id = idz*(W*H)
 *     x_forward[pix_id + page_id] is the complex output for scan idz.
 *
 * Notes
 *   - Frequency indexing here is "non-fftshifted signed index":
 *       f = i               if i < N/2
 *           i - N           if i >= N/2
 *     implemented branchlessly using (2*i >= N).
 *   - The phase factor uses cos/sin to avoid complex exp.
 */
__global__ void fullyfused_shiftNprop(
    const creal32_t * __restrict__ wavefront1,
    const creal32_t * __restrict__ prop,
    const float2 * __restrict__ scanning_pos,
    const dim3 imgSzH,
    creal32_t * __restrict__ x_forward
){
    // Thread coordinates in (idx, idy, idz)
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned idz = blockIdx.z;

    // Bounds check (3D: W x H x Z)
    const bool inside = (idx < imgSzH.x) && (idy < imgSzH.y) && (idz < imgSzH.z);
    if (!inside) return;

    // Flattened indices: pix within a page, and page offset for the batch
    const unsigned pix_id  = idx * imgSzH.y + idy;                 // [0, W*H)
    const unsigned page_id = idz * (imgSzH.x * imgSzH.y);          // idz * pageSz

    // Scan displacement for this frame
    const float2 pos = scanning_pos[idz];                          // (dx, dy)

    // Signed frequency indices (non-fftshifted ordering):
    //   freq = i (i < N/2) else i - N
    float2 freq = make_float2(
        (float)((int)idx - (int)((2U * idx) >= imgSzH.x) * (int)imgSzH.x),
        (float)((int)idy - (int)((2U * idy) >= imgSzH.y) * (int)imgSzH.y)
    );

    // Phase argument: 2π * (kx*dx + ky*dy)
    float fqs = TwoPI * (freq.x * pos.x + freq.y * pos.y);

    // exp(+j fqs) = cos(fqs) + j sin(fqs)
    freq = make_float2(__cosf(fqs), __sinf(fqs));

    // Load input spectrum and propagator
    const creal32_t this_w1 = wavefront1[pix_id];
    const creal32_t this_Pr = prop[pix_id];

    // Apply Fourier-domain shift: wavefront1 * exp(+j fqs)
    creal32_t this_x;
    this_x.re = this_w1.re * freq.x - this_w1.im * freq.y;
    this_x.im = this_w1.re * freq.y + this_w1.im * freq.x;

    // Multiply by propagation transfer function: this_x * prop
    creal32_t this_O;
    this_O.re = this_x.re * this_Pr.re - this_x.im * this_Pr.im;
    this_O.im = this_x.re * this_Pr.im + this_x.im * this_Pr.re;

    // Store batched output
    x_forward[pix_id + page_id] = this_O;
}


/**
 * Kernel: fullyfused_shiftNprop_Bwd (Backward, in-place on `outpus`)
 * -----------------------------------------------------------------------------
 * Purpose
 *   Apply the adjoint of "shift + propagation" to the gradient stored in outpus.
 *   This is typically used to backpropagate through:
 *       forward: wavefront1 -> (shift) -> (multiply prop) -> x_forward
 *
 * Expected adjoint operations (conceptually)
 *   If forward does: y = (x * exp(+jφ)) * prop
 *   Then adjoint should do: x_grad = (y_grad * conj(prop)) * exp(-jφ)
 *   plus possible FFT/IFFT scaling depending on conventions.
 *
 * What this kernel currently implements
 *   - Multiplies by exp(-jφ) (implemented as combination of re/im)
 *   - Multiplies by conj(prop) (based on signs in complex multiply)
 *   - Applies a normalization factor ratio = 1/(W*H)
 *
 * Memory layout
 *   outpus: [HxW x Z], updated in place per page.
 *
 * Notes
 *   - Uses cooperative_groups to compute idx/idy/idz (same as blockIdx style).
 *   - The phase factor here uses cos/sin multiplied by `ratio` (FFT scaling).
 *   - Be consistent with your FFT library scaling: cuFFT does unnormalized FFT/IFFT.
 */
__global__ void fullyfused_shiftNprop_Bwd(
    const creal32_t * __restrict__ prop,
    const float2 * __restrict__ scanning_pos,
    const dim3 imgSzH,
    // in/out: gradient buffer over pages
    creal32_t * __restrict__ outpus
){
    auto block = cg::this_thread_block();

    // Note: you swapped x/y mapping vs the forward kernel:
    //   idy uses group_index().x, idx uses group_index().y
    const unsigned idy = block.group_index().x * block.group_dim().x + block.thread_index().x;
    const unsigned idx = block.group_index().y * block.group_dim().y + block.thread_index().y;
    const unsigned idz = block.group_index().z;

    const bool inside = (idx < imgSzH.x) && (idy < imgSzH.y) && (idz < imgSzH.z);
    if (!inside) return;

    const unsigned pageSz = imgSzH.x * imgSzH.y;
    const float ratio = 1.0f / (float)pageSz;  // normalization factor (FFT/IFFT convention)

    const unsigned pix_id  = idx * imgSzH.y + idy;
    const unsigned page_id = idz * pageSz;

    const float2 pos = scanning_pos[idz];

    // Signed frequency indices (non-fftshifted)
    float2 freq = make_float2(
        (float)((int)idx - (int)((2U * idx) >= imgSzH.x) * (int)imgSzH.x),
        (float)((int)idy - (int)((2U * idy) >= imgSzH.y) * (int)imgSzH.y)
    );

    // Phase argument: 2π * (kx*dx + ky*dy)
    float fqs   = TwoPI * (freq.x * pos.x + freq.y * pos.y);

    // exp(-j fqs) = cos(fqs) - j sin(fqs)
    // Here you fold the ratio into the trig outputs.
    float fq_re = __cosf(fqs) * ratio;
    float fq_im = __sinf(fqs) * ratio;

    // Load gradient at this pixel for this page
    const creal32_t this_out = outpus[pix_id + page_id];

    // Load propagator at this pixel (single page)
    const creal32_t this_Pr = prop[pix_id];

    // Apply exp(-jφ) to this_out (note the sign pattern)
    // out = this_out * (cos - j sin) * ratio
    creal32_t out;
    out.re = this_out.re * fq_re + this_out.im * fq_im;
    out.im = this_out.im * fq_re - this_out.re * fq_im;

    // Multiply by conj(prop):
    // xout = out * conj(this_Pr)
    // conj(a+jb) = a - jb
    // (re,im) formula:
    creal32_t xout;
    xout.re = out.re * this_Pr.re + out.im * this_Pr.im;
    xout.im = out.im * this_Pr.re - out.re * this_Pr.im;

    // Write back in place
    outpus[pix_id + page_id] = xout;
}


/**
 * Kernel: fullyfused_ReducedSum
 * -----------------------------------------------------------------------------
 * Purpose
 *   For each spatial frequency pixel (idx, idy), reduce/sum across all pages idz.
 *   Intended use: accumulate gradient contributions over scans for wavefront1
 *   (or some shared parameter) after applying the adjoint shift and conj(prop).
 *
 * Current implementation (IMPORTANT)
 *   - reduced_dldw1 is initialized once.
 *   - Inside the loop, you compute a per-page contribution into `pos`.
 *   - BUT you assign:
 *         reduced_dldw1.re = ...
 *         reduced_dldw1.im = ...
 *     instead of accumulating with +=.
 *   This means the final output is ONLY from the last idz (overwrite), not a sum.
 *   If this is intentional (e.g., debugging), keep it; otherwise it's a bug.
 *
 * Mathematical intention (likely)
 *   dldw1(k) = Sum_idz  [ input(k,idz) * exp(-j Phi _idz(k)) * conj(prop(k)) ] * (1/pageSz)
 *
 * Memory layout
 *   input:  [HxW x Z] page-major
 *   prop:   [HxW]
 *   outpus: [HxW]     reduced result per frequency pixel
 */
__global__ void fullyfused_ReducedSum(
    const creal32_t * __restrict__ input,
    const creal32_t * __restrict__ prop,
    const float2 * __restrict__ scanning_pos,
    const dim3 imgSzH,
    creal32_t * __restrict__ outpus
){
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;

    const bool inside = (idx < imgSzH.x) && (idy < imgSzH.y);
    if (!inside) return;

    const unsigned pageSz = imgSzH.x * imgSzH.y;
    const float ratio = 1.0f / (float)pageSz; // to cancel the effect of ifft2

    const unsigned pix_id  = idx * imgSzH.y + idy;
    const creal32_t this_Pr = prop[pix_id]; // propagator at this pixel

    // Precompute 2π * signed frequency indices
    float2 freq = make_float2(
        TwoPI * (float)((int)idx - (int)((2U * idx) >= imgSzH.x) * (int)imgSzH.x),
        TwoPI * (float)((int)idy - (int)((2U * idy) >= imgSzH.y) * (int)imgSzH.y)
    );

    // Accumulator for reduced sum over idz
    creal32_t reduced_dldw1 = {0.0f, 0.0f};

    #pragma unroll
    for (unsigned idz = 0; idz < imgSzH.z; ++idz){
        const unsigned page_id = idz * pageSz;
        float2 pos = scanning_pos[idz];

        // Phase argument: (2π*kx)*dx + (2π*ky)*dy
        float fqs   = (freq.x * pos.x + freq.y * pos.y);

        // exp(-j fqs) * ratio
        float fq_re = __cosf(fqs) * ratio;
        float fq_im = __sinf(fqs) * ratio;

        // Load per-page complex value
        const creal32_t this_out = input[pix_id + page_id];

        // Apply exp(-jφ): (cos - j sin)
        // Using (re,im) formula:
        pos.x = this_out.re * fq_re + this_out.im * fq_im; // real part after phase
        pos.y = this_out.im * fq_re - this_out.re * fq_im; // imag part after phase

        // Multiply by conj(prop)
        // NOTE: currently OVERWRITES reduced_dldw1 rather than accumulating.
        reduced_dldw1.re = pos.x * this_Pr.re + pos.y * this_Pr.im;
        reduced_dldw1.im = pos.x * this_Pr.im - pos.y * this_Pr.re;
        // If you intended sum, it should be:
        // reduced_dldw1.re += ...
        // reduced_dldw1.im += ...
    }

    outpus[pix_id] = reduced_dldw1;
}


/**
 * Kernel: fused_pixelwiseProduct
 * -----------------------------------------------------------------------------
 * Purpose
 *   Compute Z = (X/pageSz) .* Y for each page idz, and also write back the scaled X.
 *   This is a fused "scale + complex multiply" kernel.
 *
 * Mathematical form
 *   X_scaled = X / (W*H)
 *   Z = X_scaled * Y
 *
 * Memory layout
 *   X: [HxW x Z] (in/out)
 *   Y: [HxW]     (shared for all pages)
 *   Z: [HxW x Z] (output)
 *
 * Notes
 *   - Uses shared memory to cache Y per block for better reuse.
 *   - BLOCK_SIZE must match blockDim.x * blockDim.y (thread_rank indexing).
 */
__global__ void fused_pixelwiseProduct(
    creal32_t * __restrict__ X,
    const creal32_t * __restrict__ Y,
    const dim3 imgSz,
    creal32_t * __restrict__ Z
){
    auto block = cg::this_thread_block();

    // Map thread to (idx, idy, idz)
    const unsigned idy = block.group_index().x * block.group_dim().x + block.thread_index().x;
    const unsigned idx = block.group_index().y * block.group_dim().y + block.thread_index().y;
    const unsigned idz = block.group_index().z;

    // Cache Y in shared memory (one element per thread)
    __shared__ creal32_t shared_Y[BLOCK_SIZE];
    const unsigned pix_id = idx * imgSz.y + idy;
    const unsigned tr = block.thread_rank();
    shared_Y[tr] = Y[pix_id];
    block.sync();

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y);
    if (inside){
        const unsigned pageSz  = imgSz.x * imgSz.y;
        const unsigned page_id = idz * pageSz;
        const unsigned pix_bc  = pix_id + page_id;

        creal32_t this_X = X[pix_bc];
        const creal32_t this_Y = shared_Y[tr];

        // Normalize (typical IFFT scaling or convention)
        const float ratio = 1.0f / (float)pageSz;
        this_X.re *= ratio;
        this_X.im *= ratio;

        // Complex multiply: Z = X * Y
        creal32_t this_Z;
        this_Z.re = this_X.re * this_Y.re - this_X.im * this_Y.im;
        this_Z.im = this_X.im * this_Y.re + this_X.re * this_Y.im;

        Z[pix_bc] = this_Z;
        X[pix_bc] = this_X; // write back scaled X (fused op)
    }
}


/**
 * Kernel: pixelwiseProduct_conj
 * -----------------------------------------------------------------------------
 * Purpose
 *   Compute Z = X .* conj(Y) for each page idz (Y shared across pages).
 *
 * Complex multiply with conjugate
 *   conj(Y) = (Yr, -Yi)
 *   X * conj(Y) =
 *     re: Xr*Yr + Xi*Yi
 *     im: Xi*Yr - Xr*Yi
 *
 * Layout
 *   X: [HxW x Z]
 *   Y: [HxW]
 *   Z: [HxW x Z]
 */
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

        // Z = X * conj(Y)
        creal32_t this_Z;
        this_Z.re = this_X.re * this_Y.re + this_X.im * this_Y.im;
        this_Z.im = this_X.im * this_Y.re - this_X.re * this_Y.im;

        Z[pix_id + page_id] = this_Z;
    }
}


/**
 * Kernel: fused_pixelwiseProduct_inplace
 * -----------------------------------------------------------------------------
 * Purpose
 *   In-place: X = (X .* Y) / (W*H) for each page idz.
 *
 * Notes
 *   - Equivalent to complex multiplication followed by global scaling.
 *   - Y is shared (single page) and cached in shared memory.
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

    __shared__ creal32_t shared_Y[BLOCK_SIZE];
    const unsigned pix_id = idx * imgSz.y + idy;
    const unsigned tr = block.thread_rank();
    shared_Y[tr] = Y[pix_id];
    block.sync();

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y);
    if (inside){
        const unsigned page_id = idz * (imgSz.x * imgSz.y);

        const creal32_t this_X = X[pix_id + page_id];
        const creal32_t this_Y = shared_Y[tr];

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
 *   In-place: X = (X .* conj(Y)) / (W*H) for each page idz.
 *
 * Complex multiply:
 *   X * conj(Y):
 *     re = Xr*Yr + Xi*Yi
 *     im = Xi*Yr - Xr*Yi
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

    __shared__ creal32_t shared_Y[BLOCK_SIZE];
    const unsigned pix_id = idx * imgSz.y + idy;
    const unsigned tr = block.thread_rank();
    shared_Y[tr] = Y[pix_id];
    block.sync();

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y);
    if (!inside) return;

    const unsigned page_id = idz * (imgSz.x * imgSz.y);

    const creal32_t this_X = X[pix_id + page_id];
    const creal32_t this_Y = shared_Y[tr];

    const float ratio = 1.0f / (float)(imgSz.x * imgSz.y);

    // X <- (X * conj(Y)) * ratio
    creal32_t this_O;
    this_O.re = (this_X.re * this_Y.re + this_X.im * this_Y.im) * ratio;
    this_O.im = (this_X.im * this_Y.re - this_X.re * this_Y.im) * ratio;

    X[pix_id + page_id] = this_O;
}


/**
 * Kernel: ifftCorrection
 * -----------------------------------------------------------------------------
 * Purpose
 *   Apply 1/(W*H) normalization on a single 2D spectrum page.
 *   Useful when using an unnormalized IFFT (e.g., cuFFT inverse).
 *
 * Layout
 *   spectrum: [HxW] single page
 *
 * Notes
 *   - If you already normalize elsewhere, avoid double scaling.
 */
__global__ void ifftCorrection(
    creal32_t* __restrict__ spectrum,
    const dim3 imHs_sz
){
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;

    const bool inside = (idx < imHs_sz.x) && (idy < imHs_sz.y);
    if (inside){
        unsigned pix_id = idx * imHs_sz.y + idy;

        float ratio = 1.0f / ((float)(imHs_sz.x * imHs_sz.y));
        spectrum[pix_id].re *= ratio;
        spectrum[pix_id].im *= ratio;
    }
}


/**
 * Kernel: ifftCorrection_many
 * -----------------------------------------------------------------------------
 * Purpose
 *   Apply 1/(W*H) normalization for a batched spectrum: [HxW x Z].
 */
__global__ void ifftCorrection_many(
    creal32_t * __restrict__ spectrum,
    const dim3 imgSz
){
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idz = blockIdx.z;

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y) && (idz < imgSz.z);
    if (inside){
        unsigned page_id = idz * (imgSz.x * imgSz.y);
        unsigned pix_id  = idx * imgSz.y + idy;

        creal32_t temp = spectrum[pix_id + page_id];
        float ratio = 1.0f / (float)(imgSz.x * imgSz.y);
        temp.re *= ratio;
        temp.im *= ratio;
        spectrum[pix_id + page_id] = temp;
    }
}


/**
 * Kernel: reducedSum_Bwd_simple
 * -----------------------------------------------------------------------------
 * Purpose
 *   Reduce (sum) across batch pages for a complex buffer `forward`, producing
 *   a single-page gradient `dldw1`.
 *
 * Mathematical form (as implemented)
 *   reduced.re += forward.re
 *   reduced.im -= forward.im   (note the minus sign: corresponds to conj in adjoint)
 *
 * Layout
 *   forward: [HxW x Z]
 *   dldw1:   [HxW]
 */
__global__ void reducedSum_Bwd_simple(
    const creal32_t * __restrict__ forward,
    const dim3 imgSz,
    creal32_t * __restrict__ dldw1
){
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y);
    if (!inside) return;

    creal32_t reduced_dldw1 = {0.0f, 0.0f};
    const unsigned pix_id = idx * imgSz.y + idy;

    #pragma unroll
    for (unsigned idz = 0; idz < imgSz.z; ++idz){
        const unsigned page_id = idz * (imgSz.x * imgSz.y);
        const unsigned pix_bc  = pix_id + page_id;

        const creal32_t this_fwd = forward[pix_bc];

        reduced_dldw1.re += this_fwd.re;
        reduced_dldw1.im -= this_fwd.im; // conj-like accumulation
    }
    dldw1[pix_id] = reduced_dldw1;
}


/**
 * Kernel: reducedSum_Bwd
 * -----------------------------------------------------------------------------
 * Purpose
 *   Reduce across pages for two gradient outputs:
 *     - dldw1: sum of `forward` (with imag sign flip)
 *     - dldw2: sum of bwd_record * conj(fwd_record) (with imag sign handling)
 *
 * Layout
 *   forward, bwd_record, fwd_record: [HxW x Z]
 *   dldw1, dldw2:                   [HxW]
 */
__global__ void reducedSum_Bwd(
    const creal32_t * __restrict__ forward,
    const creal32_t * __restrict__ bwd_record,
    const creal32_t * __restrict__ fwd_record,
    const dim3 imgSz,
    creal32_t * __restrict__ dldw1,
    creal32_t * __restrict__ dldw2
){
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y);
    if (!inside) return;

    creal32_t reduced_dldw1 = {0.0f, 0.0f};
    creal32_t reduced_dldw2 = {0.0f, 0.0f};
    const unsigned pix_id = idx * imgSz.y + idy;

    #pragma unroll
    for (unsigned idz = 0; idz < imgSz.z; ++idz){
        const unsigned page_id = idz * (imgSz.x * imgSz.y);
        const unsigned pix_bc  = pix_id + page_id;

        const creal32_t this_fwd    = forward[pix_bc];
        const creal32_t this_bwdrec = bwd_record[pix_bc];
        const creal32_t this_fwdrec = fwd_record[pix_bc];

        // dldw1 accumulates forward with an imaginary sign flip (adjoint convention)
        reduced_dldw1.re += this_fwd.re;
        reduced_dldw1.im -= this_fwd.im;

        // dldw2 accumulates: bwd_record * conj(fwd_record)
        // real: ar*br + ai*bi
        // imag: ai*br - ar*bi  (then another sign flip per your convention)
        reduced_dldw2.re += this_bwdrec.re * this_fwdrec.re + this_bwdrec.im * this_fwdrec.im;
        reduced_dldw2.im -= this_bwdrec.im * this_fwdrec.re - this_bwdrec.re * this_fwdrec.im;
    }

    dldw1[pix_id] = reduced_dldw1;
    dldw2[pix_id] = reduced_dldw2;
}


/**
 * Kernel: deconvCodedSurf
 * -----------------------------------------------------------------------------
 * Purpose
 *   1) Update `forward` in-place by multiplying with conj(codedSurf) (as implemented).
 *   2) Accumulate per-pixel scalar products into dldw2 via atomicAdd.
 *
 * Layout
 *   codedSurf:  [HxW]     (shared across pages, cached in shared mem)
 *   fwd_record: [HxW x Z]
 *   forward:    [HxW x Z] (in/out)
 *   dldw2:      [HxW]     (atomic accumulated; stored as complex float pair)
 *
 * Notes
 *   - Atomic adds are performed on float* view of dldw2 (2 floats per pixel).
 *   - Make sure dldw2 is zero-initialized before launching this kernel.
 */
__global__ void deconvCodedSurf(
    const creal32_t * __restrict__ codedSurf,
    const creal32_t * __restrict__ fwd_record,
    const dim3 imgSz,
    creal32_t * __restrict__ forward,
    creal32_t * __restrict__ dldw2
){
    auto block = cg::this_thread_block();
    const unsigned idy = block.group_index().x * block.group_dim().x + block.thread_index().x;
    const unsigned idx = block.group_index().y * block.group_dim().y + block.thread_index().y;
    const unsigned idz = block.group_index().z;

    __shared__ creal32_t shared_Y[BLOCK_SIZE];
    const unsigned pix_id = idx * imgSz.y + idy;
    const unsigned tr = block.thread_rank();
    shared_Y[tr] = codedSurf[pix_id];
    block.sync();

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y);
    if (inside) {
        const unsigned page_id = idz * (imgSz.x * imgSz.y);

        const creal32_t this_X = forward[pix_id + page_id];
        const creal32_t this_Y = shared_Y[tr];
        const creal32_t this_C = fwd_record[pix_id + page_id];

        // forward <- this_X * conj(this_Y)
        creal32_t this_Z;
        this_Z.re = this_X.re * this_Y.re + this_X.im * this_Y.im;
        this_Z.im = this_X.im * this_Y.re - this_X.re * this_Y.im;
        forward[pix_id + page_id] = this_Z;

        // Accumulate correlation-like term into dldw2 (atomic)
        float v_re = this_X.re * this_C.re + this_X.im * this_C.im;
        float v_im = this_X.re * this_C.im - this_X.im * this_C.re;

        float * temp = (float *) dldw2;     // interpret complex as 2 floats
        atomicAdd(temp + pix_id * 2,     v_re);
        atomicAdd(temp + pix_id * 2 + 1, v_im);
    }
}


/**
 * Kernel: fullyfused_DownSample_Fwd
 * -----------------------------------------------------------------------------
 * Purpose
 *   Downsample intensity magnitude by factor ds (box reduction):
 *     output = sqrt( mean_{ds x ds} |input|^2 )
 *
 * Layout
 *   input:  [ (ds*H) x (ds*W) x Z ]   (larger grid)
 *   output: [ H x W x Z ]             (downsampled magnitude)
 *
 * Notes
 *   - `imgSz` here is the SMALL size (H x W x Z).
 *   - `page_large = page_id * ds * ds` assumes each page in `input` is ds^2 larger.
 *     Make sure this matches your actual memory layout.
 */
__global__ void fullyfused_DownSample_Fwd(
    const creal32_t * __restrict__ input,
    const int ds,
    const dim3 imgSz,
    float * __restrict__ output
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z;

    if (idx >= (int)imgSz.x || idy >= (int)imgSz.y) return;

    int pix_id     = idx * imgSz.y + idy;
    int page_id    = idz * (imgSz.x * imgSz.y);          // small-page offset
    int page_large = page_id * ds * ds;                  // large-page offset (assumed)

    float reducedSum = 0.0f;

    // Box-reduce over ds x ds neighborhood in the large image
    #pragma unroll
    for (int xx = 0; xx < ds; ++xx){
        #pragma unroll
        for (int yy = 0; yy < ds; ++yy){
            int pixLarge_idx =
                (ds * idx + xx) * (ds * imgSz.y) + (ds * idy + yy);

            creal32_t this_X = input[pixLarge_idx + page_large];
            reducedSum += (this_X.re * this_X.re + this_X.im * this_X.im); // |X|^2
        }
    }

    reducedSum /= (float)(ds * ds);                 // mean of |X|^2
    output[pix_id + page_id] = sqrtf(reducedSum);   // RMS magnitude
}


/**
 * Kernel: fullyfused_DownSample_Bwd
 * -----------------------------------------------------------------------------
 * Purpose
 *   Backprop through output = sqrt(mean(|X|^2)):
 *     dL/dX = dL/dout * dout/dX
 *
 * Current implementation
 *   - Computes scalar factor:
 *       this_O = dldout / (out + eps)
 *     then multiplies each complex sample in the ds x ds block by this_O.
 *
 * Notes
 *   - This is a simplified gradient that ignores the exact derivative of
 *     mean(|X|^2) wrt X (which would include X/out and 1/(ds^2) factors).
 *   - If you intentionally use this approximation for stability, keep it.
 *   - forward is updated in-place for the large grid.
 */
__global__ void fullyfused_DownSample_Bwd(
    const float * __restrict__ dldout,
    const float * __restrict__ out,
    const int ds,
    const dim3 imgSz,
    creal32_t * __restrict__ forward
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z;

    if (idx >= (int)imgSz.x || idy >= (int)imgSz.y) return;

    int pix_id     = idx * imgSz.y + idy;
    int page_id    = idz * (imgSz.x * imgSz.y);
    int page_large = page_id * ds * ds;

    // Gradient scaling (with epsilon to avoid division by zero)
    float this_O = dldout[pix_id + page_id] / (out[pix_id + page_id] + 0.001f);

    // Distribute the scalar gradient back to the ds x ds block (in-place)
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
