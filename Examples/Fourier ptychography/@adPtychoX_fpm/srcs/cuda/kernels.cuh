#include "addon.h"
#include "fftshift_kernel.cuh"
#include <cooperative_groups.h>
#include <iostream>

namespace cg = cooperative_groups;

/**
 * Device helper: absC
 * -----------------------------------------------------------------------------
 * Purpose
 *   Compute the magnitude of a complex number and apply a scaling factor:
 *     |in| / ratio
 *
 * Notes
 *   - `ratio` is passed in as a float and used as a divisor.
 *   - Be careful: if ratio is very small, this amplifies noise; if ratio is 0,
 *     it will produce Inf/NaN.
 */
__device__ float absC(const creal32_T in, const float ratio){
    // sqrt(re^2 + im^2) / ratio
    float out = sqrtf(in.re * in.re + in.im * in.im) / ratio;
    return out;
}

/**
 * Device helper: sign
 * -----------------------------------------------------------------------------
 * Purpose
 *   Return the sign (+1 or -1) of a scalar using copysign.
 *
 * Notes
 *   - copysignf(1.0f, 0.0f) returns +1.0f, so sign(0) = +1 here.
 *     If you need sign(0)=0, you must handle it explicitly.
 */
__device__ float sign(const float in){
    float out = copysignf(1.0f, in);
    return out;
}


/**
 * Kernel: getSubpupil
 * -----------------------------------------------------------------------------
 * Purpose
 *   For each illumination (idz), extract a sub-spectrum ("subpupil") region from
 *   a larger Fourier spectrum wavefront1 (high-res), and also build a latent
 *   field latentZ by multiplying that sub-spectrum with the pupil wavefront2.
 *
 * Typical context (Fourier Ptychography / PIE-like update)
 *   - wavefront1: the global (high-res) Fourier spectrum of the object
 *   - wavefront2: the pupil function (low-res support)
 *   - ledindex[idz]: the shift offset (in pixels) locating the subpupil window
 *   - subwave: extracted patch (per idz)
 *   - latentZ: subwave .* pupil  (complex multiply, no conjugation)
 *
 * Memory layout
 *   wavefront1: [Hh x Wh]          (single page, high-res spectrum)
 *   wavefront2: [Hl x Wl]          (single page, pupil on low-res grid)
 *   subwave:    [Hl x Wl x Z]      (batched extracted patches)
 *   latentZ:    [Hl x Wl x Z]      (batched pupil-multiplied patches)
 *
 * Indexing notes
 *   - pix_id_large = (idx + pixL_x - 1, idy + pixL_y - 1) on the high-res grid.
 *   - The "-1" implies ledindex stores a 1-based coordinate or a convention
 *     that requires shifting by 1. Keep consistent across host code.
 *
 * Performance notes
 *   - Pupil is cached in shared memory (BLOCK_SIZE must equal block threads).
 *   - ledindex is read per thread but only depends on idz; consider caching
 *     ledindex[idz] in a register early (you already do).
 */
__global__ void getSubpupil(
    const creal32_t* __restrict__ wavefront1,
    const creal32_t* __restrict__ wavefront2,
    const int2* __restrict__ ledindex,
    const dim3 imgSz,    // low-res size: (Hl, Wl, Z)
    const dim3 imgSzL,   // high-res size: (Hh, Wh, ?), used for indexing wavefront1
    creal32_t* __restrict__ subwave,
    creal32_t* __restrict__ latentZ
){
    auto block = cg::this_thread_block();
    unsigned idx = block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned idy = block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned idz = block.group_index().z;

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y) && (idz < imgSz.z);

    // Subpupil top-left (or center) offset on the high-res Fourier plane
    unsigned pixL_x = (unsigned) ledindex[idz].x;
    unsigned pixL_y = (unsigned) ledindex[idz].y;

    // Cache pupil in shared memory (one element per thread)
    __shared__ creal32_t this_pupil[BLOCK_SIZE];
    unsigned tr = block.thread_rank();
    unsigned pix_id = idx * imgSz.y + idy;
    this_pupil[tr] = wavefront2[pix_id];
    block.sync();

    if (inside) {
        unsigned page_id = idz * (imgSz.x * imgSz.y);

        // Map low-res coordinate (idx,idy) into the high-res spectrum window
        // NOTE: `imgSzL.y` is the stride of the high-res grid.
        unsigned pix_id_large = (idx + pixL_x - 1) * imgSzL.y + (idy + pixL_y - 1);

        // Extract the sub-spectrum from the global spectrum
        creal32_t temp = wavefront1[pix_id_large];
        subwave[pix_id + page_id] = temp;

        // Multiply by pupil: latentZ = subwave * pupil
        creal32_t P = this_pupil[tr];
        creal32_t temp_latentZ;
        temp_latentZ.re = (temp.re * P.re - temp.im * P.im);
        temp_latentZ.im = (temp.re * P.im + temp.im * P.re);
        latentZ[pix_id + page_id] = temp_latentZ;
    }
}


/**
 * Kernel: fused_deconvPIE
 * -----------------------------------------------------------------------------
 * Purpose
 *   This kernel fuses two gradient-like operations commonly used in PIE/FPM:
 *
 *   (1) Compute a pupil gradient contribution (accumulate into dldw2):
 *       dldw2 += recordZ .* conj(subwave)          (as implemented via (a,b,c,d))
 *
 *   (2) Stitch/accumulate the Fourier spectrum update into the global spectrum
 *       gradient buffer dldw1 using ledindex offsets:
 *       dldw1(high_res_region) += recordZ .* conj(pupil)
 *
 * Inputs
 *   recordZ: [Hl x Wl x Z]   (per-frame complex residual / gradient in pupil plane)
 *   pupil:   [Hl x Wl]       (shared pupil function)
 *   subwave: [Hl x Wl x Z]   (extracted object sub-spectrum for each idz)
 *   ledindex:[Z]             (placement offsets into high-res Fourier plane)
 *
 * Outputs (must be zero-initialized before launch)
 *   dldw1: [Hh x Wh]         (accumulated high-res spectrum gradient, atomicAdd)
 *   dldw2: [Hl x Wl]         (accumulated pupil gradient, atomicAdd)
 *
 * Memory layout
 *   - dldw1 and dldw2 are treated as float2 via (float*) casting.
 *     Ensure creal32_t is exactly two 32-bit floats in memory.
 *
 * Notes / Footguns
 *   - Because atomicAdd is used, dldw1/dldw2 must be cleared to 0 before running.
 *   - `inside_FP` guards out-of-bounds writes when the subpupil would exceed the
 *     high-res spectrum bounds.
 *   - Shared pupil requires BLOCK_SIZE == blockDim.x * blockDim.y.
 */
__global__ void fused_deconvPIE(
    const creal32_t* __restrict__ recordZ,
    const creal32_t* __restrict__ pupil,
    const creal32_t* __restrict__ subwave,
    const int2* __restrict__ ledindex,
    const dim3 imLs_sz,    // low-res: (Hl, Wl, Z)
    const dim3 imHs_sz,    // high-res: (Hh, Wh, ?)
    creal32_t* __restrict__ dldw1, // high-res spectrum gradient (accumulated)
    creal32_t* __restrict__ dldw2  // pupil gradient (accumulated)
){
    auto block = cg::this_thread_block();
    unsigned idx = block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned idy = block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned idz = block.group_index().z;

    // Cache pupil in shared memory
    __shared__ creal32_t this_pupil[BLOCK_SIZE];
    unsigned tr = block.thread_rank();
    unsigned pix_id = idx * imLs_sz.y + idy;
    this_pupil[tr] = pupil[pix_id];
    block.sync();

    const bool inside = (idx < imLs_sz.x) && (idy < imLs_sz.y) && (idz < imLs_sz.z);
    if (inside){
        unsigned pageSz = imLs_sz.x * imLs_sz.y;
        unsigned idxZ   = idz * pageSz;
        unsigned pix_id = idx * imLs_sz.y + idy;

        // recordZ at (idx,idy,idz)
        creal32_t tempX = recordZ[pix_id + idxZ];

        // conj(pupil) multiplication:
        // tempX <- recordZ * conj(pupil)
        creal32_t tempP = this_pupil[tr];
        float a = tempX.re;
        float b = tempX.im;
        float c = tempP.re;
        float d = tempP.im;

        // (a+jb) * conj(c+jd) = (a+jb)*(c-jd)
        tempX.re = a * c + b * d;
        tempX.im = c * b - a * d;

        // pupil gradient accumulation: recordZ * conj(subwave)
        // Here you use original a,b (recordZ) with c,d = subwave re/im
        c = subwave[pix_id + idxZ].re;
        d = subwave[pix_id + idxZ].im;

        // (a+jb) * conj(c+jd) -> re = a*c + b*d, im = a*d - b*c
        float pupil_re = a * c + b * d;
        float pupil_im = a * d - c * b;

        // Accumulate into dldw2[pix_id] (as float2 via casting)
        float *temp = (float *) dldw2;
        atomicAdd(temp + 2 * pix_id + 0, pupil_re);
        atomicAdd(temp + 2 * pix_id + 1, pupil_im);

        // Stitch the low-res patch back to high-res Fourier plane
        unsigned pixL_x = (unsigned) ledindex[idz].x;
        unsigned pixL_y = (unsigned) ledindex[idz].y;

        const bool inside_FP =
            ((idx + pixL_x - 1) < imHs_sz.x) &&
            ((idy + pixL_y - 1) < imHs_sz.y);

        if (inside_FP){
            unsigned pix_id_large =
                (idx + pixL_x - 1) * imHs_sz.y + (idy + pixL_y - 1);

            // Accumulate into high-res gradient buffer
            float *temp_large = (float *) dldw1;
            atomicAdd(temp_large + pix_id_large * 2 + 0, tempX.re);
            atomicAdd(temp_large + pix_id_large * 2 + 1, tempX.im);
        }
    }
}


/**
 * Kernel: ifftCorrection
 * -----------------------------------------------------------------------------
 * Purpose
 *   Apply normalization and (optionally) conjugation correction after an IFFT.
 *
 * What it does
 *   spectrum <- spectrum * (1/(W*H)), and flips the imaginary sign:
 *     re <- re * ratio
 *     im <- -im * ratio
 *
 * Interpretation
 *   - The 1/(W*H) is standard if your IFFT is unnormalized (cuFFT inverse).
 *   - The sign flip on imaginary part is equivalent to complex conjugation,
 *     which sometimes appears when switching between FFT sign conventions or
 *     when using correlation vs convolution forms.
 *
 * Notes
 *   - You ignore idz in pix_id, so this kernel currently corrects ONLY one page
 *     unless spectrum already points to a single page. If spectrum is batched,
 *     add page offset (not changed here; just documenting).
 */
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

        float ratio = 1.0f / ((float)(imHs_sz.x * imHs_sz.y));

        // Apply scaling and conjugation-like sign flip on imaginary part
        float a = spectrum[pix_id].re;
        float b = spectrum[pix_id].im;
        spectrum[pix_id].re = a * ratio;
        spectrum[pix_id].im = -b * ratio;
    }
}


/**
 * Kernel: setConstraint
 * -----------------------------------------------------------------------------
 * Purpose
 *   Build a complex-valued constraint/gradient `out` from amplitude residuals
 *   between |latentz| and measured intensity img_Y, using a local sign-based
 *   finite-difference stencil.
 *
 * High-level idea (as implemented)
 *   1) Compute amplitude residual at current pixel:
 *        r(i,j) = |latentz(i,j)|/ratio - img_Y(i,j)
 *      then tempA = r(i,j) - img_Y(i,j)  (your code uses: absC - tempB)
 *
 *   2) For right and down neighbors, compute sign of (neighbor residual - tempA):
 *        blk_x stores sign(...) along +x direction
 *        blk_y stores sign(...) along +y direction
 *
 *   3) Fill boundary (left/up) signs for threads on block edges
 *
 *   4) Compute a discrete divergence-like quantity:
 *        tempA = (blk_x(left)-blk_x(right) + blk_y(up)-blk_y(down)) * pratio^2
 *
 *   5) Convert scalar tempA into a complex vector aligned with current phase:
 *        out = tempA * exp(j * angle(latentz))
 *
 * Inputs
 *   img_Y:   [Hl x Wl x Z]   measured amplitude (or magnitude-like) target
 *   latentz: [Hl x Wl x Z]   complex field (latent)
 *   pratio:  pupil/object sampling ratio (used in scaling)
 *
 * Outputs
 *   out:     [Hl x Wl x Z]   complex-valued constraint gradient
 *
 * Notes / Footguns
 *   - sign(0) returns +1 in your `sign()` implementation.
 *   - Shared arrays sizes: blk_x is (BLOCK_X+1)xBLOCK_Y, blk_y is BLOCK_Xx(BLOCK_Y+1)
 *     This supports computing left/right and up/down differences within a block.
 *   - The neighbor indexing wraps at image boundaries (periodic boundary condition).
 *   - `ratio = Hl*Wl*pratio^2` is used to scale amplitude; ensure this matches
 *     your forward normalization convention.
 */
__global__ void setConstraint(
    const dim3 imLs_sz,
    const int pratio,
    const real32_t* __restrict__ img_Y,
    const creal32_t* __restrict__ latentz,
    creal32_t* __restrict__ out
){
    auto block = cg::this_thread_block();
    unsigned tr_x = block.thread_index().x;
    unsigned tr_y = block.thread_index().y;

    unsigned idx = block.group_index().x * block.group_dim().x + tr_x;
    unsigned idy = block.group_index().y * block.group_dim().y + tr_y;

    // group_index().z acts as the batch index (page)
    const bool inside =
        (idx < imLs_sz.x) && (idy < imLs_sz.y) && (block.group_index().z < imLs_sz.z);

    unsigned pageSz  = imLs_sz.x * imLs_sz.y;
    unsigned page_id = block.group_index().z * pageSz;

    // Linear indices for current pixel and two forward neighbors (right, down)
    unsigned pix_A = idx * imLs_sz.y + idy;
    unsigned pix_X = (idx < (imLs_sz.x - 1)) ? ((idx + 1) * imLs_sz.y + idy) : idy;              // wrap x
    unsigned pix_Y = (idy < (imLs_sz.y - 1)) ? (idx * imLs_sz.y + idy + 1) : (idx * imLs_sz.y);  // wrap y

    // Shared sign buffers for computing differences across block boundaries
    __shared__ float blk_x[BLOCK_X + 1][BLOCK_Y];
    __shared__ float blk_y[BLOCK_X][BLOCK_Y + 1];

    creal32_t lat_A = latentz[pix_A + page_id];

    // Scaling used when mapping |latentz| to the measurement domain
    float ratio = (float)(imLs_sz.x * imLs_sz.y * pratio * pratio);

    // Residual at current pixel: |Z|/ratio - Y
    float tempB = img_Y[pix_A + page_id];
    float tempA = absC(lat_A, ratio) - tempB;

    // Store signs for +x and +y neighbors (shifted indices in shared arrays)
    blk_x[tr_x + 1][tr_y] =
        sign(absC(latentz[pix_X + page_id], ratio) - img_Y[pix_X + page_id] - tempA);

    blk_y[tr_x][tr_y + 1] =
        sign(absC(latentz[pix_Y + page_id], ratio) - img_Y[pix_Y + page_id] - tempA);

    block.sync();

    // Fill boundary entries (left edge for blk_x, top edge for blk_y) using wrap-around neighbors
    unsigned test_p2 = 0;

    if (tr_x == 0){
        // left neighbor (wrap)
        test_p2 = (idx == 0) ? ((imLs_sz.x - 1) * imLs_sz.y + idy) : ((idx - 1) * imLs_sz.y + idy);
        tempB = absC(latentz[test_p2 + page_id], ratio) - img_Y[test_p2 + page_id];
        blk_x[0][tr_y] = sign(tempA - tempB);
    }

    if (tr_y == 0){
        // up neighbor (wrap)
        test_p2 = (idy == 0) ? (idx * imLs_sz.y + imLs_sz.y - 1) : (idx * imLs_sz.y + idy - 1);
        tempB = absC(latentz[test_p2 + page_id], ratio) - img_Y[test_p2 + page_id];
        blk_y[tr_x][0] = sign(tempA - tempB);
    }

    block.sync();

    // Discrete divergence-like combination of sign differences
    tempA = (blk_x[tr_x][tr_y] - blk_x[tr_x + 1][tr_y] +
             blk_y[tr_x][tr_y] - blk_y[tr_x][tr_y + 1]) * ((float)pratio * pratio);

    // Extract current phase angle of latentz at this pixel
    float ang = atan2f(lat_A.im, lat_A.re);

    // Output a complex vector aligned with the current phase: tempA * exp(j*ang)
    if (inside){
        out[pix_A + page_id].re = cosf(ang) * tempA;
        out[pix_A + page_id].im = sinf(ang) * tempA;
    }
}


/**
 * Kernel: backward_Z
 * -----------------------------------------------------------------------------
 * Purpose
 *   Convert a real-valued gradient dl_doutput into a complex gradient on latentz
 *   by aligning it with the current phase of latentz.
 *
 * What it does (per pixel)
 *   ang = arg(latentz)
 *   latentz <- dl_doutput * exp(j * ang)
 *
 * Interpretation
 *   - If the forward mapping is something like output = |Z| or output depends
 *     only on the magnitude of Z, then the gradient w.r.t. Z often points along
 *     the current phase direction exp(j*arg(Z)).
 *
 * Notes
 *   - This overwrites latentz in place.
 *   - If you need to preserve original latentz for other gradients, store it elsewhere.
 */
__global__ void backward_Z(
    const dim3 imLs_sz,
    const real32_t* __restrict__ dl_doutput,
    creal32_t* __restrict__ latentz
){
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idz = blockIdx.z;

    const bool inside = (idx < imLs_sz.x) && (idy < imLs_sz.y) && (idz < imLs_sz.z);
    if (inside){
        unsigned pageSz  = imLs_sz.x * imLs_sz.y;
        unsigned pix_A   = idx * imLs_sz.y + idy;
        unsigned page_id = idz * pageSz;

        creal32_t lat_A = latentz[pix_A + page_id];

        // Current phase
        float ang = atan2f(lat_A.im, lat_A.re);

        // Real scalar gradient from upstream
        float tempA = dl_doutput[pix_A + page_id];

        // Map scalar gradient onto complex direction exp(j*ang)
        lat_A.re = cosf(ang) * tempA;
        lat_A.im = sinf(ang) * tempA;

        latentz[pix_A + page_id] = lat_A;
    }
}
