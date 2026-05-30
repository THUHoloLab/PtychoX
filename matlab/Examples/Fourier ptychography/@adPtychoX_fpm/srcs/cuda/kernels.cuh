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
 * Kernel: fused_getSubpupil_shift
 * -----------------------------------------------------------------------------
 * Purpose
 *   Fused forward kernel for the shifted Fourier-pupil extraction path.
 *   For each illumination/view (idz), it extracts a low-resolution sub-spectrum
 *   from the high-resolution object spectrum and multiplies it by the pupil to
 *   produce latentZ.
 *
 *   This variant folds shift logic into indexing:
 *   - the read from wavefront1 uses a shifted high-resolution coordinate
 *   - the write to latentZ uses a shifted low-resolution coordinate
 *
 *   The intent is to avoid launching separate fftshift kernels, but it also
 *   means this kernel must stay strictly consistent with
 *   fused_deconvPIE_shifted() in the backward pass.
 *
 * Inputs
 *   wavefront1: [Hh x Wh]
 *     High-resolution object spectrum after the forward FFT.
 *
 *   wavefront2: [Hl x Wl]
 *     Shared pupil function on the low-resolution support.
 *
 *   ledindex: [Z]
 *     Per-view subpupil offsets on the high-resolution Fourier plane.
 *     The current indexing uses (idx + ledindex.x - 1, idy + ledindex.y - 1),
 *     so this path assumes the same 1-based offset convention as the original
 *     getSubpupil() kernel.
 *
 *   imgSz:
 *     Low-resolution tensor shape packed as dim3(x=width, y=height, z=batch).
 *
 *   imgSzL:
 *     High-resolution spectrum shape packed as dim3(x=width, y=height, z=?).
 *
 * Outputs
 *   subwave: [Hl x Wl x Z]
 *     Extracted sub-spectrum for each illumination.
 *
 *   latentZ: [Hl x Wl x Z]
 *     Complex field subwave .* pupil, written in a shifted low-resolution
 *     layout for the downstream FFT/IFFT path.
 */
__global__ void fused_getSubpupil_shift(
    const creal32_t* __restrict__ wavefront1,
    const creal32_t* __restrict__ wavefront2,
    const int2* __restrict__ ledindex,
    const dim3 imgSz,    // low-res size: (Hl, Wl, Z)
    const dim3 imgSzL,   // high-res size: (Hh, Wh, 1), used for indexing wavefront1
    creal32_t* __restrict__ subwave,
    creal32_t* __restrict__ latentZ
){
    const unsigned idy = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned idx = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned idz = blockIdx.z;
    
    const bool inside = (idx < imgSz.x) && (idy < imgSz.y) && (idz < imgSz.z);

    // Subpupil offset on the high-resolution Fourier plane for this view.
    const unsigned pixL_x = (unsigned) ledindex[idz].x;
    const unsigned pixL_y = (unsigned) ledindex[idz].y;

    // Linear index on the low-resolution grid for the current thread.
    const unsigned pix_id = idx * imgSz.y + idy;

    if (inside) {
        // Per-view page offset in the [Hl x Wl x Z] output tensors.
        const unsigned page_id = idz * (imgSz.x * imgSz.y);

        // Read the high-resolution spectrum through a shifted coordinate,
        // equivalent to applying a high-res fftshift during sampling.
        unsigned sx_large = idx + pixL_x - 1 + (imgSzL.x >> 1);
        unsigned sy_large = idy + pixL_y - 1 + (imgSzL.y >> 1);
        if (sx_large >= imgSzL.x) sx_large -= imgSzL.x;
        if (sy_large >= imgSzL.y) sy_large -= imgSzL.y;
        const unsigned pix_id_large = sx_large * imgSzL.y + sy_large;

        // Extract the shifted sub-spectrum patch for this LED/view.
        creal32_t temp = wavefront1[pix_id_large];
        subwave[pix_id + page_id] = temp;

        // Multiply by pupil: latentZ = subwave .* pupil
        creal32_t P = wavefront2[pix_id];
        creal32_t temp_latentZ;
        temp_latentZ.re = (temp.re * P.re - temp.im * P.im);
        temp_latentZ.im = (temp.re * P.im + temp.im * P.re);

        // Store latentZ in a shifted low-resolution layout so the following
        // transform path can consume it without an explicit fftshift kernel.
        unsigned sx = idx + (imgSz.x >> 1);
        unsigned sy = idy + (imgSz.y >> 1);
        if (sx >= imgSz.x) sx -= imgSz.x;
        if (sy >= imgSz.y) sy -= imgSz.y;
        const unsigned shifted_pix_id = sx * imgSz.y + sy;
        latentZ[shifted_pix_id + page_id] = temp_latentZ;
    }
}


/**
 * Kernel: fused_deconvPIE_shifted
 * -----------------------------------------------------------------------------
 * Purpose
 *   Fused backward kernel for the "shifted" forward path. It performs two
 *   coupled gradient accumulations in one pass:
 *
 *   (1) Accumulate the pupil gradient into dldw2:
 *         dldw2 += recordZ .* conj(subwave)
 *
 *   (2) Accumulate the object-spectrum gradient into dldw1:
 *         dldw1(high_res_window) += recordZ .* conj(pupil)
 *
 *   Compared with fused_deconvPIE(), this variant assumes the forward kernel
 *   wrote latentZ / subwave using an inlined fftshift convention, so the
 *   backward kernel must undo or mirror those layout choices when reading
 *   recordZ and when writing back to the high-resolution spectrum grid.
 *
 * Inputs
 *   recordZ: [Hl x Wl x Z]
 *     Complex gradient in the low-resolution pupil plane after:
 *       backward_Z(...)
 *       cufftExecC2C(..., CUFFT_FORWARD)
 *     This buffer is read with an inlined low-res fftshift.
 *
 *   pupil: [Hl x Wl]
 *     Shared pupil function on the low-resolution grid.
 *
 *   subwave: [Hl x Wl x Z]
 *     Extracted object sub-spectrum for each LED / view. This should match the
 *     same coordinate convention used by fused_getSubpupil_shift().
 *
 *   ledindex: [Z]
 *     Per-view subpupil offsets on the high-resolution Fourier plane.
 *     The current arithmetic uses (idx + ledindex.x - 1, idy + ledindex.y - 1),
 *     so host-side code must keep this 1-based offset convention consistent.
 *
 *   imLs_sz:
 *     Low-resolution tensor shape packed as dim3(x=width, y=height, z=batch).
 *
 *   imHs_sz:
 *     High-resolution spectrum shape packed as dim3(x=width, y=height, z=?).
 *
 * Outputs
 *   dldw1: [Hh x Wh]
 *     Accumulated high-resolution object-spectrum gradient. The writeback uses
 *     atomicAdd because many low-res views may overlap on the same Fourier
 *     coordinate. This buffer must be zero-initialized before launch.
 *
 *   dldw2: [Hl x Wl]
 *     Accumulated pupil gradient across all views. Also requires zero-init
 *     before launch because accumulation is done with atomicAdd.
 *
 * Shift / layout notes
 *   - recordZ is not read at pix_id directly. Instead, this kernel first maps
 *     (idx, idy) to a shifted coordinate rec_id, equivalent to applying a
 *     low-resolution fftshift on read.
 *   - The stitched writeback to dldw1 also applies a high-resolution shift on
 *     the destination coordinate before atomic accumulation.
 *   - Because the forward path also folds shift logic into indexing, this
 *     kernel should be kept in sync with fused_getSubpupil_shift().
 *
 * Arithmetic notes
 *   - obj_re / obj_im implement:
 *         recordZ * conj(pupil)
 *   - pupil_re / pupil_im implement:
 *         recordZ * conj(subwave)
 *   - dldw1 and dldw2 are reinterpreted as float* so the real and imaginary
 *     parts can be accumulated with atomicAdd. This assumes creal32_t is a
 *     tightly packed pair of 32-bit floats.
 *
 * Footguns
 *   - Any mismatch between the forward and backward shift convention will
 *     silently move spectrum energy to the wrong Fourier coordinates.
 *   - The ledindex convention is effectively 1-based because of the "- 1".
 *   - If dldw1 / dldw2 are not cleared before launch, gradients will include
 *     stale data from previous iterations.
 */
__global__ void fused_deconvPIE_shifted(
    const creal32_t* __restrict__ recordZ,
    const creal32_t* __restrict__ pupil,
    const creal32_t* __restrict__ subwave,
    const int2* __restrict__ ledindex,
    const dim3 imLs_sz,
    const dim3 imHs_sz,
    const float ratio,
    creal32_t* __restrict__ dldw1,
    creal32_t* __restrict__ dldw2
){
    const unsigned idy = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned idx = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned idz = blockIdx.z;

    if ((idx >= imLs_sz.x) || (idy >= imLs_sz.y) || (idz >= imLs_sz.z)) {
        return;
    }

    const unsigned pageSz = imLs_sz.x * imLs_sz.y;
    const unsigned idxZ = idz * pageSz;
    const unsigned pix_id = idx * imLs_sz.y + idy;

    // Inline the low-res fftshift on recordZ read.
    unsigned rec_x = idx + (imLs_sz.x >> 1);
    unsigned rec_y = idy + (imLs_sz.y >> 1);
    if (rec_x >= imLs_sz.x) rec_x -= imLs_sz.x;
    if (rec_y >= imLs_sz.y) rec_y -= imLs_sz.y;
    const unsigned rec_id = rec_x * imLs_sz.y + rec_y;

    const creal32_t Z = recordZ[rec_id + idxZ];
    const creal32_t P = pupil[pix_id];
    const creal32_t S = subwave[pix_id + idxZ];

    // recordZ * conj(pupil), keeping the current kernel's arithmetic.
    const float obj_re = (Z.re * P.re + Z.im * P.im) * ratio;
    const float obj_im = (Z.re * P.im - Z.im * P.re) * ratio;

    // recordZ * conj(subwave)
    const float pupil_re = Z.re * S.re + Z.im * S.im;
    const float pupil_im = Z.re * S.im - S.re * Z.im;

    float *dldw2_f = reinterpret_cast<float *>(dldw2);
    atomicAdd(dldw2_f + 2 * pix_id + 0, pupil_re);
    atomicAdd(dldw2_f + 2 * pix_id + 1, pupil_im);

    const unsigned pixL_x = (unsigned) ledindex[idz].x;
    const unsigned pixL_y = (unsigned) ledindex[idz].y;
    const unsigned large_idx = idx + pixL_x - 1;
    const unsigned large_idy = idy + pixL_y - 1;

    if ((large_idx < imHs_sz.x) && (large_idy < imHs_sz.y)) {
        // Inline the high-res fftshift on dldw1 writeback.
        unsigned spec_x = large_idx + (imHs_sz.x >> 1);
        unsigned spec_y = large_idy + (imHs_sz.y >> 1);
        if (spec_x >= imHs_sz.x) spec_x -= imHs_sz.x;
        if (spec_y >= imHs_sz.y) spec_y -= imHs_sz.y;

        const unsigned spec_id = spec_x * imHs_sz.y + spec_y;
        float *dldw1_f = reinterpret_cast<float *>(dldw1);
        atomicAdd(dldw1_f + spec_id * 2 + 0, obj_re);
        atomicAdd(dldw1_f + spec_id * 2 + 1, obj_im);
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
    const unsigned idy = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned idx = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned idz = blockIdx.z;

    const bool inside = (idx < imLs_sz.x) && (idy < imLs_sz.y) && (idz < imLs_sz.z);
    if (inside){
        const unsigned pageSz  = imLs_sz.x * imLs_sz.y;
        const unsigned pix_A   = idx * imLs_sz.y + idy;
        const unsigned page_id = idz * pageSz;

        creal32_t lat_A = latentz[pix_A + page_id];
        // Current phase
        float ang = atan2f(lat_A.im, lat_A.re);
        // Real scalar gradient from upstream
        const float tempA = dl_doutput[pix_A + page_id];
        // Map scalar gradient onto complex direction exp(j*ang)
        lat_A.re = cosf(ang) * tempA;
        lat_A.im = sinf(ang) * tempA;
        latentz[pix_A + page_id] = lat_A;
    }
}


__global__ void ifftCorrection_sub(
    creal32_t *__restrict__ latentZ,
    float* __restrict__ predY,
    const dim3 imgSz,
    const float ratio
){
    const unsigned idy = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned idx = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned idz = blockIdx.z;  

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y) && (idz < imgSz.z);
    if(!inside) return;
   
    const unsigned page_id = idz * (imgSz.x * imgSz.y);
    const unsigned pix_id = idx * imgSz.y + idy;

    creal32_t temp = latentZ[pix_id + page_id];
    // float ratio = 1.0f / (float) (imgSz.x * imgSz.y);
    temp.re = temp.re * ratio;
    temp.im = temp.im * ratio;
    latentZ[pix_id + page_id] = temp;
    predY[pix_id + page_id] = sqrtf(temp.re * temp.re + temp.im * temp.im);
}
