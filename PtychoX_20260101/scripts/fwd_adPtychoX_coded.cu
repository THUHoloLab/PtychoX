 /* =========================================================================
 Author: Shuhe Zhang
 Affiliation: Tsinghua University
 Email: shuhe-zhang@tsinghua.edu.cn

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 =========================================================================*/
 
#include "mex/mex.h"
#include "mex/mxGPUArray.h"
#include "kernels/kernels_coded.cuh"

/**
 * MEX entry: Coded ptychography forward model (GPU)
 * ============================================================================
 * This MEX function implements a CUDA-accelerated forward operator for a
 * coded-illumination ptychography model with multiple scan positions.
 *
 * Compared to the near-field version, this coded forward model additionally
 * applies a (possibly coded) transfer function H1(k) in Fourier domain BEFORE
 * returning to the spatial domain (i.e., before multiplying by the probe/pupil).
 *
 * High-level pipeline (conceptual)
 *   1) FFT2 of object field wavefront1: O(x,y) -> O(kx,ky)
 *   2) For each scan position (idz):
 *        - Apply scan-dependent phase ramp in Fourier domain:
 *            O_shift(k) = O(k) * exp(+j*2π*(kx*dx_idz + ky*dy_idz))
 *        - Apply coded transfer function / aperture / modulation H1(k):
 *            X_record(k,idz) = O_shift(k) .* H1(k)
 *        - IFFT2 to spatial domain: x_record(x,idz)
 *        - Multiply by probe/pupil (wavefront2):
 *            x_forward(x,idz) = x_record(x,idz) .* P(x)
 *        - ASP propagation (angular spectrum propagation) using H2(k):
 *            X = FFT2(x_forward),  X <- X .* H2,  x_forward <- IFFT2(X)
 *        - Downsample amplitude measurements by ds = 4:
 *            observeds = sqrt(mean(|x_forward|^2 over 4x4 blocks))
 *
 * Inputs (prhs)
 *   prhs[0] : wavefront1  (GPU complex single)  [Hx, Hy]    object field (spatial)
 *   prhs[1] : wavefront2  (GPU complex single)  [Hx, Hy]    coded surfacce in spatial domain
 *   prhs[2] : diffracH1   (GPU complex single)  [Hx, Hy]    propagation transfer function to the coded surface
 *   prhs[3] : diffracH2   (GPU complex single)  [Hx, Hy]    propagation transfer function to the camera sensor
 *   prhs[4] : shiftspox   (GPU float2)          [Z, ...]    scanning positions (dx,dy) per measurement
 *
 * Outputs (plhs)
 *   plhs[0] : X_record    (GPU complex single)  [Hx, Hy, Z] coded spectrum after shift (then IFFT'd in-place)
 *   plhs[1] : X_forward   (GPU complex single)  [Hx, Hy, Z] field after probe and propagation (spatial domain)
 *   plhs[2] : observeds   (GPU single real)     [Hx/4, Hy/4, Z] downsampled amplitude measurements
 *
 * Notes / caveats
 *   - cuFFT is unnormalized. Your kernels (e.g., fused_pixelwiseProduct*) apply
 *     1/(Hx*Hy) normalization at specific steps. Be consistent across forward/backward.
 *   - The cuFFT PlanMany embed parameters (inembed[1] = imHs_bc.x) look suspicious:
 *     for a 2D array it is usually {Hx, Hy}. Keep it if it matches your memory
 *     convention, but consider adding a comment or fixing if it is an error.
 *   - It is recommended to check nrhs >= 5 and nlhs >= 3 (not shown here).
 */
void mexFunction(
    int nlhs, mxArray *plhs[],
    int nrhs, mxArray const * __restrict__ prhs[]
){
    // Initialize MATLAB GPU API.
    mxInitGPU();

    // -------------------------------------------------------------------------
    // 1) Validate inputs
    // -------------------------------------------------------------------------
    CHECK_THROW(mxIsGPUArray(prhs[0]));  // wavefront1
    CHECK_THROW(mxIsGPUArray(prhs[1]));  // wavefront2
    CHECK_THROW(mxIsGPUArray(prhs[2]));  // diffracH1
    CHECK_THROW(mxIsGPUArray(prhs[3]));  // diffracH2
    CHECK_THROW(mxIsGPUArray(prhs[4]));  // shiftspox

    // -------------------------------------------------------------------------
    // 2) Wrap GPU inputs as mxGPUArray handles
    // -------------------------------------------------------------------------
    // These are read-only views (no copy).
    CST_GPU_PTR wavefront2 = mxGPUCreateFromMxArray(prhs[1]);
    CST_GPU_PTR diffracH1  = mxGPUCreateFromMxArray(prhs[2]);
    CST_GPU_PTR diffracH2  = mxGPUCreateFromMxArray(prhs[3]);
    CST_GPU_PTR shiftspox  = mxGPUCreateFromMxArray(prhs[4]);

    // wavefront1 is copied because we overwrite it in-place (FFT).
    mxGPUArray_t * wavefront1 = mxGPUCopyFromMxArray(prhs[0]);

    // -------------------------------------------------------------------------
    // 3) Infer tensor sizes and set batch dimensions
    // -------------------------------------------------------------------------
    // imLs_sz is used primarily to determine the number of scan positions Z.
    dim3 imLs_sz = size2dim3(shiftspox);

    // imHs_sz is the high-resolution spatial grid size.
    dim3 imHs_sz = size2dim3(wavefront1);

    // Batched high-res: [Hx, Hy, Z]
    dim3 imHs_bc = {imHs_sz.x, imHs_sz.y, imLs_sz.x};

    // Fixed downsampling factor for coded ptychography measurements: ds = 4
    dim3 imLs_bc = {imHs_sz.x / 4, imHs_sz.y / 4, imLs_sz.x};

    // MATLAB size arrays for output allocations
    const mwSize Hsz[3] = {imHs_sz.x, imHs_sz.y, imLs_sz.x};
    const mwSize Lsz[3] = {imHs_sz.x / 4, imHs_sz.y / 4, imLs_sz.x};

    // -------------------------------------------------------------------------
    // 4) Allocate GPU outputs
    // -------------------------------------------------------------------------
    // X_record: intermediate per-scan complex buffer (same shape as high-res batch)
    mxGPUArray_t * X_record = mxGPUCreateGPUArray(
        3, Hsz,
        mxSINGLE_CLASS, mxCOMPLEX,
        MX_GPU_DO_NOT_INITIALIZE
    );

    // X_forward: final per-scan propagated complex field
    mxGPUArray_t * X_forward = mxGPUCreateGPUArray(
        3, Hsz,
        mxSINGLE_CLASS, mxCOMPLEX,
        MX_GPU_DO_NOT_INITIALIZE
    );

    // observeds: downsampled real-valued amplitude measurements
    mxGPUArray_t * observeds = mxGPUCreateGPUArray(
        3, Lsz,
        mxSINGLE_CLASS, mxREAL,
        MX_GPU_DO_NOT_INITIALIZE
    );

    // -------------------------------------------------------------------------
    // 5) Extract raw device pointers
    // -------------------------------------------------------------------------
    const creal32_t * v_wavefront2 = getGPUDataRO<creal32_t>(wavefront2);
    const creal32_t * v_diffracH1  = getGPUDataRO<creal32_t>(diffracH1);
    const creal32_t * v_diffracH2  = getGPUDataRO<creal32_t>(diffracH2);
    const float2    * v_shiftspox  = getGPUDataRO<float2>(shiftspox);

    creal32_t * v_wavefront1 = (creal32_t *) mxGPUGetData(wavefront1); // copied object field
    creal32_t * v_X_record   = (creal32_t *) mxGPUGetData(X_record);
    creal32_t * v_X_forward  = (creal32_t *) mxGPUGetData(X_forward);
    float     * v_observeds  = (float *)     mxGPUGetData(observeds);

    // -------------------------------------------------------------------------
    // 6) Configure CUDA kernel launch parameters
    // -------------------------------------------------------------------------
    dim3 N_THREADS = {BLOCK_X, BLOCK_Y, 1};
    dim3 N_BLOCKS  = {
        (unsigned)((imHs_sz.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned)((imHs_sz.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) imHs_bc.z   // batch dimension Z
    };

    // -------------------------------------------------------------------------
    // 7) Create cuFFT plans
    // -------------------------------------------------------------------------
    // plan: FFT2 of a single high-res page (object field).
    cufftHandle plan;
    cufftPlan2d(&plan, imHs_sz.x, imHs_sz.y, CUFFT_C2C);

    // many_plan: batched FFT2 for tensors shaped [Hx, Hy, Z].
    cufftHandle many_plan;

    // cuFFT embed parameters (physical leading dimensions).
    // NOTE: inembed[1] = imHs_bc.x is potentially a typo; commonly it should be imHs_bc.y.
    int inembed[2];
    inembed[0] = (int) imHs_bc.x;
    inembed[1] = (int) imHs_bc.x; // <-- consider verifying / documenting this choice

    cufftPlanMany(
        &many_plan, 2,
        &inembed[0],
        &inembed[0], 1,  imHs_bc.x * imHs_bc.y, // istride, idist
        &inembed[0], 1,  imHs_bc.x * imHs_bc.y, // ostride, odist
        CUFFT_C2C, imHs_bc.z
    );

    // -------------------------------------------------------------------------
    // 8) Forward model execution
    // -------------------------------------------------------------------------

    // 8.1) FFT2 of wavefront1 (in-place)
    cufftExecC2C(
        plan,
        (cufftComplex *)&v_wavefront1[0],
        (cufftComplex *)&v_wavefront1[0],
        CUFFT_FORWARD
    );
    cufftDestroy(plan);

    // (Optional) FFT-shift the spectrum (disabled)
    // cufftShift_2D_kernel<<<N_BLOCKS, N_THREADS>>>(v_wavefront1, imHs_sz.x);

    // 8.2) Apply scan-dependent Fourier shift AND coded transfer function H1(k)
    //      X_record(k,idz) = wavefront1(k) * exp(+j*2π*k·pos_idz) * H1(k)
    fullyfused_shiftNprop<<<N_BLOCKS, N_THREADS>>>(
        v_wavefront1,   // object spectrum (shared across all idz)
        v_diffracH1,    // coded transfer function in Fourier domain
        v_shiftspox,    // scanning positions (dx,dy) for each idz
        imHs_bc,        // [Hx, Hy, Z]
        v_X_record      // output: per-scan spectrum buffer
    );

    // 8.3) IFFT2 per scan (batched): spectrum -> spatial domain
    cufftExecC2C(
        many_plan,
        (cufftComplex *)&v_X_record[0],
        (cufftComplex *)&v_X_record[0],
        CUFFT_INVERSE
    );

    // 8.4) Multiply by probe/pupil in spatial domain:
    //      X_forward = (X_record / (Hx*Hy)) .* wavefront2
    fused_pixelwiseProduct<<<N_BLOCKS, N_THREADS>>>(
        v_X_record,
        v_wavefront2,
        imHs_bc,
        v_X_forward
    );

    // 8.5) Angular spectrum propagation using H2(k)
    //      X_forward <- IFFT( FFT(X_forward) .* H2(k) )
    {
        // FFT2 batched
        cufftExecC2C(
            many_plan,
            (cufftComplex *)&v_X_forward[0],
            (cufftComplex *)&v_X_forward[0],
            CUFFT_FORWARD
        );

        // Multiply by propagation kernel (and normalize inside kernel)
        fused_pixelwiseProduct_inplace<<<N_BLOCKS, N_THREADS>>>(
            v_diffracH2,   // propagation transfer function in Fourier domain
            imHs_bc,
            v_X_forward
        );

        // IFFT2 batched back to spatial domain
        cufftExecC2C(
            many_plan,
            (cufftComplex *)&v_X_forward[0],
            (cufftComplex *)&v_X_forward[0],
            CUFFT_INVERSE
        );
    }
    cufftDestroy(many_plan);

    // 8.6) Downsample the propagated field magnitude to detector resolution.
    //      Here the downsampling factor is fixed to ds = 4.
    fullyfused_DownSample_Fwd<<<N_BLOCKS, N_THREADS>>>(
        v_X_forward,
        (int)4,         // ds = 4 (hard-coded)
        imLs_bc,        // output size [Hx/4, Hy/4, Z]
        v_observeds
    );

    // -------------------------------------------------------------------------
    // 9) Return outputs to MATLAB (still on GPU)
    // -------------------------------------------------------------------------
    plhs[0] = mxGPUCreateMxArrayOnGPU(X_record);
    plhs[1] = mxGPUCreateMxArrayOnGPU(X_forward);
    plhs[2] = mxGPUCreateMxArrayOnGPU(observeds);

    // -------------------------------------------------------------------------
    // 10) Cleanup (prevent GPU memory leaks)
    // -------------------------------------------------------------------------
    // Destroy output handles (MATLAB owns the returned plhs arrays now)
    mxGPUDestroyGPUArray(X_record);
    mxGPUDestroyGPUArray(X_forward);
    mxGPUDestroyGPUArray(observeds);

    // Destroy input handles and copied arrays
    mxGPUDestroyGPUArray(wavefront1);  // copied buffer
    mxGPUDestroyGPUArray(wavefront2);
    mxGPUDestroyGPUArray(diffracH1);
    mxGPUDestroyGPUArray(diffracH2);
    mxGPUDestroyGPUArray(shiftspox);
}