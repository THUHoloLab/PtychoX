/**
 * MEX entry: Near-field ptychography forward model (GPU)
 * ============================================================================
 * This MEX function implements a CUDA-accelerated forward operator for a
 * near-field / Fresnel-like ptychographic model with multiple scan positions.
 *
 * High-level pipeline (conceptual)
 *   1) FFT2 of the object spectrum (wavefront1): O(x,y) -> O(kx,ky)
 *   2) For each scan position (idz):
 *        - Apply a scan-dependent phase ramp in Fourier domain:
 *            O_shift(k) = O(k) * exp(+j*2π*(kx*dx_idz + ky*dy_idz))
 *        - IFFT2 to spatial domain: o_shift(x)
 *        - Multiply by pupil / probe (wavefront2): u(x) = o_shift(x) * P(x)
 *        - (Optional) ASP propagation (angular spectrum propagation):
 *            U(k) = FFT2(u(x))
 *            U_prop(k) = U(k) * H2(k)
 *            u_prop(x) = IFFT2(U_prop(k))
 *        - Downsample amplitude to detector resolution (RMS magnitude)
 *
 * Inputs (prhs)
 *   prhs[0] : wavefront1  (GPU complex single)  [Hx, Hy]    object field (spatial)
 *   prhs[1] : wavefront2  (GPU complex single)  [Hx, Hy]    pupil/probe in spatial domain
 *   prhs[2] : diffracH2   (GPU complex single)  [Hx, Hy]    propagation transfer function (Fourier domain)
 *   prhs[3] : shiftspox   (GPU float2)          [Z, ...]    scanning positions (dx,dy) for each frame
 *   prhs[4] : downSam_ratio (CPU scalar)        int         downsampling factor ds
 *
 * Outputs (plhs)
 *   plhs[0] : X_record    (GPU complex single)  [Hx, Hy, Z] shifted object field (after shift & IFFT)
 *   plhs[1] : X_forward   (GPU complex single)  [Hx, Hy, Z] field after pupil and propagation (near-field)
 *   plhs[2] : observeds   (GPU single real)     [Hx/ds, Hy/ds, Z] downsampled amplitude measurements
 *
 * Notes
 *   - This code assumes cuFFT is unnormalized; normalization is handled in
 *     fused kernels (e.g., fused_pixelwiseProduct scales by 1/(Hx*Hy)).
 *   - Many kernels assume BLOCK_SIZE == BLOCK_X * BLOCK_Y and contiguous memory.
 *   - Proper CUDA error checking (cudaGetLastError/cudaDeviceSynchronize)
 *     is recommended after kernel launches and cuFFT calls for debugging.
 */
void mexFunction(
    int nlhs, mxArray *plhs[],
    int nrhs, mxArray const * __restrict__ prhs[]
){
    // Initialize MATLAB GPU API. Must be called before any mxGPUArray usage.
    mxInitGPU();

    // -------------------------------------------------------------------------
    // 1) Validate inputs (GPU arrays and scalar)
    // -------------------------------------------------------------------------
    CHECK_THROW(mxIsGPUArray(prhs[0]));  // wavefront1
    CHECK_THROW(mxIsGPUArray(prhs[1]));  // wavefront2
    CHECK_THROW(mxIsGPUArray(prhs[2]));  // diffracH2
    CHECK_THROW(mxIsGPUArray(prhs[3]));  // shiftspox
    // prhs[4] is a CPU scalar (downSam_ratio), so no mxIsGPUArray check here.

    // -------------------------------------------------------------------------
    // 2) Wrap inputs as mxGPUArray handles
    // -------------------------------------------------------------------------
    // Read-only wrappers (no copy) for inputs that will not be modified.
    CST_GPU_PTR wavefront2 = mxGPUCreateFromMxArray(prhs[1]);
    CST_GPU_PTR diffracH2  = mxGPUCreateFromMxArray(prhs[2]);
    CST_GPU_PTR shiftspox  = mxGPUCreateFromMxArray(prhs[3]);

    // wavefront1 is copied because we will overwrite it in-place with FFT results.
    mxGPUArray_t * wavefront1 = mxGPUCopyFromMxArray(prhs[0]);

    // Downsampling ratio ds (detector pixel aggregation factor)
    int downSam_ratio = (int) mxGetScalar(prhs[4]);

    // -------------------------------------------------------------------------
    // 3) Infer tensor sizes and batch sizes
    // -------------------------------------------------------------------------
    // imLs_sz: size of scanning positions array (used for number of frames Z)
    dim3 imLs_sz = size2dim3(shiftspox);

    // imHs_sz: size of wavefront1 (high-resolution grid Hx x Hy)
    dim3 imHs_sz = size2dim3(wavefront1);

    // Batched shapes:
    //   imHs_bc: [Hx, Hy, Z]  high-res batch for each scan position
    //   imLs_bc: [Hx/ds, Hy/ds, Z] low-res (detector) batch
    dim3 imHs_bc = {imHs_sz.x, imHs_sz.y, imLs_sz.x};
    dim3 imLs_bc = {imHs_sz.x / downSam_ratio,
                    imHs_sz.y / downSam_ratio,
                    imLs_sz.x};

    // MATLAB dimension arrays for output allocations
    const mwSize Hsz[3] = {imHs_sz.x, imHs_sz.y, imLs_sz.x};
    const mwSize Lsz[3] = {imHs_sz.x / downSam_ratio,
                           imHs_sz.y / downSam_ratio,
                           imLs_sz.x};

    // -------------------------------------------------------------------------
    // 4) Allocate GPU outputs
    // -------------------------------------------------------------------------
    // X_record: shifted-and-IFFT field per scan (complex)
    mxGPUArray_t * X_record = mxGPUCreateGPUArray(
        3, Hsz,
        mxSINGLE_CLASS, mxCOMPLEX,
        MX_GPU_DO_NOT_INITIALIZE   // faster; memory is uninitialized
    );

    // X_forward: final propagated field per scan (complex)
    mxGPUArray_t * X_forward = mxGPUCreateGPUArray(
        3, Hsz,
        mxSINGLE_CLASS, mxCOMPLEX,
        MX_GPU_DO_NOT_INITIALIZE
    );

    // observeds: downsampled amplitude measurements (real)
    mxGPUArray_t * observeds = mxGPUCreateGPUArray(
        3, Lsz,
        mxSINGLE_CLASS, mxREAL,
        MX_GPU_DO_NOT_INITIALIZE
    );

    // -------------------------------------------------------------------------
    // 5) Extract raw device pointers
    // -------------------------------------------------------------------------
    // Read-only device pointers (const)
    const creal32_t * v_wavefront2 = getGPUDataRO<creal32_t>(wavefront2);
    const creal32_t * v_diffracH2  = getGPUDataRO<creal32_t>(diffracH2);
    const float2    * v_shiftspox  = getGPUDataRO<float2>(shiftspox);

    // Writable pointers
    creal32_t * v_wavefront1 = (creal32_t *) mxGPUGetData(wavefront1); // copied buffer
    creal32_t * v_X_record   = (creal32_t *) mxGPUGetData(X_record);
    creal32_t * v_X_forward  = (creal32_t *) mxGPUGetData(X_forward);
    float     * v_observeds  = (float *)     mxGPUGetData(observeds);

    // -------------------------------------------------------------------------
    // 6) Configure CUDA launch geometry
    // -------------------------------------------------------------------------
    dim3 N_THREADS = {BLOCK_X, BLOCK_Y, 1};

    // Blocks cover the high-res grid (Hx, Hy) and batch dimension Z
    dim3 N_BLOCKS = {
        (unsigned)((imHs_sz.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned)((imHs_sz.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) imHs_bc.z
    };

    // -------------------------------------------------------------------------
    // 7) Create cuFFT plans
    // -------------------------------------------------------------------------
    // plan: a single 2D FFT plan for wavefront1 (one page)
    cufftHandle plan;
    cufftPlan2d(&plan, imHs_sz.x, imHs_sz.y, CUFFT_C2C);

    // many_plan: batched 2D FFT plan for tensors shaped [Hx, Hy, Z]
    cufftHandle many_plan;

    // cuFFT "embed" describes physical in-memory leading dimensions.
    // WARNING: inembed[1] is set to imHs_bc.x in your code. Typically for a 2D
    // array you would use {Hx, Hy}. If your memory is laid out as [Hx x Hy],
    // inembed should usually be {Hx, Hy}. Keep this as-is if it matches your
    // project’s convention; otherwise it is a potential bug.
    int inembed[2];
    inembed[0] = (int) imHs_bc.x;
    inembed[1] = (int) imHs_bc.x; // <-- check if this should be imHs_bc.y

    cufftPlanMany(
        &many_plan, 2,
        &inembed[0],
        &inembed[0], 1,  imHs_bc.x * imHs_bc.y, // istride, idist
        &inembed[0], 1,  imHs_bc.x * imHs_bc.y, // ostride, odist
        CUFFT_C2C, imHs_bc.z
    );

    // -------------------------------------------------------------------------
    // 8) Forward pipeline execution
    // -------------------------------------------------------------------------

    // 8.1) FFT2 of wavefront1 (in-place, single page)
    cufftExecC2C(
        plan,
        (cufftComplex *)&v_wavefront1[0],
        (cufftComplex *)&v_wavefront1[0],
        CUFFT_FORWARD
    );
    cufftDestroy(plan);

    // (Optional) FFT-shift the spectrum (disabled)
    // cufftShift_2D_kernel<<<N_BLOCKS, N_THREADS>>>(v_wavefront1, imHs_sz.x);

    // 8.2) Apply scan-dependent Fourier shift and replicate into X_record pages
    //      X_record(:,:,idz) = wavefront1(:,:) * exp(+j*2π*(kx*dx+ky*dy))
    fullyfused_shiftNprop<<<N_BLOCKS, N_THREADS>>>(
        v_wavefront1,   // global spectrum (shared)
        v_shiftspox,    // scanning positions (dx,dy) per idz
        imHs_bc,        // [Hx, Hy, Z]
        v_X_record      // output: shifted spectrum per idz
    );

    // 8.3) IFFT2 for each page (batched) to return to spatial domain
    cufftExecC2C(
        many_plan,
        (cufftComplex *)&v_X_record[0],
        (cufftComplex *)&v_X_record[0],
        CUFFT_INVERSE
    );

    // 8.4) Multiply by pupil/probe in spatial domain:
    //      X_forward = (X_record / (Hx*Hy)) .* wavefront2
    //      (kernel performs scaling + multiply)
    fused_pixelwiseProduct<<<N_BLOCKS, N_THREADS>>>(
        v_X_record,
        v_wavefront2,
        imHs_bc,
        v_X_forward
    );

    // 8.5) Angular spectrum propagation (ASP) in Fourier domain (optional step)
    //      X_forward <- IFFT( FFT(X_forward) .* diffracH2 )
    {
        // FFT2 batched
        cufftExecC2C(
            many_plan,
            (cufftComplex *)&v_X_forward[0],
            (cufftComplex *)&v_X_forward[0],
            CUFFT_FORWARD
        );

        // Multiply by propagation transfer function H2(k) and normalize:
        // X_forward <- (X_forward .* diffracH2) / (Hx*Hy)
        fused_pixelwiseProduct_inplace<<<N_BLOCKS, N_THREADS>>>(
            v_diffracH2,   // propagation kernel in Fourier domain
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

    // Destroy batched FFT plan after all FFTs are finished
    cufftDestroy(many_plan);

    // 8.6) Downsample the propagated complex field to detector resolution
    //      observeds = sqrt(mean(|X_forward|^2 over ds x ds blocks))
    fullyfused_DownSample_Fwd<<<N_BLOCKS, N_THREADS>>>(
        v_X_forward,
        (int) downSam_ratio,
        imLs_bc,       // NOTE: imgSz here is the *small* output size [Hx/ds, Hy/ds, Z]
        v_observeds
    );

    // -------------------------------------------------------------------------
    // 9) Return GPU outputs to MATLAB
    // -------------------------------------------------------------------------
    plhs[0] = mxGPUCreateMxArrayOnGPU(X_record);
    plhs[1] = mxGPUCreateMxArrayOnGPU(X_forward);
    plhs[2] = mxGPUCreateMxArrayOnGPU(observeds);

    // -------------------------------------------------------------------------
    // 10) Cleanup GPU arrays (prevent memory leaks)
    // -------------------------------------------------------------------------
    // Destroy output mxGPUArray handles (MATLAB now owns plhs copies)
    mxGPUDestroyGPUArray(X_record);
    mxGPUDestroyGPUArray(X_forward);
    mxGPUDestroyGPUArray(observeds);

    // Destroy input wrappers / copied arrays
    mxGPUDestroyGPUArray(wavefront1);   // copied buffer
    mxGPUDestroyGPUArray(wavefront2);
    mxGPUDestroyGPUArray(diffracH2);
    mxGPUDestroyGPUArray(shiftspox);
}