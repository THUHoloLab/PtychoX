clc
clear
reset(gpuDevice());            % Reset GPU state (clear memory, reset kernels/context)

pix = 256;                     % Low-res measurement size (detector patch size)
mag = 4;                       % Magnification factor (object grid = pix*mag)

%% ------------------------------------------------------------------------
% 1) Prepare a synthetic ptychographic dataset
%    - Create a complex-valued object U0 (amplitude + phase)
%    - Generate a probe/illumination function 'prop'
%    - Simulate diffraction measurements for random scan positions
% -------------------------------------------------------------------------

% Load two images to synthesize amplitude and phase
img1 = single(imread('source/cameraman512.png'))/255;   % Used to build amplitude
img2 = mean(single(imread('source/I01.bmp'))/255,3);    % Used to build phase (RGB -> grayscale)

% Resize to the high-resolution object grid
img1 = imresize(img1,[pix * mag, pix * mag]);
img2 = imresize(img2,[pix * mag, pix * mag]);

% Construct complex object field U0:
%   amplitude: mat2gray(img1 + 0.2) (avoid too dark amplitude)
%   phase:     exp(1i*pi*img2)      (phase in [0, pi])
U0 = mat2gray(img1 + 0.2) .* exp(1i*pi*img2);
U0 = gpuArray(single(U0));         % Move object to GPU (single precision)

rng(12);                           % Fix random seed for reproducibility

% Define scan boundary so that cropped patches stay inside U0
boundary_left  = round(pix/7) + 1;
boundary_right = pix * mag - pix - boundary_left;

sample = 300;                      % Number of diffraction measurements (scan positions)
% Random scan positions in [boundary_left, boundary_right]
% xy is 2 x sample, where xy(1,:) is x, xy(2,:) is y
xy = round((boundary_right - boundary_left) * rand(2,sample)) + boundary_left;

%% ------------------------------------------------------------------------
% 2) Build a probe 'prop' using an Angular Spectrum Propagation-like recipe
%    This is just for synthetic data generation (can be any probe).
% -------------------------------------------------------------------------

len = 100;                         % Frequency scaling / sampling parameter
fx = (-pix/2:pix/2-1)/len;         % Frequency coordinates (normalized)
[fx,fy] = meshgrid(fx);

lambda = 0.532;                    % Wavelength (same units as propagation distance)
k = 2*pi/lambda;                   % Wavenumber

% Angular spectrum propagation kernel component:
% fz = sqrt(1 - (lambda*fx)^2 - (lambda*fy)^2)
% (valid where inside the propagating region)
fz = sqrt(1 - lambda^2 * (fx.^2 + fy.^2));

% Build an aperture mask U (a circular pupil in frequency/spatial domain proxy)
U = (fx.^2 + fy.^2) < (1/len * 60).^2;

% Create a random low-res pattern then upsample to 'U' size
% (box interpolation makes a smooth-ish random pattern)
p0 = imresize(gpuArray.rand(32,'single'), size(U), 'box');

% Build probe via:
%   1) FFT2 of (U .* p0)
%   2) Multiply by propagation phase exp(i*k*z*fz)
%   3) IFFT2 back to spatial domain -> prop
prop = ifftshift(fftshift(fft2(U .* p0)) .* exp(1i * k * 160 * fz));
prop = ifft2(prop);
prop = gpuArray(single(prop));     % Ensure GPU single complex

% Threshold small values (probe support / sparsify)
prop = prop .* (abs(prop) > 0.1);

% Visualize probe amplitude
figure(); imshow(abs(prop), []);

%% ------------------------------------------------------------------------
% 3) Forward simulation: generate diffraction magnitudes for each scan
%    Measurement model (conventional ptychography):
%      y = | FFT2( sub_object .* probe ) |
% -------------------------------------------------------------------------

img_data = gpuArray(zeros(pix, pix, sample, 'single')); % Measurements: [pix, pix, sample]

for con = 1:sample
    con                                % Print current index (progress)
    this_pos = xy(:,con);              % [x; y] position for this measurement

    % Extract a pix-by-pix patch from the high-res object U0 at scan position
    % Note: MATLAB indexing is (row, col) => (y, x)
    sub = U0(this_pos(2,1):this_pos(2,1)+pix-1, ...
             this_pos(1,1):this_pos(1,1)+pix-1);

    % Diffraction magnitude (Fourier magnitude of exit wave)
    temp_img = abs(fftshift(fft2(sub .* prop)));

    % Add optional noise (currently scaled by 0 so it is disabled)
    img_data(:,:,con) = temp_img + 0*max(temp_img(:)) .* gpuArray.rand(pix,'single');
end

%% ------------------------------------------------------------------------
% 4) Inverse problem: reconstruct object (wavefront1) and probe (wavefront2)
%    using a differentiable forward operator + Adam optimization.
% -------------------------------------------------------------------------

img_data = gpuArray(img_data);        % Ensure measurements on GPU

% Initialize unknowns:
% wavefront1: reconstructed object on high-res grid (pix*mag x pix*mag)
% wavefront2: reconstructed probe/pupil (here initialized as mask U)
wavefront1 = rand(pix*mag, 'single');
wavefront2 = single(U);

wavefront1 = gpuArray(complex(wavefront1));
wavefront2 = gpuArray(complex(wavefront2));

% Wrap in dlarray for automatic differentiation
wavefront1 = dlarray(wavefront1);
wavefront2 = dlarray(wavefront2);

batchSize = 32;                      % Mini-batch size (number of positions per step)

% Adam optimizers for object and probe
optimizer1 = optimizers.Adam(0.9, 0.999, 1e-16);
optimizer2 = optimizers.Adam(0.9, 0.999, 1e-16);

lr = 0.08;                           % Learning rate
iteration = 0;

enable_cuda = true;                  % Use CUDA-accelerated forward/backward in adPtychoX
adfoo = adPtychoX_base(enable_cuda); % Differentiable ptychography operator (custom)

error_data = [];
epoch_tic = tic;

frame = 0;                           % GIF frame counter for visualization
takes_time = 0;                      % Accumulate iteration time
stops = false;                       % (Unused) flag

%% ------------------------------------------------------------------------
% 5) Training loop (epochs + batches)
% -------------------------------------------------------------------------
for epoch = 1:600
    listBlock = 1:sample;            % Indices of all measurements
    loss_epoch = 0;

    while ~isempty(listBlock)
        start_tic = tic; 
        iteration = iteration + 1;

        % Build a mini-batch index list
        len = length(listBlock);
        batchIdx = listBlock(1:min(batchSize, len));
        listBlock(1:min(batchSize, len)) = [];

        % Extract scan positions for this batch (2 x B)
        pos = single(gpuArray(xy(:,batchIdx)));

        % Evaluate loss and gradients using dlfeval (enables AD)
        [loss, dldw1, dldw2] = dlfeval(@model_loss, adfoo, ...
                                       wavefront1, ...
                                       wavefront2, ...
                                       pos, ...
                                       img_data(:,:,batchIdx));

        % Adam update: object and probe
        wavefront1 = optimizer1.step(wavefront1, dldw1, iteration, lr);
        wavefront2 = optimizer2.step(wavefront2, dldw2, iteration, lr);

        
        loss_epoch = loss_epoch + extractdata(loss);

        % Optional constraint/projection on object amplitude:
        % clamp |wavefront1| <= 1 while keeping its "sign/phase"
        % (Note: for complex numbers, sign() is not standard; see your custom sign definition if any)
        wavefront1 = min(abs(wavefront1), 1) .* sign(wavefront1);
        wait(gpuDevice());
        takes_time = takes_time + toc(start_tic); % accumulate time of this iteration
        %% ----------------------------------------------------------------
        % Visualization / logging every 5 iterations
        % ----------------------------------------------------------------
        if mod(iteration, 5) == 0
            % Fetch current estimates to CPU for display
            w1 = extractdata(wavefront1);
            w2 = extractdata(wavefront2);

            figure(121);

            % Display object amplitude and phase side-by-side
            img_show = [abs(w1), mat2gray(angle(w1))];
            img_show = imresize(img_show, 0.6);

            % Display probe amplitude and phase side-by-side
            probe_show = [mat2gray(abs(w2)), mat2gray(angle(w2))];
            probe_show = imresize(probe_show, size(img_show));

            % Concatenate object/probe displays
            img_show = [img_show, probe_show];

            frame = frame + 1;

            % Overlay runtime text and write GIF frames
            textStr = sprintf('PtychoX, takes %.2f s', takes_time);
            frame_uint8 = uint8(255 * img_show);

            frame_rgb = insertText(gather(frame_uint8), ...
                [5, 5], textStr, ...          % Top-left position [x, y]
                'FontSize', 48, ...
                'TextColor', 'white', ...
                'BoxColor', 'black', ...
                'BoxOpacity', 0.6);
            frame_rgb = imresize(frame_rgb,0.3);
            % frame_gray = rgb2gray(frame_rgb);

            imshow(frame_rgb, []);
            drawnow;
            [map,idx] = rgb2ind(frame_rgb,8);
            % Write or append to GIF
            if frame == 1
                imwrite(map,idx, 'test.gif', 'gif', ...
                        'LoopCount', 65535, ...
                        'DelayTime', 0.03);
            elseif (frame > 1) && (frame < 120)
                imwrite(map,idx, 'test.gif', 'gif', ...
                        'WriteMode', 'append', ...
                        'DelayTime', 0.03);
            end
        end

        % Learning rate decay after 500 iterations
        if iteration > 500
            lr = lr * 0.5;
        end

        % Stop condition for visualization length
        if frame > 120
            break
        end
    end

    if frame > 120
        break
    end

    % Optional: compute error to ground truth (disabled)
    % mse = mean(abs((U0 - extractdata(wavefront1))).^2,'all');
    % error_data = [error_data, loss_epoch];
end

toc(epoch_tic)

%% ------------------------------------------------------------------------
% Helper function: compute loss and gradients for one mini-batch
% -------------------------------------------------------------------------
function [loss, dldw1, dldw2] = model_loss(adfunc, wave1, wave2, position, obseY)
    % Wrap measurements into dlarray (for compatibility with deep learning ops)
    obseY = dlarray(obseY);

    % Forward model prediction
    predY = adfunc(wave1, wave2, position);

    % L2 loss (sum/mean depends on DataFormat and l2loss implementation)
    % Here DataFormat="SSB" indicates (Spatial, Spatial, Batch)
    loss = l2loss(predY, obseY, "DataFormat", "SSB");

    % Compute gradients w.r.t. wave1 (object) and wave2 (probe)
    [dldw1, dldw2] = dlgradient(loss, wave1, wave2);
end
