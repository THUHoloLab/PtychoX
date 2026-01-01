% =========================================================================
% adPtycho.m
%
% Differentiable ptychographic forward/backward operator
% implemented as a custom deep learning operation.
%
% This class defines both the forward diffraction model and
% its analytic gradient for use in automatic differentiation.
%
% Author: Shuhe Zhang
% Affiliation: Tsinghua University
% Email: shuhe-zhang@tsinghua.edu.cn
%
% Licensed under the Apache License, Version 2.0
% =========================================================================

classdef adPtycho < deep.DifferentiableFunction
    properties
        % Flag to enable CUDA-accelerated implementation
        enable_cuda = true;
    end

    methods
        % -----------------------------------------------------------------
        % Constructor
        % -----------------------------------------------------------------
        function self = adPtycho(enable_cuda)

            % Number of outputs of the forward function
            % (required by DifferentiableFunction)
            output_num = 1; % IMPORTANT: number of primary outputs

            % Call superclass constructor
            self@deep.DifferentiableFunction( ...
                output_num, ...
                SaveInputsForBackward  = true, ... % save inputs for backprop
                SaveOutputsForBackward = false, ...
                NumMemoryValues        = 3 );      % memo1, memo2, cuffthandle

            % Enable or disable CUDA path
            self.enable_cuda = enable_cuda;
        end

        % -----------------------------------------------------------------
        % Forward function
        %
        % Inputs:
        %   wavefront1 : complex object/sample estimate (2D)
        %   wavefront2 : complex probe function (2D)
        %   pos        : scan positions [2 × N]
        %
        % Outputs:
        %   observe    : diffraction amplitude |FFT(object × probe)|
        %   memo1      : extracted object patches
        %   memo2      : exit waves in Fourier domain
        %   cuffthandle: CUDA FFT plan handle (for reuse in backward)
        % -----------------------------------------------------------------
        function [observe, memo1, memo2, cuffthandle] = forward(self, ...
                                                wavefront1, ...
                                                wavefront2, ...
                                                pos)

            if self.enable_cuda
                % ---------------------------------------------------------
                % CUDA-accelerated forward model
                %
                % cuPtycho_Fwd performs:
                %   1) patch extraction from object
                %   2) multiplication with probe
                %   3) FFT and fftshift
                %   4) amplitude computation
                % ---------------------------------------------------------
                [memo1, ...
                 memo2, ...
                 observe, ...
                 cuffthandle] = adPtychoX_base_Fwd( ...
                                                    wavefront1, ...
                                                    wavefront2, ...
                                                    int32(pos)...
                                                  );
            else
                % ---------------------------------------------------------
                % MATLAB GPU fallback implementation
                % ---------------------------------------------------------

                % No FFT plan needed in MATLAB path
                cuffthandle = [];

                % Preallocate object patches:
                % size: [probeH × probeW × numScan]
                memo1 = gpuArray.zeros( ...
                    size(wavefront2,1), ...
                    size(wavefront2,2), ...
                    size(pos,2), 'single');

                % Extract object patches according to scan positions
                for con = 1:size(pos,2)
                    memo1(:,:,con) = wavefront1( ...
                        pos(2,con):pos(2,con)+size(wavefront2,1)-1, ...
                        pos(1,con):pos(1,con)+size(wavefront2,2)-1 );
                end

                % Exit wave in real space
                memo2 = memo1 .* wavefront2;

                % Forward FFT to detector plane
                memo2 = fftshift(fftshift(fft2(memo2),1),2);

                % Measured amplitude
                observe = abs(memo2);
            end
        end

        % -----------------------------------------------------------------
        % Backward function (gradient computation)
        %
        % Inputs:
        %   dl_dout    : gradient of loss w.r.t. output amplitude
        %   wavefront1 : object estimate
        %   wavefront2 : probe estimate
        %   pos        : scan positions
        %   memo1      : saved object patches
        %   memo2      : saved Fourier-domain exit waves
        %   cuffthandle: CUDA FFT plan handle
        %
        % Outputs:
        %   dldw1      : gradient w.r.t. object
        %   dldw2      : gradient w.r.t. probe
        %   dldps      : gradient w.r.t. positions (not implemented)
        % -----------------------------------------------------------------
        function [dldw1, ...
                  dldw2, ...
                  dldps] = backward(self, ...
                                    dl_dout, ...
                                    ~, ...
                                    wavefront1, ...
                                    wavefront2, ...
                                    pos, ...
                                    memo1, ...
                                    memo2, ...
                                    cuffthandle)

            % Position gradient is not supported (fixed scan geometry)
            dldps = [];

            if self.enable_cuda
                % ---------------------------------------------------------
                % CUDA-accelerated backward pass
                %
                % Computes analytic gradients:
                %   ∂L/∂object, ∂L/∂probe
                % ---------------------------------------------------------
                [dldw1, dldw2] = adPtychoX_base_Bwd( ...
                                            dl_dout, ...
                                            wavefront1, ...
                                            wavefront2, ...
                                            int32(pos), ...
                                            memo2, ...
                                            memo1, ...
                                            cuffthandle );
            else
                % ---------------------------------------------------------
                % MATLAB GPU fallback implementation
                % ---------------------------------------------------------

                % Gradient through magnitude operator:
                % d|z|/dz = sign(z)
                x = dl_dout .* sign(memo2);

                % Backpropagate through FFT
                x = ifft2(ifftshift(ifftshift(x,1),2));

                % ---------------------------------------------------------
                % Gradient w.r.t. probe
                % ---------------------------------------------------------
                dldw2 = x .* conj(memo1);
                dldw2 = sum(dldw2, 3);

                % ---------------------------------------------------------
                % Gradient w.r.t. object
                % ---------------------------------------------------------
                dldw1 = 0 * wavefront1;

                for con = 1:size(pos,2)
                    dldw1( ...
                        pos(2,con):pos(2,con)+size(wavefront2,1)-1, ...
                        pos(1,con):pos(1,con)+size(wavefront2,2)-1 ) = ...
                    dldw1( ...
                        pos(2,con):pos(2,con)+size(wavefront2,1)-1, ...
                        pos(1,con):pos(1,con)+size(wavefront2,2)-1 ) + ...
                        x(:,:,con) .* conj(wavefront2);
                end

                % Ensure complex gradient consistency
                dldw2 = conj(dldw2);
                dldw1 = conj(dldw1);
            end
        end
    end
end