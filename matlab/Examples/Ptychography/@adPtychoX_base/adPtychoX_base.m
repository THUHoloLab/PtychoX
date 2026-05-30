% =========================================================================
% adPtychoX_base.m
%
% Differentiable ptychographic forward/backward operator
% implemented as a custom deep learning operation.
% =========================================================================

classdef adPtychoX_base < deep.DifferentiableFunction
    properties
        enable_cuda = true;
    end

    properties (SetAccess = private)
        cuFFThandle_many = [];
        latentW_buffer = [];
        latentZ_buffer = [];
    end

    methods
        function self = adPtychoX_base(arg)
            arguments
                arg.enable_cuda = true;
                arg.sample_size = [];
                arg.probe_size = [];
                arg.batch_size = [];
            end

            output_num = 1;

            self@deep.DifferentiableFunction( ...
                output_num, ...
                SaveInputsForBackward  = true, ...
                SaveOutputsForBackward = false, ...
                NumMemoryValues        = 0 );

            self.enable_cuda = arg.enable_cuda;

            if self.enable_cuda && ...
               ~isempty(arg.sample_size) && ...
               ~isempty(arg.probe_size) && ...
               ~isempty(arg.batch_size)
                self.initBuffers(arg.sample_size, arg.probe_size, arg.batch_size);
            end
        end

        function observe = forward(self, wavefront1, wavefront2, pos)
            if self.enable_cuda
                self.assertCudaBuffersReady();

                observe = fwd_adPtychoX_base( ...
                    wavefront1, ...
                    wavefront2, ...
                    int32(pos), ...
                    self.cuFFThandle_many, ...
                    self.latentW_buffer, ...
                    self.latentZ_buffer ...
                );
            else
                memo1 = gpuArray.zeros( ...
                    size(wavefront2, 1), ...
                    size(wavefront2, 2), ...
                    size(pos, 2), 'single');

                for con = 1:size(pos, 2)
                    memo1(:, :, con) = wavefront1( ...
                        pos(2, con):pos(2, con) + size(wavefront2, 1) - 1, ...
                        pos(1, con):pos(1, con) + size(wavefront2, 2) - 1);
                end

                memo2 = memo1 .* wavefront2;
                memo2 = fftshift(fftshift(fft2(memo2), 1), 2);
                observe = abs(memo2);
            end
        end

        function [dldw1, dldw2, dldps] = backward(self, ...
                                                  dl_dout, ...
                                                  ~, ...
                                                  wavefront1, ...
                                                  wavefront2, ...
                                                  pos)
            dldps = [];

            if self.enable_cuda
                self.assertCudaBuffersReady();

                [dldw1, dldw2] = bwd_adPtychoX_base( ...
                    dl_dout, ...
                    wavefront1, ...
                    wavefront2, ...
                    int32(pos), ...
                    self.cuFFThandle_many, ...
                    self.latentW_buffer, ...
                    self.latentZ_buffer ...
                );
            else
                memo1 = gpuArray.zeros( ...
                    size(wavefront2, 1), ...
                    size(wavefront2, 2), ...
                    size(pos, 2), 'single');

                for con = 1:size(pos, 2)
                    memo1(:, :, con) = wavefront1( ...
                        pos(2, con):pos(2, con) + size(wavefront2, 1) - 1, ...
                        pos(1, con):pos(1, con) + size(wavefront2, 2) - 1);
                end

                memo2 = memo1 .* wavefront2;
                memo2 = fftshift(fftshift(fft2(memo2), 1), 2);

                x = dl_dout .* sign(memo2);
                x = ifft2(ifftshift(ifftshift(x, 1), 2));
                dldw2 = sum(conj(memo1) .* x, 3);
                dldw1 = zeros(size(wavefront1), 'like', wavefront1);
                for con = 1:size(pos, 2)
                    dldw1( ...
                        pos(2, con):pos(2, con) + size(wavefront2, 1) - 1, ...
                        pos(1, con):pos(1, con) + size(wavefront2, 2) - 1) = ...
                        dldw1( ...
                            pos(2, con):pos(2, con) + size(wavefront2, 1) - 1, ...
                            pos(1, con):pos(1, con) + size(wavefront2, 2) - 1) + ...
                        conj(wavefront2) .* x(:, :, con);
                end
            end
        end

        function initBuffers(self, sample_size, probe_size, batch_size)
            if ~self.enable_cuda
                return;
            end

            if isa(sample_size, 'gpuArray')
                sample_size = gather(sample_size);
            end
            if isa(probe_size, 'gpuArray')
                probe_size = gather(probe_size);
            end
            if isa(batch_size, 'gpuArray')
                batch_size = gather(batch_size);
            end

            self.clearBuffers();

            [self.cuFFThandle_many,...
             self.latentW_buffer,...
             self.latentZ_buffer]= init_ptyX_base_buffers( ...
                int32(sample_size), ...
                int32(sample_size), ...
                int32(probe_size), ...
                int32(probe_size), ...
                int32(batch_size));
        end

        function clearBuffers(self)
            if ~isempty(self.cuFFThandle_many)
                delete_ptyX_base_buffers(self.cuFFThandle_many, self.latentW_buffer, self.latentZ_buffer);
            end
            self.cuFFThandle_many = [];
            self.latentW_buffer = [];
            self.latentZ_buffer = [];
        end

        function delete(self)
            self.clearBuffers();
        end
    end

    methods (Access = private)
        function assertCudaBuffersReady(self)
            if isempty(self.cuFFThandle_many) || isempty(self.latentW_buffer) || isempty(self.latentZ_buffer)
                error('adPtychoX_base:BuffersNotInitialized', ...
                    'CUDA buffers are not initialized. Call initBuffers or construct with sample_size/probe_size/batch_size.');
            end
        end
    end
end

