%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{

    Implementation of coded ptychography using FAIRY

    Codes were adapted from:
    Shaowei Jiang, Pengming Song, Tianbo Wang, et al.,
    "Spatial and Fourier domain ptychography for high-throughput bio-imaging", 
    Nature Protocols, 2023

%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear

addpath(genpath('CP_funcs'))


pix = 128;
img_number = 300;

load('CP_datasets/imgRawData.mat');

init_environment;

total_size = size(imRaw,3);%imRaw = imRaw;

ScanPos = ScanPos(1:1:end,:);
imRaw = imRaw(:,:,1:1:end);
cp_cube = combine(arrayDatastore(ScanPos,   'IterationDimension',1), ...
                  arrayDatastore(imRaw,    'IterationDimension',3));

batchSize = 16;
cp_cube = minibatchqueue(cp_cube,...
            'MiniBatchSize',     batchSize,...
            'MiniBatchFormat',   ["",""],...
            'OutputEnvironment', {'gpu'},...
            'OutputAsDlarray',   false,...
            'OutputCast',       'single');



epoch_max = 100;

lr = 0.05;
optimizer_w1 = optimizers.Adam(0.9,0.999,1e-5);
optimizer_w2 = optimizers.Adam(0.9,0.999,1e-5);

foo = @(x) gpuArray(single(x));
get_ones = @(x) ones(size(x));

wavefront1 = gpuArray.ones(pix*4,pix*4,'single');
wavefront2 = gpuArray.ones(pix*4,pix*4,'single');


epoch = 0;
iteration = 0;

loss_data = [];
H_d2 = fftshift(H_d2);
H_d1 = fftshift(H_d1);

while epoch < epoch_max
    epoch = epoch + 1;
    cp_cube.shuffle();
    this_loss = 0;
    remain = size(imRaw,3);
  
    start_tic = tic;
    while cp_cube.hasdata()
        iteration = iteration + 1;
        remain = max(remain - batchSize,0);

        [dX,y_obs] = cp_cube.next();

        Hs = gpuArray(zeros(pix*mag,pix*mag,size(y_obs,3),'single'));
        for channel = 1:size(dX,1)
            x_pos = dX(channel,1);
            y_pos = dX(channel,2);
            Hs(:,:,channel) = exp(1j*2*pi.*(FX.*x_pos/imSize0 + ...
                                            FY.*y_pos/imSize0));
        end


        [loss,dldw1,dldw2] = cp_forward(wavefront1,...
                                        wavefront2, ...
                                        y_obs, ...
                                        Hs, ...
                                        H_d1, ...
                                        H_d2);

        this_loss = this_loss + loss;
    
        %% learning for parameters
        wavefront1 = optimizer_w1.step(wavefront1,dldw1,iteration,lr);
        wavefront2 = optimizer_w2.step(wavefront2,dldw2,iteration,lr);
    end
    if mod(epoch,10) == 0
        w1 = gather(wavefront1);
        w2 = gather(wavefront2);
        figure(2024)
        subplot(121); imshow(abs(w1),[])
        subplot(122); imshow(angle(w1),[])
        drawnow;
    end

    clc
    fprintf("epoch %d done! takes %f \n",epoch,toc(start_tic));
end

%% helper function
function [loss,dldw1,dldw2] = cp_forward(wavefront1,...
                                         wavefront2, ...
                                         y_obs, ...
                                         Hs, ...
                                         H_d1, ...
                                         H_d2)

x_forward   = ifft2(fft2(wavefront1) .* H_d1 .* Hs);
x           = ifft2(fft2(x_forward .* wavefront2).*H_d2);

dX_ds       = sqrt(imresize(abs(x).^2,[size(y_obs,1),...
                                       size(y_obs,2)],'box'));

dm          = (dX_ds - y_obs); loss = sum(dm(:).^2);
x           = imresize(dm./(dX_ds + 1e-5),...
                      [size(Hs,1),size(Hs,2)],'nearest') .* x;

x_backward  = ifft2(fft2(x) .* conj(H_d2));
x           = deconvPIE(x_backward, wavefront2,'rPIE');

dldw1 = sum(ifft2(fft2(x) .* conj(Hs) .* conj(H_d1)),3);
dldw2 = sum(deconvPIE(x_backward,x_forward,'rPIE'),3);
end

function out = deconvPIE(in,ker,type)
    switch type
        case 'ePIE'
            out = conj(ker) .* in ./ max(max(abs(ker).^2));
        case 'tPIE'
            bias = abs(ker) ./ max(max(abs(ker)));
            fenzi = conj(ker) .* in ;
            fenmu = (abs(ker).^2 + 0.0001);
            out = bias .* fenzi ./ fenmu;
        case 'rPIE'
            % bias = abs(ker) ./ max(max(abs(ker)));
            fenzi = conj(ker) .* in ;
            fenmu = (0.4*abs(ker).^2 + 0.6 * max(max(abs(ker).^2)));
            out = fenzi ./ fenmu;
        case 'none'
            out = conj(ker) .* in;
        otherwise 
            error()
    end
end
