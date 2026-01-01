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
reset(gpuDevice());
addpath(genpath('CP_funcs'))


pix = 128;
img_number = 300;
init_environment;

load('CP_datasets/imgRawData.mat');
imRaw = sqrt(imRaw);
imRaw = imRaw - min(imRaw(:));
imRaw = imRaw / max(imRaw(:));

ScanPos = gpuArray(single(ScanPos));
imRaw = gpuArray(single(imRaw));

total_size = size(imRaw,3);%imRaw = imRaw;

batchSize = 16;


epoch_max = 300;

lr = 0.06;
optimizer_w1 = optimizers.Adam(0.9,0.999,1e-5);
optimizer_w2 = optimizers.Adam(0.9,0.999,1e-5);

foo = @(x) gpuArray(single(x));

wavefront1 = gpuArray.ones(pix*4,pix*4,'single');
wavefront1 = dlarray(complex(wavefront1));

wavefront2 = gpuArray.ones(pix*4,pix*4,'single');
wavefront2 = dlarray(complex(wavefront2));



epoch = 0;
iteration = 0;

loss_data = [];
fwdfoo = adPtychoX_coded();

pos_new = gpuArray([ScanPos(:,1)';ScanPos(:,2)']);
H_d2 = fftshift(H_d2);
H_d1 = fftshift(H_d1);

while epoch < epoch_max
    epoch = epoch + 1;

    this_loss = 0;
    remain = size(imRaw,3);

    listBlock = randperm(total_size);
    start_tic = tic;
    while ~isempty(listBlock)
        iteration = iteration + 1;
        % disp(num2str(remain))

        remain = max(remain - batchSize,0);

        % [dX,y_obs] = cp_cube.next();
        len = length(listBlock);
        batchIdx = listBlock(1:min(batchSize,len));
        listBlock(1:min(batchSize,len)) = [];
        pos = pos_new(:,batchIdx) / imSize0;

        [loss,dldw1,dldw2] = dlfeval(@model_loss,...
                                        fwdfoo,...
                                        wavefront1,...
                                        wavefront2,...
                                        H_d1,...
                                        H_d2,...
                                        pos,...
                                        imRaw(:,:,batchIdx));

        this_loss = this_loss + loss;
        % learning for parameters
        wavefront1 = optimizer_w1.step(wavefront1,dldw1,iteration,lr);
        wavefront2 = optimizer_w2.step(wavefront2,dldw2,iteration,lr);
    end
    
    if mod(epoch,10) == 0
        w1 = extractdata(wavefront1);
        w2 = extractdata(wavefront2);
        figure(2024)
        subplot(121); imshow(abs(w1),[])
        subplot(122); imshow(angle(w1),[])
        % subplot(133); imshow(abs(w2),[])
        drawnow;
    end

    wait(gpuDevice());
    clc
    fprintf("epoch %d done! takes %f \n",epoch,toc(start_tic));
end

function [loss,dldw1,dldw2] = model_loss(fwdfoo,wave1,wave2,H1,H2,dX,dY)

dY = dlarray(dY,"SSB");

obser = fwdfoo(wave1,wave2,H1,H2,dX);

loss = fd_loss(obser,dY,'isotropic');
[dldw1,dldw2] = loss.dlgradient(wave1,wave2);
end


function loss = fd_loss(x, y, type)


dW = x - y;

RxdW = [dW(:,2:end,:) - dW(:,1:end-1,:), dW(:,1,:) - dW(:,end,:)];
RydW = [dW(2:end,:,:) - dW(1:end-1,:,:); dW(1,:,:) - dW(end,:,:)];

switch type
    case "isotropic"
        loss = sqrt(abs(RxdW).^2 + abs(RydW).^2  + 1e-5);
    case "anisotropic"
        loss = abs(RxdW) + abs(RydW);
    otherwise
    error("parameter #3 should be a string either 'isotropic', or 'anisotropic'")
end

loss = sum(loss(:));

end