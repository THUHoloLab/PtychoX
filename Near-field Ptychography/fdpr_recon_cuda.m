clc
clear
reset(gpuDevice());


mag = 4;
pix = 128;
% env = helpers.init_expEnvs(mag, [pix,pix]);

load('CP_datasets/imgRawData.mat');

imRaw = imRaw - min(imRaw(:));
imRaw = imRaw / max(imRaw(:));
imRaw = sqrt(imRaw);

imRaw = gpuArray(single(imRaw));
ScanPos = single(ScanPos');
ScanPos = gpuArray(ScanPos);

img_total = size(imRaw,3);
batchSize = 32;

epoch_max = 100;
iteration = 0;
epoch = 0;

%% recon 
lr = 0.06;

wave1 = dlarray(complex(gpuArray.ones(pix * mag,pix * mag,'single')));
wave2 = dlarray(complex(gpuArray.ones(pix * mag,pix * mag,'single')));

global cuPtycho
cuPtycho = adNFPtycho(4);

optimizer_W = optimizers.Adam(0.9,0.9,1e-15);
optimizer_E = optimizers.Adam(0.9,0.9,1e-15);

while epoch < epoch_max
    epoch = epoch + 1;
    % cp_cube.shuffle();

    this_loss = 0;
    listBlock = randperm(img_total);
    remain = size(imRaw,3);
    start_tic = tic;
    while ~isempty(listBlock)
        iteration = iteration + 1;
        % disp(num2str(remain))

        remain = max(remain - batchSize,0);

        len = length(listBlock);

        batchIdx = listBlock(1:min(batchSize,len));

        listBlock(1:min(batchSize,len)) = [];

        scanDx = gpuArray(single(ScanPos(:,batchIdx) / pix));
        [loss,dldw1,dldw2] = dlfeval(@model_loss,...
                                                 wave1,...
                                                 wave2,...
                                                 env.size,...
                                                 imRaw(:,:,batchIdx), ...
                                                 scanDx, env.prop);

        this_loss = this_loss + loss;
        fprintf("at %d iter, remain = %d \n",iteration,remain)
        %% learning for parameters
        wave1 = optimizer_W.step(wave1, dldw1, iteration, lr);
        wave2 = optimizer_E.step(wave2, dldw2, iteration, lr);

        if mod(iteration,10) == 0
            figure(2024)
            ww = extractdata(wave1);
            ill = extractdata(wave2);

            img_all = [mat2gray(abs(ww)),  mat2gray(angle(ww));
                       mat2gray(abs(ill)), mat2gray(angle(ill))];

            imshow(img_all,[])
            drawnow;
        end
    end

    clc
    fprintf("epoch %d done! takes %f \n",epoch,toc(start_tic));
end


function [loss,dldw1,dldw2] = model_loss(wave1,...
                                         wave2,...
                                         img_sz,...
                                         y_obs, ...
                                         shifts,prop)


global cuPtycho

dX_ds = cuPtycho(wave1, wave2, prop, shifts);

loss    = loss_fun.fd_loss(dX_ds, y_obs, 'sum');
loss_tv = loss_fun.fd_loss(wave1,0,'sum');
[dldw1, dldw2] = dlgradient(loss + 0.03*loss_tv, wave1, wave2);
end