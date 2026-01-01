clc
clear
addpath(genpath("adPtycho"));

reset(gpuDevice());

pix = 256;
mag = 4;

%% prepare ptychographic dataset
img1 = single(imread('source/cameraman512.png'))/255;
img2 = mean(single(imread('source/I01.bmp'))/255,3);

img1 = imresize(img1,[pix * mag, pix * mag]);
img2 = imresize(img2,[pix * mag, pix * mag]);

U0 = mat2gray(img1 + 0.2) .* exp(1i*pi*img2);
U0 = gpuArray(single(U0));

rng(12);

boundary_left = round(pix/7) + 1;
boundary_right = pix * mag - pix - boundary_left;

sample = 300;
xy = round( (boundary_right - boundary_left) * rand(2,sample)) + boundary_left;

len = 100;
fx = (-pix/2:pix/2-1)/len;
[fx,fy] = meshgrid(fx);
lambda = 0.532;
k = 2*pi/lambda;
fz = sqrt(1 - lambda^2 * (fx.^2 + fy.^2));
% prepare prop beam pattern
U = (fx.^2 + fy.^2) < (1/len * 60).^2;
p0 = imresize(gpuArray.rand(32,'single'),size(U),'box');
prop = ifftshift(fftshift(fft2(U .* p0)) .* exp(1i * k * 160 * fz));
prop = ifft2(prop);
prop = gpuArray(single(prop));
prop = prop .* (abs(prop) > 0.1);

figure();imshow(abs(prop),[])

img_data = gpuArray(zeros(pix,pix,sample,'single'));
for con = 1:sample
    con
    this_pos = xy(:,con);
    sub = U0(this_pos(2,1):this_pos(2,1)+pix-1,...
                  this_pos(1,1):this_pos(1,1)+pix-1);
    
    temp_img = abs(fftshift(fft2(sub .* prop)));
    img_data(:,:,con) = temp_img + 0*max(temp_img(:)) .* gpuArray.rand(pix,'single');
end

%% begin solving the inverse problem
img_data = gpuArray(img_data);

wavefront1 = rand(pix*mag,'single');
wavefront2 = single(U);

wavefront1 = gpuArray(complex(wavefront1));
wavefront2 = gpuArray(complex(wavefront2));
% 
wavefront1 = dlarray(wavefront1);
wavefront2 = dlarray(wavefront2);

batchSize = 35;

optimizer1 = optimizers.Adam(0.9,0.999,1e-16);
optimizer2 = optimizers.Adam(0.9,0.999,1e-16);
lr = 0.01;
iteration = 0;

enable_cuda = true;
adfoo = adPtycho(enable_cuda);

error_data = [];
epoch_tic = tic;
for epoch = 1:600
    listBlock = 1:sample;
    start_tic = tic;
    loss_epoch = 0;

    while ~isempty(listBlock)
        iteration = iteration + 1;

        len = length(listBlock);
        batchIdx = listBlock(1:min(batchSize,len));
        listBlock(1:min(batchSize,len)) = [];

        pos = single(gpuArray(xy(:,batchIdx)));

        [loss,dldw1,dldw2] = dlfeval(@model_loss,adfoo, ...
                                                 wavefront1, ...
                                                 wavefront2, ...
                                                 pos, ...
                                                 img_data(:,:,batchIdx)...
                                                 );

        wavefront1 = optimizer1.step(wavefront1,dldw1,iteration,lr);
        wavefront2 = optimizer2.step(wavefront2,dldw2,iteration,lr);
        
        loss_epoch = loss_epoch + extractdata(loss);
        % 
    end
    fprintf("epoch %d takes %f \n", epoch, toc(start_tic));
    % mse = mean(abs((U0 - extractdata(wavefront1))).^2,'all');
    error_data = [error_data,loss_epoch];

    if mod(epoch,5) == 0
        w1 = extractdata(wavefront1);
        w2 = extractdata(wavefront2);
        figure(121);
        img_show = [abs(w1),mat2gray(angle(w1))];
        img_show = imresize(img_show,0.6);

        subplot(211);imshow(img_show,[0,1]);
        subplot(212);imshow(abs(w2),[]);
        drawnow;
        pause(0.01);

        
        figure(123);
        
        plot(1:epoch,log(error_data));
    end
end

toc(epoch_tic)

function [loss,dldw1,dldw2] = model_loss(adfunc,wave1,wave2,position,obseY)
obseY = dlarray(obseY);

predY = adfunc(wave1,wave2,position);

loss = sum((predY - obseY).^2,'all');

[dldw1,dldw2] = dlgradient(loss,wave1,wave2);

end













