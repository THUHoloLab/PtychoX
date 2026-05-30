clc
clear
reset(gpuDevice());

pix = 1024;
mag = 4;

%% prepare ptychographic dataset
img1 = single(imread('source/cameraman512.png'))/255;
img2 = mean(single(imread('source/I01.bmp'))/255,3);

img1 = imresize(img1,[pix * mag, pix * mag]);
img2 = imresize(img2,[pix * mag, pix * mag]);

U0 = mat2gray(img1 + 0.2) .* exp(1i*pi*img2);

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

U = (fx.^2 + fy.^2) < (1/len * 130).^2;
p0 = imresize(rand(32,'single'),size(U),'box');
prop = ifftshift(fftshift(fft2(U .* p0)) .* exp(1i * k * 160 * fz));
prop = ifft2(prop);
prop = (single(prop));
prop = prop .* (abs(prop) > 0.1);

img_data = zeros(pix,pix,sample,'single');
for con = 1:sample
    con
    this_pos = xy(:,con);
    this_img = U0(this_pos(2,1):this_pos(2,1)+pix-1,...
                  this_pos(1,1):this_pos(1,1)+pix-1);
    
    temp_img = abs(fftshift(fft2(this_img .* prop)));

    img_data(:,:,con) = temp_img + 0*max(temp_img(:)) .* rand(pix,'single');
end

%% begin solving the inverse problem
wavefront1 = rand(pix*mag,'single');
wavefront2 = single(U);
wavefront2 = (wavefront2);

lr = 1;
error_data = [];
tic
for epoch = 1:400

    start_tic = tic;
    loss_epoch = 0;
    for iter = 1:sample
        % iter
        this_pos = xy(:,iter);
        

        sub = wavefront1(this_pos(2,1):this_pos(2,1)+pix-1,...
                         this_pos(1,1):this_pos(1,1)+pix-1);

        x = fftshift(fft2(sub .* wavefront2));
        loss_epoch = loss_epoch + sum((abs(x) - img_data(:,:,iter)).^2,'all');

        x = x - img_data(:,:,iter) .* sign(x);
        x = ifft2(ifftshift(x)) ;
        
        dldw1    = deconv_pie(x, wavefront2,'rPIE');
        dldw2    = deconv_pie(x, sub,'rPIE');

        % update parameters
        wavefront1(this_pos(2,1):this_pos(2,1)+pix-1,...
                   this_pos(1,1):this_pos(1,1)+pix-1) = ...
        wavefront1(this_pos(2,1):this_pos(2,1)+pix-1,...
                   this_pos(1,1):this_pos(1,1)+pix-1) - lr * dldw1;

        wavefront2 = wavefront2 - lr * dldw2;
    end
    fprintf("epoch %d finished, takes %f second\n", epoch, toc(start_tic));
        figure(121);
        img_show = [mat2gray(abs(wavefront1)),mat2gray(angle(wavefront1))];
        img_show = imresize(img_show,0.6);
        subplot(211);imshow(img_show,[]);
        subplot(212);imshow(abs(wavefront2),[]);
        drawnow;
        pause(0.01);

        mse = mean(abs((U0 - wavefront1)).^2,'all');
        error_data = [error_data,loss_epoch];
        figure(122);
        plot(1:epoch,log(error_data));
end
toc

function out = deconv_pie(in,ker,type)
    switch type
        case 'ePIE'
            out = conj(ker) .* in ./ max(max(abs(ker).^2));
        case 'tPIE'
            bias = abs(ker) ./ max(max(abs(ker)));
            fenzi = conj(ker) .* in ;
            fenmu = (abs(ker).^2 + 1);
            out = bias .* fenzi ./ fenmu;
        case 'rPIE'
            fenzi = conj(ker) .* in ;
            fenmu = (abs(ker).^2 + 0.6 * max(abs(ker(:).^2)));
            out = fenzi ./ fenmu;
        case 'none'
            out = conj(ker) .* in;
        otherwise 
            error()
    end
end

function [loss] = retinexloss_l1(x,y,type)
    temp_img = x - y;
    tv_x = [temp_img(2:end,:) - temp_img(1:end-1,:);temp_img(1,:) - temp_img(end,:)];
    tv_y = [temp_img(:,2:end) - temp_img(:,1:end-1),temp_img(:,1) - temp_img(:,end)];

    switch type
        case 'sum'
            loss = sum(sqrt(abs(tv_x).^2 + abs(tv_y).^2 + 1e-5),"all");
        case 'mean'
            loss = mean(sqrt(abs(tv_x).^2 + abs(tv_y).^2 + 1e-5),"all");
        otherwise
            error('type must be sum or mean');
    end
end









