% generating simulation data for coded ptychography
clc
clear
addpath(genpath('CP_funcs'))
reset(gpuDevice());

pix = 512;
mag = 4;

img_number = 200;
env = helpers.init_expEnvs(mag, [pix,pix]);
imRaw = zeros(pix,pix,img_number);

%% prepare ptychographic dataset
img1 = single(imread('CP_datasets/cameraman512.png'))/255;
img2 = mean(single(imread('CP_datasets/I01.bmp'))/255,3);
img1 = imresize(img1,[pix * mag, pix * mag]);
img2 = imresize(img2,[pix * mag, pix * mag]);

U0 = mat2gray(img1 + 0.01) .* exp(1i*pi*img2);
U0 = gpuArray(single(U0));



len = 100;
fx = (-pix * mag/2:pix * mag/2-1)/len;
[fx,fy] = meshgrid(fx);
lambda = 0.532;
k = 2*pi/lambda;
fz = sqrt(1 - lambda^2 * (fx.^2 + fy.^2));
% prepare prop beam pattern
p0 = imresize(gpuArray.rand(64,'single'),[pix * mag, pix * mag],'box');
prop = ifftshift(fftshift(fft2(p0)) .* exp(1i * k * 1 * fz));
prop = ifft2(prop);
prop = gpuArray(single(prop));
pattern = prop;

% scanning positions
shift_range = 17;
locX = shift_range * (2 * rand(img_number,1) - 1);
locY = shift_range * (2 * rand(img_number,1) - 1);

ScanPos = [locX,locY];

for con = 1:size(ScanPos,1)
    con
    Hs = fftshift(exp(1j*2*pi.*(env.Fx.*ScanPos(con,1) * mag + ...
                       env.Fy.*ScanPos(con,2) * mag)));
                
    x   = ifft2(fft2(U0) .* Hs) .* pattern;
    x   = ifft2(fft2(x) .* env.prop);

    temp = imresize(abs(x).^2,[pix,pix],'box');

    imRaw(:,:,con) = temp + 0.001*max(temp(:)) * rand(pix);

end

save('CP_datasets/imgRawData.mat','imRaw','ScanPos','pattern','env');