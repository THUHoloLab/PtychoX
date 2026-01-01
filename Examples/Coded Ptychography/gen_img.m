% generating simulation data for coded ptychography
clc
clear
addpath(genpath('CP_funcs'))
reset(gpuDevice());

pix = 128;
img_number = 300;

mag = 4;
imRaw = zeros(pix,pix,img_number);

%% prepare ptychographic dataset
img1 = single(imread('CP_datasets/cameraman512.png'))/255;
img2 = mean(single(imread('CP_datasets/I01.bmp'))/255,3);
img1 = imresize(img1,[pix * mag, pix * mag]);
img2 = imresize(img2,[pix * mag, pix * mag]);

U0 = mat2gray(img1 + 0.2) .* exp(1i*pi*img2);
U0 = gpuArray(single(U0));


coded_pattern = imresize(gpuArray.rand(32,'single'),[pix * mag, pix * mag],'box');


init_environment;

d1 = -(396.95).*1e-6;
H_d1 = (exp(1i.*(-d1).*real(kzm)).*exp(-abs((-d1)).*abs(imag(kzm))).*...
       ((k0^2-kxm.^2-kym.^2)>=0));


ScanPos = [locX,locY];

for con = 1:size(ScanPos,1)
    con
    Hs = exp(1j*2*pi.*(FX.*ScanPos(con,1)/imSize0 + ...
                       FY.*ScanPos(con,2)/imSize0));
                
    Hs = fftshift(fftshift(Hs,1),2);

    x           = fftshift(fft2(U0)) .* H_d1;
    x_forward   = ifft2(ifftshift(x .* Hs));
    x           = x_forward .* coded_pattern;
    x           = ifft2(ifftshift((fftshift(fft2(x)).*H_d2)));

    dX          = abs(x);
    imRaw(:,:,con) = imresize(dX.^2,[pix,pix],'box');
end

save('CP_datasets/imgRawData.mat','imRaw','ScanPos','coded_pattern');