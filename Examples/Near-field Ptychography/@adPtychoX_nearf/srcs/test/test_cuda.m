clc
clear

rng(256);

img = single(imread('cameraman.tif'))/255;
img = gpuArray(img);


pix = 1024;

img = imresize(img,[pix,pix]);

pix_sz = 5.86;
lambda = 0.532;

fx = (-pix/2:pix/2-1)/(pix * pix_sz);
[fx,fy] = meshgrid(fx);

fz = sqrt(1 - lambda.^2 * (fx.^2 + fy.^2));

mask = lambda.^2 * (fx.^2 + fy.^2) < 1;

prop = exp(1i*2*pi/lambda * 4000 * fz .* mask);

prop = complex(gpuArray(single(prop)));



shifts = 25*randn(2,10,'single') ;
shifts = gpuArray(shifts);

U = complex(img + 1i * 1e-15);
T = complex(gpuArray.ones(size(U),'single'));

[o1,o2,o3] = cuPtycho_fft_Fwd(U, T, fftshift(prop), shifts / pix);


%%
fwd_A = @(x) fft(fft(x,[],1),[],2);
fwd_B = @(x) ifft(ifft(x,[],1),[],2);

fx = (-pix/2:pix/2-1)/(pix);
[fx,fy] = meshgrid(fx);

for con = 1:size(shifts,2)
    Hs(:,:,con) = fftshift(exp(1i*2*pi.*(fx .* shifts(1,con) + ...
                                         fy .* shifts(2,con))));
end
x_record         = ifft2(fft2(U) .* Hs);
% x         = x .* illu0;
x         = fwd_B(fwd_A(x_record) .* fftshift(prop));
dX        = abs(x).^2;
dX        = sqrt(imresize(dX,1/4,'box'));


max(abs(o1(:)))
max(abs(x_record(:)))

mean(abs(o1 - x_record).^2,'all')

mean(abs(o3 - dX).^2,'all')
% 
% figure();
% imshow(abs(o3(:,:,1) - dX(:,:,1)),[])

dX = gather(dX);
o3 = gather(o3);