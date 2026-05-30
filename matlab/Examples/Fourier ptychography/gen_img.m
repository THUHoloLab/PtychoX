clc
clear

img_counter = 15^2;

pix = 256;
downsam = 4;

img1 = single(imread('dataset/cameraman512.png'))/255;
img2 = mean(single(imread('dataset/I01.bmp'))/255,3);

img1 = imresize(img1,[pix * downsam, pix * downsam]);
img2 = imresize(img2,[pix * downsam, pix * downsam]);

U0 = mat2gray(img1 + 0.01) .* exp(1i * pi * img2);
U0 = gpuArray(single(U0));

% pupil function 
[x,y] = meshgrid(linspace(-1,1,pix));
r = sqrt(x.^2 + y.^2);
t = atan2(y,x);
cut_off = 0.6;

otf = r < cut_off;
idx = (r/cut_off) < 1;
z = gpuArray.zeros(size(r),'single');

rng(1)
ww = 0.2*rand(1,15);
for con = 1:15
z(idx) = z(idx) + ww(con) * zernike.zernfun2(con, ...
                                         (r(idx))/cut_off, t(idx), 'norm');
end
pupil = otf .* exp(1i*2*pi .* z);


% fourier scanning position
rng(12);
boundary_left = round(pix) / 2;
boundary_right = pix * downsam - pix - boundary_left;

x = linspace(boundary_left,boundary_right,sqrt(img_counter));
[x,y] = meshgrid(x);
pos = round([x(:)';y(:)']);

plot(pos(1,:),pos(2,:),'x')
% generate image
imRaw = zeros(pix,pix,img_counter,'single');
FP = fftshift(fft2(U0));
for con = 1:img_counter
    this_pos = pos(:,con);
    sub = FP(this_pos(2,1):this_pos(2,1)+pix-1,...
             this_pos(1,1):this_pos(1,1)+pix-1) / downsam^2;
    
    temp_img = abs(ifft2(fftshift(sub .* pupil))).^2;
    imRaw(:,:,con) = temp_img + 0*max(temp_img(:)) .* gpuArray.rand(pix,'single');

    % imwrite(mat2gray(temp_img),['imgs/img',num2str(con),'.jpg']);
end
imRaw = imRaw - min(imRaw(:));
imRaw = imRaw / max(imRaw(:));

save('dataset/imgRawData.mat','imRaw','pos','otf','downsam');