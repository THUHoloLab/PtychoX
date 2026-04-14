clc
clear
reset(gpuDevice());

pix = 512;
mag = 4;

%% prepare ptychographic dataset
img1 = single(imread('source/cameraman512.png'))/255;
img2 = mean(single(imread('source/I01.bmp'))/255,3);

img1 = imresize(img1,[pix * mag, pix * mag]);
img2 = imresize(img2,[pix * mag, pix * mag]);

U0 = mat2gray(img1 + 0.2) .* exp(1i*pi*img2);



boundary_left = round(pix/7) + 1;
boundary_right = pix * mag - pix - boundary_left;


sample = 900;
xy = round( (boundary_right - boundary_left) * rand(2,sample)) + boundary_left;

len = 100;
fx = (-pix/2:pix/2-1)/len;
[fx,fy] = meshgrid(fx);
lambda = 0.532;
k = 2*pi/lambda;
fz = sqrt(1 - lambda^2 * (fx.^2 + fy.^2));

U = (fx.^2 + fy.^2) < (1/len * 80).^2;
rng(12);
p0 = imresize(gpuArray(rand(32,'single')),size(U),'box');
prop = ifftshift(fftshift(fft2(U .* p0)) .* exp(1i * k * 90 * fz));
prop = ifft2(prop);
prop = gpuArray(single(prop));
% prop = prop .* (abs(prop) > 0.1);
figure(); imshow(abs(prop), []);

img_data = zeros(pix,pix,sample,'single');
for con = 1:sample
    con
    this_pos = xy(:,con);
    this_img = U0(this_pos(2,1):this_pos(2,1)+pix-1,...
                  this_pos(1,1):this_pos(1,1)+pix-1);
    
    temp_img = abs(fftshift(fft2(this_img .* prop)));

    img_data(:,:,con) = temp_img + 0*max(temp_img(:)) .* gpuArray.rand(pix,'single');
end

%% begin solving the inverse problem
wavefront1 = gpuArray.rand(pix*mag,'single');
wavefront2 = single(U);
wavefront2 = gpuArray(wavefront2);

iteration = 0;
lr = 1;
error_data = [];
tic
takes_time = 0;
frame = 0;
for epoch = 1:600

    % start_tic = tic;
    loss_epoch = 0;
    for iter = 1:sample
        % iter
        start_tic = tic;
        iteration = iteration + 1;
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

        wavefront1 = min(abs(wavefront1),1) .* sign(wavefront1);

        takes_time = takes_time + toc(start_tic);

        if mod(iteration,100) == 0
            frame = frame + 1;
            figure(121);
            img_show = [(abs(wavefront1)),mat2gray(angle(wavefront1))];
            img_show = imresize(img_show,0.6);

            probe_show = [mat2gray(abs(wavefront2)),mat2gray(angle(wavefront2))];
            probe_show = imresize(probe_show,size(img_show));

            img_show = [img_show,probe_show];
            img_show = imresize(img_show,0.6);
            textStr = sprintf('gpuArray, takes %.2f s', takes_time);
            frame_uint8 = uint8(255*img_show);
            frame_rgb = insertText(gather(frame_uint8), ...
                [5, 5], textStr, ...           % 左上角位置 [x, y]
                'FontSize', 48, ...
                'TextColor', 'white', ...
                'BoxColor', 'black', ...
                'BoxOpacity', 0.6);
            frame_gray = rgb2gray(frame_rgb);
            frame_rgb = imresize(frame_rgb,0.3);
            [map,idx] = rgb2ind(frame_rgb,8);
            if frame == 1
                imwrite(frame_gray, 'testPIE.gif', 'gif', ...
                        'LoopCount', 65535, ...
                        'DelayTime', 0.03);
            elseif (frame > 1) && (frame < 120)
                imwrite(frame_gray, 'testPIE.gif', 'gif', ...
                        'WriteMode', 'append', ...
                        'DelayTime', 0.03);
            end

            imshow(frame_gray,[])
            drawnow;
            % pause(0.01);
        end
        if frame > 120
            break
        end

    end
    if frame > 120
            break
    end
    % fprintf("epoch %d finished, takes %f second\n", epoch, toc(start_tic));

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









