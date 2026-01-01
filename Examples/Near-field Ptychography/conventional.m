clc
clear

pix = 128;
mag = 4;

img_number = 300;
% env = helpers.init_expEnvs(mag, [pix,pix]);
load('CP_datasets/imgRawData.mat');


imRaw = imRaw - min(imRaw(:));
imRaw = imRaw / max(imRaw(:));
imRaw = sqrt(imRaw);

imRaw = gpuArray(single(imRaw));

wavefront1 = gpuArray.ones(pix * mag,'single');
wavefront2 = gpuArray.ones(pix * mag,'single');


epoch_max = 10;
iteration = 0;
lr = 0.6;

for epoch = 1:epoch_max
    start_tic = tic;
    for img_count = 1:size(imRaw,3)
        iteration = iteration + 1;

        Hs = fftshift(exp(1j*2*pi.*(env.Fx.*ScanPos(img_count,1) * mag + ...
                           env.Fy.*ScanPos(img_count,2) * mag)));
                
        x_record    = ifft2(fft2(wavefront1) .* Hs);
        x           = x_record .* wavefront2;
        x           = ifft2(fft2(x) .* env.prop);

        y_obs = sqrt(imresize(abs(x).^2,1/mag,'box'));

        dm = (y_obs - imRaw(:,:,img_count));

        x = imresize(dm./(y_obs + 1e-5), mag,'nearest') .* x;

        x_bwd = ifft2(fft2(x) .* conj(env.prop));

        dldw1 = deconvPIE(x_bwd, wavefront2, 'tPIE');
        dldw2 = deconvPIE(x_bwd, x_record,   'tPIE');
        
        dldw1 = ifft2(fft2(dldw1) .* conj(Hs));


        wavefront1 = wavefront1 - lr * dldw1;
        wavefront2 = wavefront2 - lr * dldw2;

        if mod(iteration,10) == 0
            figure(2024)
            ww = wavefront1;
            ill = wavefront2;

            img_all = [mat2gray(abs(ww)),  mat2gray(angle(ww));
                       mat2gray(abs(ill)), mat2gray(angle(ill))];

            imshow(img_all,[])
            drawnow;
        end
    end
    fprintf("epoch %d done! takes %f \n",epoch,toc(start_tic));
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
            fenmu = (0.2*abs(ker).^2 + 0.8 * max(max(abs(ker).^2)));
            out = fenzi ./ fenmu;
        case 'none'
            out = conj(ker) .* in;
        otherwise 
            error()
    end
end
