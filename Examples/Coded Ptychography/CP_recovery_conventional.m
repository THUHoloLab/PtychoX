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

epoch_max = 10;
iteration = 0;

epoch = 0;

wavefront1 = gpuArray.ones(pix * 4, pix * 4,'single');
wavefront2 = gpuArray.ones(pix * 4, pix * 4,'single');

lr = 1;

H_d1 = fftshift(H_d1);
H_d2 = fftshift(H_d2);
while epoch < epoch_max
    epoch = epoch + 1;
    start_tic = tic;
    for pos_count = 1:size(imRaw,3)
        iteration = iteration + 1;

        Hs = exp(1j*2*pi.*(ScanPos(pos_count,1).*FX/imSize0 + ...
                           ScanPos(pos_count,2).*FY/imSize0));

        x_forward   = ifft2(fft2(wavefront1) .* H_d1 .* Hs);
        x           = ifft2(fft2(x_forward .* wavefront2).*H_d2);

        dX_ds       = sqrt(imresize(abs(x).^2,[size(imRaw,1),...
                                               size(imRaw,2)],'box'));

        dm          = (dX_ds - imRaw(:,:,pos_count)); loss = sum(dm(:).^2);
        x           = imresize(dm./(dX_ds + 1e-5),...
                      [size(Hs,1),size(Hs,2)],'nearest') .* x;
        x_backward  = ifft2(fft2(x) .* conj(H_d2));
        x           = deconvPIE(x_backward, wavefront2,'rPIE');

        dldw2 = deconvPIE(x_backward, x_forward,'rPIE');
        dldw1 = ifft2(fft2(x) .* conj(Hs) .* conj(H_d1));
        
        wavefront1 = wavefront1 - lr * dldw1;
        wavefront2 = wavefront2 - lr * dldw2;
    end
    fprintf("epoch %d done! takes %f \n",epoch,toc(start_tic));

    if mod(epoch,5) == 0
        w1 = gather(wavefront1);
        w2 = gather(wavefront2);
        figure(2024)
        subplot(121); imshow(abs(w1),[])
        subplot(122); imshow(angle(w1),[])
        drawnow;
    end

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
