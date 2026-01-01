clc
clear

load('dataset/imgRawData.mat');
imRaw = sqrt(imRaw);

downsam = 4;
pix = size(imRaw,1);

% wave1 = onesimresize(mean(imRaw,3),downsam);
wave1 = gpuArray.ones(size(imRaw,1) * downsam,'single');
wave1 = fftshift(fft2(wave1));
wave2 = gpuArray(otf);

epoch_max = 60;
iteration = 0;
lr = 1;

%% resort the updating order of the fre-scanning
rr = abs(pos(1,:) + pix/2 - pix * downsam / 2) + ...
     abs(pos(2,:) + pix/2 - pix * downsam / 2);
[B,I] = sort(rr);

for epoch = 1:epoch_max

    start_tic = tic;
    for img_counter = I
        iteration = iteration + 1;

        this_pos = pos(:,img_counter);
        sub = wave1(this_pos(2,1):this_pos(2,1)+pix-1,...
                    this_pos(1,1):this_pos(1,1)+pix-1);

        x = sub .* wave2;

        x = ifft2(ifftshift(x)) / downsam^2;

        dm = (abs(x) - imRaw(:,:,img_counter));

        x = dm .* sign(x) * downsam^2;
        x = fftshift(fft2(x)) ;

        dldw1    = deconv_pie(x, wave2,'rPIE');
        dldw2    = deconv_pie(x, sub,'rPIE');

        wave1(this_pos(2,1):this_pos(2,1)+pix-1,...
              this_pos(1,1):this_pos(1,1)+pix-1) = ...
        wave1(this_pos(2,1):this_pos(2,1)+pix-1,...
              this_pos(1,1):this_pos(1,1)+pix-1) - lr * dldw1;

        wave2 = wave2 - lr * dldw2;
        wave2 = wave2 .* otf;
    end
    sprintf("at %d epoch, takes = %2f",epoch,toc(start_tic))

    if mod(epoch,5) == 0
        figure(121);
        wavet1 = ifft2(ifftshift(wave1));
        wavet2 = log(abs(wave1) + 1);

        img_show = [mat2gray(abs(wavet1)),mat2gray(angle(wavet1));
                    mat2gray(abs(wavet2)),mat2gray(imresize(angle(wave2).*otf,downsam))];
           
        img_show = imresize(img_show,0.6);
        imshow(img_show,[]);
        drawnow;
        pause(0.01);
    end

end


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