clc
clear

pix = 128;
mag = 4;

img_number = 300;
load('CP_datasets/imgRawData.mat');


imRaw = imRaw - min(imRaw(:));
imRaw = imRaw / max(imRaw(:));
imRaw = sqrt(imRaw);

imRaw = gpuArray(single(imRaw));

cp_cube = combine(arrayDatastore(ScanPos, 'IterationDimension',1), ...
                  arrayDatastore(imRaw, 'IterationDimension',3));

batchSize = 6;
cp_cube = minibatchqueue(cp_cube,...
            'MiniBatchSize',     batchSize,...
            'MiniBatchFormat',   ["",""],...
            'OutputEnvironment', {'gpu'},...
            'OutputAsDlarray',   false,...
            'OutputCast',       'single');


epoch_max = 100;
iteration = 0;
epoch = 0;

%% recon 
lr = 0.01;

% 
optimizer1 = optimizers.Adam(0.9,0.999,1e-15);
optimizer2 = optimizers.Adam(0.9,0.999,1e-15);

%% init_guess;
wavefront1 = dlarray(gpuArray.ones(pix*mag,pix*mag,'single'));
wavefront2 = dlarray(gpuArray.ones(pix*mag,pix*mag,'single'));


while epoch < epoch_max
    epoch = epoch + 1;
    cp_cube.shuffle();

    this_loss = 0;
    remain = size(imRaw,3);
    start_tic = tic;
    while cp_cube.hasdata()
        iteration = iteration + 1;
        % disp(num2str(remain))

        remain = max(remain - batchSize,0);

        [dX,y_obs] = cp_cube.next();

        Hs = gpuArray.zeros(pix*mag, pix*mag, size(y_obs,3), 'single');

        for channel = 1:size(dX,1)
            x_pos = dX(channel,1);
            y_pos = dX(channel,2);
            Hs(:,:,channel) = fftshift(exp(1j*2*pi.*(env.Fx .* x_pos * mag + ...
                                            env.Fy .* y_pos * mag)));
        end

        [loss,dldw1,dldw2] = dlfeval(@cp_forward,...
                                           wavefront1,...
                                           wavefront2,...
                                           y_obs, ...
                                           Hs, env.prop,...
                                           mag);

        this_loss = this_loss + loss;
    
        %% learning for parameters
        wavefront1 = optimizer1.step(wavefront1, dldw1, iteration, lr);
        wavefront2 = optimizer2.step(wavefront2, dldw2, iteration, lr);
    end
    this_loss
    if mod(epoch,5) == 0
        figure(2024)
        ww = extractdata(wavefront1);
        subplot(121); imshow(abs(ww),[])
        subplot(122); imshow(angle(ww),[])
        drawnow;
    end
        clc
    fprintf("epoch %d done! takes %f \n",epoch,toc(start_tic));
end


function [loss,dldw1,dldw2] = cp_forward(wavefront1,...
                                         wavefront2,...
                                         y_obs, ...
                                         Hs,prop,...
                                         mag)

fwd = @(x) fft(fft(x,[],1),[],2);
bwd = @(x) ifft(ifft(x,[],1),[],2);

x        = bwd(fwd(wavefront1) .* Hs);
x        = x .* wavefront2;
x        = bwd(fwd(x) .* prop);
dX       = abs(x);

dX_ds = avgpool(dX.^2,[mag,mag],"Stride",mag,"DataFormat","SSB");
% dX_ds = dlresize(dX.^2,"Scale",1/mag,"DataFormat","SSB","Method","nearest");
loss = l2loss(sqrt(dX_ds), y_obs, "DataFormat","SSB");

[dldw1,dldw2] = dlgradient(loss, wavefront1, wavefront2);
end

