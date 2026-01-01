% 13*13
%% Simulate the forward imaging process of Fourier ptychography
clear
clc
reset(gpuDevice());


load('dataset/imgRawData.mat');
imRaw = sqrt(imRaw);

batchSize = 15; 

numEpochs = 50;
epoch = 0;
iteration = 0;

%% The iterative recovery process for FP
disp('initializing parameters')
foo = @(x) complex(gpuArray(single(x)));

pratio = 4;

oI = (imresize(mean(imRaw,3),pratio)); 
wave1 = dlarray(foo(ones(size(oI))));
wave2 = dlarray(foo(otf));   

disp('begin solving-----')

optimizer_w1 = optimizers.Adam(0.9,0.999,1e-5);
optimizer_w2 = optimizers.Adam(0.9,0.999,1e-5);

lr = 0.01;

cuFPM_foo = adPtychoX_fpm(true); % init differentiable function

while epoch < numEpochs
    epoch = epoch + 1;

    first = 1;
    last = 1;
    mask = 0;

    start_tic = tic;
    while last < size(imRaw,3)
        iteration = iteration + 1;
        
        last = first + batchSize - 1;

        batchIdx = first:min(last,size(imRaw,3));
        ledIdx =  int32(gpuArray(pos(:,batchIdx)));

        [loss,dldw1,dldw2] = dlfeval(@autodiff_fpm,...
                                           cuFPM_foo,...
                                           wave1, ...
                                           wave2, ...
                                           imRaw(:,:,batchIdx), ...
                                           ledIdx, ...
                                           pratio);
    
        first = last + 1;
  
        wave1 = optimizer_w1.step(wave1,dldw1,iteration,lr);
        wave2 = optimizer_w2.step(wave2,dldw2,iteration,lr);

        wave2 = min(max(abs(wave2),0.96),1.04) .* sign(wave2) .* otf;
    end
    clc
    
    this_timer = toc(start_tic);
    sprintf("at %d epoch, takes = %2f",epoch,this_timer)

    if mod(epoch,5) == 0 
        w2 = gather(extractdata(wave1));
        c1 = log(abs(fftshift(fft2(w2)))+1);

        w1 = gather(extractdata(wave2));
    
        img_show = [mat2gray(abs(w2)),mat2gray(angle(w2));
                    mat2gray(abs(c1)),mat2gray(imresize(angle(w1).*otf,pratio))];
        img_show = imresize(img_show,0.6);

        figure(7);
        imshow(img_show,[]);
        title(['Iteration No. = ',int2str(epoch), '  \alpha = ',num2str(lr)])
        drawnow;
    end

end

% w1 = extractdata(wave2);
% w2 = extractdata(wave1);


%% helpers
function [loss,dldw1,dldw2] = autodiff_fpm(foo_fpm,...
                                           wave1, ...
                                           wave2, ...
                                           obse_Y, ...
                                           ledIdx, ...
                                           pratio)

pred_Y = foo_fpm(wave1, ...
                 wave2, ...
                 obse_Y, ...
                 single(ledIdx), ...
                 pratio);

loss = srcs.fd_loss(pred_Y, obse_Y, 'isotropic');
% loss = l1loss(pred_Y, obse_Y,"DataFormat","SSB");
[dldw1, dldw2] = dlgradient(loss, wave1, wave2);
end