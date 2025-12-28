classdef diffractor < handle
    %DIFFRACTOR 此处显示有关此类的摘要
    %   此处显示详细说明
    
    properties
        lambda;
        toz;
        img_sz;
        fz;
        pix_size;
        pupil;
        mask;
    end
    
    methods
        function obj = diffractor(args)
            arguments
                args.lambda        = 0.532;
                args.toz           = 1000;
                args.img_sz        = [256,256];
                args.pix_size      = 6.5;
            end

            obj.lambda       = args.lambda;
            obj.toz          = args.toz;
            obj.img_sz       = args.img_sz;
            obj.pix_size     = args.pix_size;

            img_w = obj.img_sz(2);
            img_h = obj.img_sz(1);

            fx = (-img_w/2:img_w/2-1) / (obj.pix_size * img_w);
            fy = (-img_h/2:img_h/2-1) / (obj.pix_size * img_h);

            [fx,fy] = meshgrid(fx,fy);

            obj.fz = sqrt(1 - obj.lambda.^2 .* (fx.^2 + fy.^2));
            mask = (obj.lambda^2 * (fx.^2 + fy.^2)) < 1;

            k = 2 * pi /obj.lambda;
            
            obj.pupil = (exp(1i * k .* obj.fz .* obj.toz  .* mask) .* mask);
            obj.mask = mask;
            obj.fz = k.*obj.fz;
            obj.pupil = gpuArray(single(obj.pupil));
            obj.pupil = fftshift(obj.pupil);
        end
        

        function prop = set_propagation(obj)
            fft2_dl = @(x) fft(fft(x,[],1),[],2);
            ifft2_dl = @(x) ifft(ifft(x,[],1),[],2);
            prop = @(x) ifft2_dl(fft2_dl(x) .* obj.pupil);
        end
    end
end
