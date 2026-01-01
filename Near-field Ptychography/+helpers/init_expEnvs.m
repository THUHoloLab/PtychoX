function env = init_expEnvs(mag, imgSz)
    pix_sz = 5.86 / mag;
    img_sz = imgSz * mag;

    diffractor_u = helpers.diffractor("pix_size", pix_sz,...
                          "lambda",   0.532,...
                          "toz",      5740,...
                          "img_sz",   imgSz * mag);

    imSize0 = imgSz(1);

    fx = (-img_sz(2)/2:img_sz(2)/2-1) / (img_sz(2));
    fy = (-img_sz(1)/2:img_sz(1)/2-1) / (img_sz(1));
    [FX,FY] = meshgrid(fx,fy);

    prop = diffractor_u.set_propagation();
    prop = diffractor_u.pupil;
    prop = gpuArray(single(prop));

    env.Fx = FX;
    env.Fy = FY;
    env.prop = prop;
    env.size = img_sz;
    env.pixSz = pix_sz;
    env.Fz = diffractor_u.fz;
    env.Mask = diffractor_u.mask;
    
end