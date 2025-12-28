function loss = hessian_loss(dW,type)

dwdx = dW(:,[2:end,1],:) - dW;
dwdy = dW([2:end,1],:,:) - dW;



switch type
    case 'isotropic'
        dwdxx = dwdx(:,[2:end,1],:) - dwdx;
        dwdyy = dwdy([2:end,1],:,:) - dwdy;
        dwdxy = dwdx([2:end,1],:,:) - dwdx;

        loss = sqrt(abs(dwdxx).^2 + abs(dwdyy).^2 + 2 * abs(dwdxy).^2);
    case 'anisotropic' 
        dwdxx = dwdx(:,[2:end,1],:) - dwdx;
        dwdyy = dwdy([2:end,1],:,:) - dwdy;
        dwdxy = dwdx([2:end,1],:,:) - dwdx;
        
        loss = abs(dwdxx) + abs(dwdyy) + 2 * abs(dwdxy);
    case 'L2'
        loss = abs(dwdx).^2 + abs(dwdy).^2;
    otherwise
    error("parameter #3 should be a string either 'isotropic', or 'anisotropic'")
end
loss = sum(loss(:));
end
