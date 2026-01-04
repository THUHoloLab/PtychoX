function loss = sparse_loss(dW,type)

dwdx = dW(:,[2:end,1],:) - dW;
dwdy = dW([2:end,1],:,:) - dW;



switch type
    case 'isotropic'
        loss = 1 - exp(-8.*sqrt(abs(dwdx).^2 + abs(dwdy).^2 + 1e-5));
    case 'anisotropic' 
        loss = 1 - exp(-8.*(abs(dwdx) + abs(dwdy)));
    case 'L2'
        loss = abs(dwdx).^2 + abs(dwdy).^2;
    otherwise
    error("parameter #3 should be a string either 'isotropic', or 'anisotropic'")
end
loss = sum(loss(:));
end
