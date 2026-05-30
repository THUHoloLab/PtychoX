function loss = tv_loss(dW,type)
RxdW = [dW(:,2:end,:) - dW(:,1:end-1,:), dW(:,1,:) - dW(:,end,:)];
RydW = [dW(2:end,:,:) - dW(1:end-1,:,:); dW(1,:,:) - dW(end,:,:)];

switch type
    case 'isotropic'
        loss = sqrt(abs(RxdW).^2 + abs(RydW).^2);
    case 'anisotropic' 
        loss = abs(RxdW) + abs(RydW);
    case 'L2'
        loss = abs(RxdW).^2 + abs(RydW).^2;
    otherwise
    error("parameter #3 should be a string either 'isotropic', or 'anisotropic'")
end
loss = sum(loss(:));
end
