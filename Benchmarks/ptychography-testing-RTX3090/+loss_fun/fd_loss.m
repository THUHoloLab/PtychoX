function loss = fd_loss(x1,x2,type)
w = x1 - x2;

loss = abs(w([2:end,1],:,:) - w) + abs(w(:,[2:end,1],:) - w);

switch type
    case 'sum'
        loss = sum(loss(:));
    case 'mean'
        loss = mean(loss(:));
    otherwise 
        error('type must be sum or mean')
end

end