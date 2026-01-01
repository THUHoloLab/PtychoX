function [imgCube,imgSz] = init_loadImgs(folder,img_all,if_pre)

if ~if_pre
figure;
[temp,rect] = imcrop(imread([folder,'/imgs/img_3.tif']));
if rem(size(temp,1),2) == 1
    rect(4) = rect(4) - 1;
end
if rem(size(temp,2),2) == 1
    rect(3) = rect(3) - 1;
end
pix = fix((rect(4) + rect(3))/2);
pix = pix + mod(pix,2);
rect = fix(rect);
save loc_pos rect pix
else
load loc_pos.mat
end

close all;
frame = 0;
for con = 3:img_all
    frame = frame + 1;
    % temp = imread(['img_many/img_',num2str(con),'.png']);
    imgCube(:,:,frame) = single(imread([folder,'/imgs/img_',num2str(con),'.tif'], ...
                          'PixelRegion',{[rect(2),rect(2)+pix-1],...
                                         [rect(1),rect(1)+pix-1]}));

    fprintf("loading images %d/%d \n",frame,img_all);
end
clc
disp("done!");
imgCube = gpuArray(imgCube);
imgCube = sqrt(imgCube);
imgCube = imgCube - min(imgCube(:));
imgCube = imgCube / max(imgCube(:));

imgSz = [size(imgCube,1),size(imgCube,2)];
end