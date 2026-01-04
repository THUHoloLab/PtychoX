clc
clear

gif1 = 'test.gif';
gif2 = 'testPIE.gif';
    info1 = imfinfo(gif1);
    info2 = imfinfo(gif2);

    % Check consistency
    assert(numel(info1) == numel(info2), 'GIF frame numbers do not match.');
    assert(info1(1).Width  == info2(1).Width,  'GIF widths do not match.');
    assert(info1(1).Height == info2(1).Height, 'GIF heights do not match.');

    numFrames = numel(info1);

    % Use delay time from the first GIF
    if isfield(info1(1), 'DelayTime')
        delayTime = info1(1).DelayTime;
    else
        delayTime = 0.1;
    end

    frame1_all = imread(gif1, 'Frame','all');
    frame2_all = imread(gif2, 'Frame','all');

    for k = 1:2:numFrames
        k 
        % Read frames
        % frame1 = imread(gif1, k);
        % frame2 = imread(gif2, k);
        frame1 = frame1_all(:,:,:,k);
        frame2 = frame2_all(:,:,:,k);
        % Ensure RGB (robust for grayscale/RGB mix)
        if size(frame1, 3) == 1
            frame1 = repmat(frame1, [1, 1, 3]);
        end
        if size(frame2, 3) == 1
            frame2 = repmat(frame2, [1, 1, 3]);
        end
        
        % Vertical concatenation
        mergedFrame = cat(1, frame1, frame2);
        mergedFrame = imresize(mergedFrame,0.6);
        [idxx,map] = rgb2ind(mergedFrame,8);
        % Write output GIF
        if k == 1
            imwrite(idxx,map, 'compared.gif', 'gif', ...
                    'LoopCount', 65535, ...
                    'DelayTime', 0.07);
        elseif (k > 1) && (k < numFrames)
            imwrite(idxx,map, 'compared.gif', 'gif', ...
                    'WriteMode', 'append', ...
                    'DelayTime', 0.07);
        % elseif k == numFrames
     

        end
    end
    imwrite(idxx,map, 'compared.gif', 'gif', ...
                    'WriteMode', 'append', ...
                    'DelayTime', 2);