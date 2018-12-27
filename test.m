% ======================= settings ======================== %
width = 3840; % frame width
height = 2160; % frame height
fr_end = 1; % end frame number
gpu_flag = 'gpu'; % 'gpu' or 'cpu'
yuv_format = '420'; % '400' or '411' or '420' or '422' or '444'
file_SDR = 'C:/test_SDR_video.yuv'; % location of SDR video file
file_pred = 'ITM-CNN_prediction.yuv'; % new file
% ========================================================= %
fclose(fopen(file_pred,'w')); % file init
[fwidth,fheight] = getformatfactor(yuv_format);

% load trained weights
netstruct = load('./net/ITM-CNN_weights.mat');
net = dagnn.DagNN.loadobj(netstruct.net);
if strcmp(gpu_flag, 'cpu')
    move(net,'cpu');
elseif strcmp(gpu_flag, 'gpu')
    move(net,'gpu');
end
net.mode = 'test' ;
pred_index = net.getVarIndex('pred');

% run
for fr = 1:fr_end
    SDR = single(yuv_load(file_SDR, fr, width, height, fwidth, fheight, 'SDR'))/255;  
    if strcmp(gpu_flag, 'gpu')
        SDR = gpuArray(SDR);
    end
    net.eval({'input',SDR});
    pred = gather(net.vars(pred_index).value);
    pred = uint16(pred*1023);
    yuv_save(pred, file_pred, fwidth, fheight, 'HDR');
end


