% ======================= settings ======================== %
factor = 4; % dividing factor
width = 3840; % frame width
height = 2160; % frame height
fr_end = 1; % end frame number
gpu_flag = 'gpu'; % 'gpu' or 'cpu'
yuv_format = '420'; % '400' or '411' or '420' or '422' or '444'
file_SDR = 'J:\QT_seq_YUV\QT-4031_Jojakko-jiTemple_709_3840x2160_420p.yuv'; % location of SDR video file
file_pred = 'ITM-CNN_prediction.yuv'; % new file
% ========================================================= %
fclose(fopen(file_pred,'w')); % file init
[fwidth,fheight] = getformatfactor(format);

pred_ch=zeros(height/factor,width/factor,3,factor*factor);
pred_full=zeros(height,width,3);

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
for fr = 1000 %:fr_end
    SDR = single(yuv_load(file_SDR, fr, width, height, fwidth, fheight, 'SDR'))/255;
    if strcmp(gpu_flag, 'gpu')
        SDR = gpuArray(SDR);
    end
    for h=1:factor
        for w=1:factor
            st_h=(h-1)*height/factor;
            e_h=h*height/factor;
            st_w=(w-1)*width/factor;
            e_w=w*width/factor;
            SDR_block=SDR(st_h+1:e_h,st_w+1:e_w,:);
            net.eval({'input',SDR_block});
            pred_ch(:,:,:,(h-1)*factor+w)=gather(net.vars(pred_index).value);
        end
    end
    for f=1:factor*factor
        q=floor((f-1)/factor)+1;
        r=mod(f, factor);
        if r==0, r=factor; end
        pred_full((q-1)*height/factor+1:q*height/factor,(r-1)*width/factor+1:r*width/factor,:)=pred_ch(:,:,:,f);
    end
    pred=uint16(pred_full*1023);
    yuv_save(pred, file_pred, fwidth, fheight, 'HDR');
end


