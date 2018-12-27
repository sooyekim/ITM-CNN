%%% load YUV file
function YUV = yuv_load(fileName, frame, width, height, factor_w, factor_h, SDR_HDR)
% get size of U and V
fileId = fopen(fileName,'r');
width_h = width*factor_w;
heigth_h = height*factor_h;
% compute factor for framesize
factor = 1+(factor_h*factor_w)*2;
% compute framesize
framesize = width*height;

if strcmp(SDR_HDR,'HDR')==1
    fseek(fileId,(frame-1)*factor*framesize*2, 'bof');
    % create Y-Matrix
    YMatrix = fread(fileId, width * height, 'uint16');
    YMatrix = int16(reshape(YMatrix,width,height)');
    % create U- and V- Matrix
    if factor_h == 0
        UMatrix = 0;
        VMatrix = 0;
    else
        UMatrix = fread(fileId,width_h * heigth_h, 'uint16');
        UMatrix = int16(UMatrix);
        UMatrix = reshape(UMatrix,width_h, heigth_h).';

        VMatrix = fread(fileId,width_h * heigth_h, 'uint16');
        VMatrix = int16(VMatrix);
        VMatrix = reshape(VMatrix,width_h, heigth_h).';
    end
elseif strcmp(SDR_HDR,'SDR')==1
    fseek(fileId,(frame-1)*factor*framesize, 'bof');
        % create Y-Matrix
    YMatrix = fread(fileId, width * height, 'uchar');
    YMatrix = int16(reshape(YMatrix,width,height)');
    % create U- and V- Matrix
    if factor_h == 0
        UMatrix = 0;
        VMatrix = 0;
    else
        UMatrix = fread(fileId,width_h * heigth_h, 'uchar');
        UMatrix = int16(UMatrix);
        UMatrix = reshape(UMatrix,width_h, heigth_h).';

        VMatrix = fread(fileId,width_h * heigth_h, 'uchar');
        VMatrix = int16(VMatrix);
        VMatrix = reshape(VMatrix,width_h, heigth_h).';
    end
end
% compose the YUV-matrix:
YUV(1:height,1:width,1) = YMatrix;

if factor_h == 0
    YUV(:,:,2) = 127;
    YUV(:,:,3) = 127;
end
% consideration of the subsampling of U and V
if factor_w == 1
    UMatrix1(:,:) = UMatrix(:,:);
    VMatrix1(:,:) = VMatrix(:,:);
    
elseif factor_w == 0.5
    UMatrix1(1:heigth_h,1:width) = int16(0);
    UMatrix1(1:heigth_h,1:2:end) = UMatrix(:,1:1:end);
    UMatrix1(1:heigth_h,2:2:end) = UMatrix(:,1:1:end);
    
    VMatrix1(1:heigth_h,1:width) = int16(0);
    VMatrix1(1:heigth_h,1:2:end) = VMatrix(:,1:1:end);
    VMatrix1(1:heigth_h,2:2:end) = VMatrix(:,1:1:end);
    
elseif factor_w == 0.25
    UMatrix1(1:heigth_h,1:width) = int16(0);
    UMatrix1(1:heigth_h,1:4:end) = UMatrix(:,1:1:end);
    UMatrix1(1:heigth_h,2:4:end) = UMatrix(:,1:1:end);
    UMatrix1(1:heigth_h,3:4:end) = UMatrix(:,1:1:end);
    UMatrix1(1:heigth_h,4:4:end) = UMatrix(:,1:1:end);
    
    VMatrix1(1:heigth_h,1:width) = int16(0);
    VMatrix1(1:heigth_h,1:4:end) = VMatrix(:,1:1:end);
    VMatrix1(1:heigth_h,2:4:end) = VMatrix(:,1:1:end);
    VMatrix1(1:heigth_h,3:4:end) = VMatrix(:,1:1:end);
    VMatrix1(1:heigth_h,4:4:end) = VMatrix(:,1:1:end);
end

if factor_h == 1
    YUV(:,:,2) = UMatrix1(:,:);
    YUV(:,:,3) = VMatrix1(:,:);
    
elseif factor_h == 0.5
    YUV(1:height,1:width,2) = int16(0);
    YUV(1:2:end,:,2) = UMatrix1(:,:);
    YUV(2:2:end,:,2) = UMatrix1(:,:);
    
    YUV(1:height,1:width,3) = int16(0);
    YUV(1:2:end,:,3) = VMatrix1(:,:);
    YUV(2:2:end,:,3) = VMatrix1(:,:);
    
elseif factor_h == 0.25
    YUV(1:height,1:width,2) = int16(0);
    YUV(1:4:end,:,2) = UMatrix1(:,:);
    YUV(2:4:end,:,2) = UMatrix1(:,:);
    YUV(3:4:end,:,2) = UMatrix1(:,:);
    YUV(4:4:end,:,2) = UMatrix1(:,:);
    
    YUV(1:height,1:width) = int16(0);
    YUV(1:4:end,:,3) = VMatrix1(:,:);
    YUV(2:4:end,:,3) = VMatrix1(:,:);
    YUV(3:4:end,:,3) = VMatrix1(:,:);
    YUV(4:4:end,:,3) = VMatrix1(:,:);
end

fclose(fileId);

