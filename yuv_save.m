%%% save YUV to file
function yuv_save(data, video_file, factor_w, factor_h, SDR_HDR)

% get data size
height = size(data, 1);
width = size(data, 2);
channels = size(data, 3);

%open file
fid = fopen(video_file,'a');

% subsampling of U and V
if channels == 2 || factor_h == 0
    %4:0:0
    y(1:height,1:width) = data(:,:,1);
elseif channels == 3
    y(1:height,1:width) = double(data(:,:,1));
    u(1:height,1:width) = double(data(:,:,2));
    v(1:height,1:width) = double(data(:,:,3));
    if factor_w == 1
        %4:1:1
        u2 = u;
        v2 = v;
    elseif factor_h == 0.5
        %4:2:0
        u2(1:height/2,1:width/2) = u(1:2:end,1:2:end)+u(2:2:end,1:2:end)+u(1:2:end,2:2:end)+u(2:2:end,2:2:end);
        u2                         = u2/4;
        v2(1:height/2,1:width/2) = v(1:2:end,1:2:end)+v(2:2:end,1:2:end)+v(1:2:end,2:2:end)+v(2:2:end,2:2:end);
        v2                         = v2/4;
    elseif factor_w == 0.25
        %4:1:1
        u2(1:height,1:width/4) = u(:,1:4:end)+u(:,2:4:end)+u(:,3:4:end)+u(:,4:4:end);
        u2                       = u2/4;
        v2(1:height,1:width/4) = v(:,1:4:end)+v(:,2:4:end)+v(:,3:4:end)+v(:,4:4:end);
        v2                       = v2/4;
    elseif factor_w == 0.5 && factor_h == 1
        %4:2:2
        u2(1:height,1:width/2) = u(:,1:2:end)+u(:,2:2:end);
        u2                       = u2/2;
        v2(1:height,1:width/2) = v(:,1:2:end)+v(:,2:2:end);
        v2                       = v2/2;
    end
end

if strcmp(SDR_HDR,'HDR')
    fwrite(fid,uint16(y'),'uint16'); % write Y data

    if factor_h ~= 0
        % write U & V data
        fwrite(fid,uint16(u2'),'uint16');
        fwrite(fid,uint16(v2'),'uint16');
    end
elseif strcmp(SDR_HDR,'SDR')
        fwrite(fid,uint8(y'),'uchar'); % write Y data

    if factor_h ~= 0
        % write U & V data
        fwrite(fid,uint8(u2'),'uchar');
        fwrite(fid,uint8(v2'),'uchar');
    end
end

fclose(fid);
