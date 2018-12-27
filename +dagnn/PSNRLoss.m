classdef PSNRLoss < dagnn.Loss

    methods
        function outputs = forward(obj, inputs, params)
%             outputs{1} = psnr(double(inputs{1}),double(inputs{2}),1);
            imdff = double(inputs{1}) - double(inputs{2});
            rmse = sqrt(mean(mean(mean(imdff.^2,1),2),3));
            psnr = 20*log10(1/rmse);
            outputs{1} = mean(squeeze(psnr));
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + size(inputs{1},4)*double(gather(outputs{1}))) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs,params, derOutputs)
            
            Y = gather(bsxfun(@minus,inputs{1},inputs{2}));
            Y(Y>1)= 1;  % x-y>1
            Y(Y<-1) = -1; % y-x<1

            derInputs{1} = gpuArray(bsxfun(@times, derOutputs{1},Y));
            derInputs{2} = [] ;
            derParams = {} ;
        end
        
        function obj = PSNRLoss(varargin)
            obj.load(varargin) ;
        end
    end
end
