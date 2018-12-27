classdef Sep < dagnn.ElementWise
  %   DagNN separation layer
  %
  %   This layer divides the input tensor into two sections in the channel direction.
  %
  %   e.g. If the input tensor is of size 128x128x64 and num = 24, 
  %   the tensor is divided into the first section of size 128x128x24 and
  %   the second section of size 128x128x40.

  properties
    num = 3;
  end

  methods
    function outputs = forward(obj, inputs, params)
        input=inputs{1};
        outputs{1} = input(:,:,1:obj.num,:);
        outputs{2} = input(:,:,obj.num+1:end,:);
        
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = cat(3,derOutputs{1},derOutputs{2});
      derParams = {} ;
    end

    function obj = Sep(varargin)
        obj.num=obj.num;
      obj.load(varargin) ;
    end
  end
end
