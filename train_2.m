function train_2(varargin)
%%%==== 2nd Training Phase ====%%%
% Train the additional convolution layers in the LDR decomposition part.
% Freeze the weights in the feature extraction & HDR reconstruction part
% with pre-trained values from the 1st phase.
%%%============================%%%

% load data & label
data = load('data\SDR_data.mat') ;
label=load('data\HDR_data.mat') ;
imdb.images.data = data.SDR;
imdb.images.label = label.HDR;
imdb.images.set = cat(2, ones(1, size(data.SDR, 4)-500), 2*ones(1, 500));

% set CNN model
net = net_main();
% set the learning rate and weight decay for biases
for i = 2:2:6
    net.params(i).learningRate = 0.1;
    net.params(i).weightDecay = 0;
end
% initialize weights with pre-trained values & set learning rate to 0
netstruct = load('./net/net_base/net-epoch-900.mat');
net_init = dagnn.DagNN.loadobj(netstruct.net);
for j = 7:24
    net.params(j).learningRate = 0;
    net.params(j).weightDecay = 0;
    net.params(j).value = net_init.params(j-6).value;
end
net.conserveMemory = true;

% options
opts.solver = @adam;
opts.train.batchSize = 32;
opts.train.continue = false; 
opts.train.gpus = 1;
opts.train.prefetch = false ;
opts.train.expDir = './net/net_main_1' ; 
opts.train.learningRate = [1e-4*ones(1, 700) 1e-5*ones(1, 100) 1e-6*ones(1, 100)];
opts.train.weightDecay = 0.0005;
opts.train.numEpochs = numel(opts.train.learningRate) ;
opts.train.derOutputs = {'objective', 1} ;
[opts, ~] = vl_argparse(opts.train, varargin) ;

%record
if(~isdir(opts.expDir))
    mkdir(opts.expDir);
end

% call training function
[net,info] = cnn_train_dag(net, imdb, @getBatch, opts) ;

function inputs = getBatch(imdb, batch, opts)
image = imdb.images.data(:, :, :, batch);
label = imdb.images.label(:, :, :, batch);

image = single(image)/255;
label = single(label)/1023;
inputs = {'input', gpuArray(image), 'label', gpuArray(label)} ;