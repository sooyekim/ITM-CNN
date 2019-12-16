function train_3(varargin)
%%%==== 3rd Training Phase ====%%%
% Train the whole network jointly, with weights initialized with values
% from the 2nd training phase.
%%%============================%%%

% load data & label
data = load('data\SDR_data.mat') ;
label=load('data\HDR_data.mat') ;
imdb.images.data = data.SDR;
imdb.images.label = label.HDR;
imdb.images.set = cat(2, ones(1, size(data.SDR, 4)-500), 2*ones(1, 500));

% set CNN model
net = net_main();
% get pre-trained values
netstruct = load('./net/net_main_1/net-epoch-900.mat');
net_init = dagnn.DagNN.loadobj(netstruct.net);
for i = 2:2:24
    net.params(i).value = net_init.params(i).value; % initialize with pre-trained values
    if mod(i, 2) == 0 % set the learning rate and weight decay for biases
        net.params(i).learningRate = 0.1;
        net.params(i).weightDecay = 0;
    end
end
net.conserveMemory = true;

% options
opts.solver = @adam;
opts.train.batchSize = 32;
opts.train.continue = false; 
opts.train.gpus = 1;
opts.train.prefetch = false ;
opts.train.expDir = './net/net_main_2' ; 
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