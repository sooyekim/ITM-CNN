function net = net_main()
%%%============== Initialize ================%%%
net = dagnn.DagNN();
reluBlock = dagnn.ReLU;

ch = 32;
convBlock_input = dagnn.Conv('size', [3 3 3 ch], 'hasBias', true, 'stride', [1,1], 'pad', [1,1,1,1]);
convBlock_c = dagnn.Conv('size', [3 3 ch ch], 'hasBias', true, 'stride', [1,1], 'pad', [1,1,1,1]);

%%%============== LDR Decomposition ================%%%
net.addLayer('conv1s', convBlock_input, {'input'}, {'conv1s'}, {'convs_1f', 'convs_1b'});
net.addLayer('relu1s', reluBlock, {'conv1s'}, {'conv1as'}, {});
net.addLayer('conv2s', convBlock_c, {'conv1as'}, {'conv2s'}, {'convs_2f', 'convs_2b'});
net.addLayer('relu2s', reluBlock, {'conv2s'}, {'conv2as'}, {});
conv3sBlock = dagnn.Conv('size', [3 3 32 6], 'hasBias', true, 'stride', [1,1], 'pad', [1,1,1,1]);
net.addLayer('conv3s', conv3sBlock, {'conv2as'}, {'conv3s'}, {'convs_3f', 'convs_3b'});
net.addLayer('sep', dagnn.Sep('num', 3),  {'conv3s'}, {'input_base', 'input_detail'});

%%%============== Feature Extraction (Base Layer) ================%%%
net.addLayer('conv1', convBlock_input, {'input_base'}, {'conv1'}, {'conv_1f', 'conv_1b'});
net.addLayer('relu1', reluBlock, {'conv1'}, {'conv1a'}, {});
net.addLayer('conv2', convBlock_c, {'conv1a'}, {'conv2'}, {'conv_2f', 'conv_2b'});
net.addLayer('relu2', reluBlock, {'conv2'}, {'conv2a'}, {});
net.addLayer('conv3', convBlock_c, {'conv2a'}, {'conv3'}, {'conv_3f', 'conv_3b'});

%%%============== Feature Extraction (Detail Layer) ================%%%
net.addLayer('conv1d', convBlock_input, {'input_detail'}, {'conv1d'}, {'convd_1f', 'convd_1b'});
net.addLayer('relu1d', reluBlock, {'conv1d'}, {'conv1ad'}, {});
net.addLayer('conv2d', convBlock_c, {'conv1ad'}, {'conv2d'}, {'convd_2f', 'convd_2b'});
net.addLayer('relu2d', reluBlock, {'conv2d'}, {'conv2ad'}, {});
net.addLayer('conv3d', convBlock_c, {'conv2ad'}, {'conv3d'}, {'convd_3f', 'convd_3b'});

%%%============== HDR Reconstruction ================%%%
net.addLayer('cat', dagnn.Concat('dim', 3), {'conv3', 'conv3d'}, {'cat'});
net.addLayer('relu3', reluBlock, {'cat'}, {'conv3a'}, {});

conv4Block = dagnn.Conv('size', [3 3 64 40], 'hasBias', true, 'stride', [1,1], 'pad', [1,1,1,1]);
net.addLayer('conv4', conv4Block, {'conv3a'}, {'conv4'}, {'conv_4f', 'conv_4b'});
net.addLayer('relu4', reluBlock, {'conv4'}, {'conv4a'}, {});

conv5Block = dagnn.Conv('size', [3 3 40 40], 'hasBias', true, 'stride', [1,1], 'pad', [1,1,1,1]);
net.addLayer('conv5', conv5Block, {'conv4a'}, {'conv5'}, {'conv_5f', 'conv_5b'});
net.addLayer('relu5', reluBlock, {'conv5'}, {'conv5a'}, {});

conv6Block = dagnn.Conv('size', [3 3 40 3], 'hasBias', true, 'stride', [1,1], 'pad', [1,1,1,1]);
net.addLayer('conv6', conv6Block, {'conv5a'}, {'pred'}, {'conv_6f', 'conv_6b'});

%%%============== Loss ================%%%
net.addLayer('loss', dagnn.PSNRLoss(), {'pred', 'label'}, 'objective');
net.initParams();