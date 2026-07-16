clear
clc
close all
rng(1)

%% Download, format, and save data (run once to localize data!)

jonah_machine_path = '/run/user/1000/gvfs/smb-share:server=ripoff.math.colostate.edu,share=eit/Anatomical_Atlas_3D/Babies_GE';

% Make SMB network datastores
[train_ds, val_ds, test_ds] = makeVoltageFileDatastores( ...
    jonah_machine_path, ...
    0.7, ...
    0.2, ...
    1);

% Load into memory
[train_ds, val_ds, test_ds] = makeVoltageArrayDatastores(train_ds, val_ds, test_ds);

save("data/voltage_arraydatastores.mat", "train_ds", "val_ds", "test_ds")

%% Make minibatchqueues

load("data/voltage_arraydatastores.mat", "train_ds", "val_ds", "test_ds")

[mbqTrain, mbqVal, mbqTest] = makeVoltageMBQs(train_ds, val_ds, train_ds, 16);

%% Define a network

% % Got stuck at loss of 1.6. Validation loss of 1.3322.
% layers = [
%     sequenceInputLayer([32 31 1],"Name","input")
% 
%     % --- Spatial Encoder (CNN) ---
%     % These layers apply to each time step independently
%     convolution2dLayer([8 1],32,"Name","conv1","Padding","same","Stride",[2 1])
%     batchNormalizationLayer("Name","bn1")
%     reluLayer("Name","relu1")
% 
%     maxPooling2dLayer([4 1],"Name","maxpool1","Padding","same","Stride",[4 1])
% 
%     convolution2dLayer([4 1],64,"Name","conv2","Padding","same","Stride",[4 1])
%     batchNormalizationLayer("Name","bn2")
%     reluLayer("Name","relu2")
% 
%     % --- Reduce Dimensionality ---
%     globalAveragePooling2dLayer("Name","gapool")
%     flattenLayer("Name","flatten")
% 
%     % --- Temporal Processor (RNN) ---
%     % Processes the sequence of 64-feature vectors, outputs a single feature vector
%     lstmLayer(numHiddenUnits,"Name","lstm","OutputMode","last")
% 
%     % --- Classification Head ---
%     fullyConnectedLayer(numClasses,"Name","fc")
%     softmaxLayer("Name","softmax")];

% layers = [
%     sequenceInputLayer([32 31 1],"Name","input")
% 
%     convolution2dLayer([4 4],64,"Name","conv1","Padding","same","Stride",[2 1])
%     batchNormalizationLayer("Name","bn1")
%     reluLayer("Name","relu1")
%     maxPooling2dLayer([2 1],"Name","maxpool1","Padding","same","Stride",[2 1])
% 
%     convolution2dLayer([4 4],128,"Name","conv2","Padding","same","Stride",[2 1])
%     batchNormalizationLayer("Name","bn2")
%     reluLayer("Name","relu2")
%     maxPooling2dLayer([2 1],"Name","maxpool2","Padding","same","Stride",[2 1])
% 
%     convolution2dLayer([2 8],256,"Name","conv3","Padding","same","Stride",[2 1])
%     batchNormalizationLayer("Name","bn3")
%     reluLayer("Name","relu3")
%     globalAveragePooling2dLayer("Name","gapool")
% 
%     flattenLayer("Name","flatten")
% 
%     lstmLayer(128,"Name","lstm","OutputMode","last")
%     fullyConnectedLayer(4,"Name","fc")
%     softmaxLayer("Name","softmax")
% ]; 

% Useful for workspace modifications
layers = [
    sequenceInputLayer([32 31 1],"Name","sequence")
    convolution2dLayer([3 3],64,"Name","conv1","Padding","same")
    batchNormalizationLayer("Name","batchnorm1")
    leakyReluLayer(0.01,"Name","leakyrelu1")
    maxPooling2dLayer([4 1],"Name","maxpool1","Padding","same","Stride",[2 1])
    convolution2dLayer([3 3],128,"Name","conv2","Padding","same")
    batchNormalizationLayer("Name","batchnorm2")
    leakyReluLayer(0.01,"Name","leakyrelu2")
    maxPooling2dLayer([4 1],"Name","maxpool2","Padding","same","Stride",[2 1])
    convolution2dLayer([3 3],256,"Name","conv3","Padding","same")
    batchNormalizationLayer("Name","batchnorm3")
    leakyReluLayer(0.01,"Name","leakyrelu3")
    maxPooling2dLayer([4 1],"Name","maxpool3","Padding","same","Stride",[2 1])
    convolution2dLayer([3 3],512,"Name","conv4","Padding","same")
    batchNormalizationLayer("Name","batchnorm4")
    leakyReluLayer(0.01,"Name","leakyrelu4")
    maxPooling2dLayer([4 1],"Name","maxpool4","Padding","same","Stride",[4 1])
    convolution2dLayer([1 31],64,"Name","conv5","Padding","same","Stride",[1 31])
    batchNormalizationLayer("Name","batchnorm5")
    leakyReluLayer(0.01,"Name","leakyrelu5")
    flattenLayer("Name","flatten")
    lstmLayer(128,"Name","lstm","OutputMode","last")
    fullyConnectedLayer(4,"Name","fc")
    softmaxLayer("Name","softmax")
];

%% Save an untrained network

net = dlnetwork(layers);

% Adjust me for each model!
model_name = 'deep_cnn_lstm';

% Save untrained network
filename = sprintf('untrained_models/%s.mat', model_name);
save(filename,"net")

%% Train a network

% Adjust me for each model!
model_name = 'deep_cnn_lstm';

% Load untrained network
filename = sprintf('untrained_models/%s.mat', model_name);
load(filename,"net")

options = trainingOptions("adam", ...
    MaxEpochs=10000, ...
    Metrics = ["accuracy"], ...
    InitialLearnRate=0.001, ...
    MiniBatchSize=16, ...
    ValidationData=mbqVal, ...
    ValidationFrequency=250, ...
    Plots="training-progress", ...
    Shuffle="every-epoch" ...
);

[net,info] = trainnet(mbqTrain,net,"crossentropy",options);

% Save
filename = sprintf('trained_models/%s.mat', model_name);
save(filename, 'net', 'info')

%% Test network

% TODO: Confusion chart and other better tests
testAccuracy = testnet(net, mbqTest, "accuracy");
fprintf('Test Accuracy: %.2f%%\n', testAccuracy);

