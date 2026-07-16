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

[mbqTrain, mbqVal, mbqTest] = makeVoltageMBQs(train_ds, val_ds, test_ds, 16);

clear train_ds val_ds test_ds;

%% Define networks

default_resnet = resnetNetwork([32 31 1],4,...
    "InitialPoolingLayer","none",...
    "InitialStride",1);

%% Save an untrained network

% Comment for Deep Network Designer workspace stuff
%net = dlnetwork(layers);

% Adjust me for each model!
model_name = 'default_resnet_lstm';

% Save untrained network
filename = sprintf('untrained_models/%s.mat', model_name);
save(filename,"net")

%% Train a network

% Adjust me for each model!
model_name = 'default_resnet_lstm';

% Load untrained network
filename = sprintf('untrained_models/%s.mat', model_name);
load(filename,"net")

options = trainingOptions("adam", ...
    MaxEpochs=100, ...
    Metrics = ["accuracy"], ...
    InitialLearnRate=0.025, ...
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

