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

[mbqTrain, mbqVal, mbqTest] = makeVoltageMBQs(train_ds, val_ds, test_ds, 32);

clear train_ds val_ds test_ds;

%% Make minibatchqueues for frame-wise data
% voltage_raw_frame_data.mat is just the readall of
% voltage_arraydatastores.mat with each row being split into 39 new rows
% for each frame in the video (with the label one-hot vector being copied
% for each new row). The code to make this file is not included here.

minibatchsize = 200;

load("data/voltage_raw_frame_data.mat","train_expanded","val_expanded","test_expanded")

% Create the arrayDatastores
train_ds = arrayDatastore(train_expanded, ...
    IterationDimension=1, ...
    OutputType="same");
val_ds   = arrayDatastore(val_expanded, ...
    IterationDimension=1, ...
    OutputType="same");
test_ds  = arrayDatastore(test_expanded, ...
    IterationDimension=1, ...
    OutputType="same");

[mbqTrain, mbqVal, mbqTest] = makeVoltageMBQsFrameWise(train_ds, val_ds, test_ds, minibatchsize);

%% Define networks
net = resnetNetwork([32 31 1],4,...
    "InitialPoolingLayer","none",...
    "InitialStride",1);

%% Save an untrained network

% Comment for Deep Network Designer workspace stuff
%net = dlnetwork(layers);

% Adjust me for each model!
model_name = 'fw_default_resnet';

% Save untrained network
filename = sprintf('untrained_models/%s.mat', model_name);
save(filename,"net")

%% Train a network

% Adjust me for each model!
model_name = 'fw_default_resnet';

% Load untrained network
filename = sprintf('untrained_models/%s.mat', model_name);
load(filename,"net")

options = trainingOptions("adam", ...
    MaxEpochs=1000, ...
    Metrics = ["accuracy"], ...
    InitialLearnRate=0.025, ...
    MiniBatchSize=minibatchsize, ...
    ValidationData=mbqVal, ...
    ValidationFrequency=250, ...
    Plots="training-progress", ...
    Shuffle="every-epoch" ...
);

[train_net,train_info] = trainnet(mbqTrain,net,"crossentropy",options);

% Save
filename = sprintf('trained_models/%s.mat', model_name);
save(filename, 'train_net', 'train_info')

%% Test network

% TODO: Confusion chart and other better tests
testAccuracy = testnet(net, mbqTest, "accuracy");
fprintf('Test Accuracy: %.2f%%\n', testAccuracy);

