clear all; clc; clear;

%% Make FileDatastores

jonah_path = "/run/user/1000/gvfs/smb-share:server=ripoff.math.colostate.edu,share=eit/Anatomical_Atlas_3D/Babies_GE";

[train_ds, val_ds, test_ds] = makeDatastores(jonah_path, 0.75, 0.15);

save("datastores.mat", "train_ds", "val_ds", "test_ds");

%% Make ArrayDatastores from FileDatastores

load("datastores.mat", "train_ds", "val_ds", "test_ds");

[train_ds, val_ds, test_ds] = makeArrayDatastores(train_ds, val_ds, test_ds);

raw_train = readall(train_ds);
raw_val = readall(val_ds);
raw_test = readall(test_ds);

% For some reason the arrayDatastores have issues saving. Save readall, and
% remake arrayDatastores with simple code.
save("raw_data.mat","raw_train","raw_val","raw_test","-7.3");
clear raw_train raw_val raw_test;

%% Make ArrayDatastores from raw_data.mat
load("raw_data.mat","raw_train","raw_val","raw_test");

train_ds = arrayDatastore(raw_train, ...
    IterationDimension=1, ...
    OutputType="same");

val_ds = arrayDatastore(raw_val, ...
    IterationDimension=1, ...
    OutputType="same");

test_ds = arrayDatastore(raw_test, ...
    IterationDimension=1, ...
    OutputType="same");

clear raw_train raw_val raw_test;

%% Convert to difference images

train_ds = transform(train_ds,@makeDiffImg);
val_ds = transform(val_ds,@makeDiffImg);
test_ds = transform(test_ds,@makeDiffImg);

%% Make miniBatchQueues (and add time dim to classification vectors for seq-to-seq)

% About n MB for a n=10 batch size w/ seq-to-seq
miniBatchSize = 10;

[mbqTrain, mbqVal, mbqTest] = makeMBQs( ...
    train_ds, val_ds, test_ds, ...
    miniBatchSize, "SSTSBC", "BCT");

%% Define network
% With Deep Network Designer, use workspace instead of this
layers = [
    
    ];

%% Save network definition

% Comment for Deep Network Designer workspace stuff
%net = dlnetwork(layers);

% Adjust me for each model!
model_name = '3d_cnn_lstm_seq_to_seq';

% Save untrained network
filename = sprintf('untrained_models/%s.mat', model_name);
save(filename,"net")

%% Train a network

% Adjust me for each model!
model_name = '3d_cnn_lstm_seq_to_seq';

% Load untrained network
filename = sprintf('untrained_models/%s.mat', model_name);
load(filename,"net")

options = trainingOptions("adam", ...
    MaxEpochs=1000, ...
    Metrics = ["accuracy"], ...
    InitialLearnRate=0.001, ...
    MiniBatchSize=miniBatchSize, ...
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
testAccuracy = testnet(train_net, mbqTest, "accuracy");
fprintf('Test Accuracy: %.2f%%\n', testAccuracy);

% 92.80% for 3d_cnn_lstm.mat

