clear
clc
close all
rng(42)

%% 1. Load data into memory
load("intubation_video_data.mat", "all_train", "all_val", "all_test");

%% 2. Unpack and format data
[X_train, Y_train] = formatReadall(all_train);
[X_val, Y_val]     = formatReadall(all_val);
[X_test, Y_test]   = formatReadall(all_test);

% One-hot encodding
labels = ["Reg_Intubate", "Left_Intubate", "Right_Intubate", "Esoph_Intubate"];
numClasses = length(labels);

Y_train = cellfun(@full, ind2vec(Y_train, numClasses), UniformOutput=false);
Y_val = cellfun(@full, ind2vec(Y_val, numClasses), UniformOutput=false);
Y_test = cellfun(@full, ind2vec(Y_test, numClasses), UniformOutput=false);

%% 3. Create the arrayDatastores
train_ds = arrayDatastore(cat(2, X_train, Y_train), ...
    IterationDimension=1, ...
    OutputType="same");
val_ds   = arrayDatastore(cat(2, X_val, Y_val), ...
    IterationDimension=1, ...
    OutputType="same");
test_ds  = arrayDatastore(cat(2, X_test, Y_test), ...
    IterationDimension=1, ...
    OutputType="same");

% Would normally use this, but over network it is too slow. My machine has
% enough RAM for this to run well enough.
%load intubation_video_datastores.mat train_ds val_ds test_ds

%% 5. Make mini-batches
miniBatchSize = 16;

% MATLAB magic: Format as "SSTBC" to get 32(S) × 31(S) × 1(C) × 16(B) × 39(T)
mbqTrain = minibatchqueue(train_ds, ...
    MiniBatchSize=miniBatchSize, ...
    OutputAsDlarray=[1 1], ...
    MiniBatchFormat=["SSTBC" "CB"], ...
    OutputEnvironment=["auto" "auto"]);
mbqVal = minibatchqueue(val_ds, ...
    MiniBatchSize=miniBatchSize, ...
    OutputAsDlarray=[1 1], ...
    MiniBatchFormat=["SSTBC" "CB"], ...
    OutputEnvironment=["auto" "auto"]);
mbqTest = minibatchqueue(test_ds, ...
    MiniBatchSize=miniBatchSize, ...
    OutputAsDlarray=[1 1], ...
    MiniBatchFormat=["SSTBC" "CB"], ...
    OutputEnvironment=["auto" "auto"]);

% Clear up memory before training
clear X_train Y_train X_val Y_val X_test Y_test

%% 6. Define network
numHiddenUnits = 128;

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

layers = [
    sequenceInputLayer([32 31 1],"Name","input")
    fullyConnectedLayer(992,"Name","fc1")
    leakyReluLayer(0.01,"Name","leakyrelu1")
    fullyConnectedLayer(496,"Name","fc2")
    leakyReluLayer(0.01,"Name","leakyrelu2")
    fullyConnectedLayer(128,"Name","fc3")
    leakyReluLayer(0.01,"Name","leakyrelu")
    flattenLayer("Name","flatten")
    lstmLayer(128,"Name","lstm","OutputMode","last")
    fullyConnectedLayer(4,"Name","fc")
    softmaxLayer("Name","softmax")
];

%% 7. Train network
options = trainingOptions("adam", ...
    MaxEpochs=120, ...
    Metrics = ["accuracy"], ...
    InitialLearnRate=0.001, ...
    MiniBatchSize=16, ...
    ValidationData=mbqVal, ...
    ValidationFrequency=250, ...
    Plots="training-progress", ...
    Shuffle="every-epoch");

% Train the network
[net,info] = trainnet(mbqTrain, layers, ...
    @(Y, T) crossentropy(Y,T, ClassificationMode="multilabel"), ...
    options);

% Create timestamp and filename (safe for filenames)
timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
filename = sprintf('cnn_lstm_network_%s.mat', timestamp);

% Save
save(filename, 'net', 'info')

%% 8. Evaluate
testAccuracy = testnet(net, mbqTest, "accuracy");
fprintf('Test Accuracy: %.2f%%\n', testAccuracy);
