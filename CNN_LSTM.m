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

%% 3. Create the arrayDatastores
dsXTrain = arrayDatastore(X_train, "IterationDimension", 1);
dsYTrain = arrayDatastore(Y_train, "IterationDimension", 1);
dsXVal   = arrayDatastore(X_val,   "IterationDimension", 1);
dsYVal   = arrayDatastore(Y_val,   "IterationDimension", 1);
dsXTest  = arrayDatastore(X_test,  "IterationDimension", 1);
dsYTest  = arrayDatastore(Y_test,  "IterationDimension", 1);

% Combine before applying transforms so cropTimeSegment sees {X, Y}
train_ds = combine(dsXTrain, dsYTrain);
val_ds   = combine(dsXVal,   dsYVal);
test_ds  = combine(dsXTest,  dsYTest);

%clear dsXTrain dsYTrain dsXVal dsYVal dsXTest dsYTest;

% Would normally use this, but over network it is too slow
%load intubation_video_datastores.mat train_ds val_ds test_ds

%% 5. Make mini-batches
miniBatchSize = 16;

% Training (allow partial batches or return them)
mbqTrain = minibatchqueue(train_ds,miniBatchSize=miniBatchSize);

% Validation (discard partial batches for clean validation)
mbqVal = minibatchqueue(val_ds,miniBatchSize=miniBatchSize);

% Test (discard partial batches or keep, choose one)
mbqTest = minibatchqueue(test_ds,miniBatchSize=miniBatchSize);

%% 6. Define network
labels = ["Reg_Intubate", "Left_Intubate", "Right_Intubate", "Esoph_Intubate"];
numClasses = length(labels);

numHiddenUnits = 128;

layers = [
    imageInputLayer([32 31 1],"Name","input")

    % --- Spatial Encoder (CNN) ---
    % These layers apply to each time step independently
    convolution2dLayer([8 1],32,"Name","conv1","Padding","same","Stride",[2 1])
    batchNormalizationLayer("Name","bn1")
    reluLayer("Name","relu1")
    
    maxPooling2dLayer([4 1],"Name","maxpool1","Padding","same","Stride",[4 1])

    convolution2dLayer([4 1],64,"Name","conv2","Padding","same","Stride",[4 1])
    batchNormalizationLayer("Name","bn2")
    reluLayer("Name","relu2")

    % --- Reduce Dimensionality ---
    globalAveragePooling2dLayer("Name","gapool")
    flattenLayer("Name","flatten")

    % --- Temporal Processor (RNN) ---
    % Processes the sequence of 64-feature vectors, outputs a single feature vector
    lstmLayer(numHiddenUnits,"Name","lstm","OutputMode","last")
    
    % --- Classification Head ---
    fullyConnectedLayer(numClasses,"Name","fc")
    softmaxLayer("Name","softmax")];

%% 7. Train network
options = trainingOptions("adam", ...
    MaxEpochs=30, ...
    InitialLearnRate=0.001, ...
    ValidationData=mbqVal, ...
    ValidationFrequency=150, ... % Validate every 150 iterations
    Plots="training-progress", ...
    Verbose=false, ...
    Shuffle="every-epoch");

% Train the network
[net,info] = trainnet(mbqTrain, layers, "crossentropy", options);

%% 8. Evaluate
testAccuracy = testnet(net, mbqTest, "accuracy");
fprintf('Test Accuracy: %.2f%%\n', testAccuracy * 100);
