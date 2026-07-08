clear
clc
close all

%load intubation_video_datastores.mat train_ds val_ds

labels = ["Reg_Intubate", "Left_Intubate", "Right_Intubate", "Esoph_Intubate"];
numClasses = length(labels);

numHiddenUnits = 128;

layers = [
    % Input layer for 32x31 single-channel images (Time is handled by dlarray)
    imageInputLayer([32 31 1], "Name", "input")

    % --- Spatial Encoder (CNN) ---
    % These layers apply to each time step independently
    convolution2dLayer(3, 16, 'Padding', 'same', "Name", "conv1")
    batchNormalizationLayer("Name", "bn1")
    reluLayer("Name", "relu1")

    convolution2dLayer(3, 32, 'Padding', 'same', 'Stride', 2, "Name", "conv2")
    batchNormalizationLayer("Name", "bn2")
    reluLayer("Name", "relu2")

    % --- Transition to Sequence ---
    % Flattens Spatial & Channel dims (16x16x32 -> 8192) while preserving Time
    flattenLayer("Name", "flatten")

    % --- Temporal Processor (RNN) ---
    % Processes the sequence of 8192-feature vectors, outputs a single feature vector
    lstmLayer(numHiddenUnits, 'OutputMode', 'last', "Name", "lstm")

    % --- Classification Head ---
    fullyConnectedLayer(numClasses, "Name", "fc")
    softmaxLayer("Name", "softmax") 
    ];

working_net = dlnetwork(layers);