clear
clc
close all

load Intubation_Datastores.mat Train_ds Val_ds
return
labels = ["Reg_Intubate", "Left_Intubate", "Right_Intubate", "Esoph_Intubate"];
numClasses = length(labels);

Train_ds_mb = transform(Train_ds, @minibatchFcn);
Val_ds_mb   = transform(Val_ds,   @minibatchFcn);

%% ---------------------------------------------------------------------- %
%                            Network Definition                           %
% ----------------------------------------------------------------------- %

fprintf("Defining Network & Options\n")
layers = [
    inputLayer([32 31 1 NaN],"SSCB","Name","input")
    flattenLayer("Name","flatten")

    fullyConnectedLayer(992,"Name","fc1")
    leakyReluLayer(0.01,"Name","leakyrelu1")

    fullyConnectedLayer(496,"Name","fc2")
    leakyReluLayer(0.01,"Name","leakyrelu2")

    fullyConnectedLayer(248,"Name","fc3")
    leakyReluLayer(0.01,"Name","leakyrelu3")

    fullyConnectedLayer(4,"Name","fc4")
    softmaxLayer("Name","softmax")];

net = dlnetwork(layers);

%% ---------------------------------------------------------------------- %
%                             Training Options                            %
% ----------------------------------------------------------------------- %

options = trainingOptions("adam", ...
                          InitialLearnRate=1e-3, ...
                          MaxEpochs=40, ...
                          MiniBatchSize=1, ...
                          Shuffle="never", ...
                          ValidationData=Val_ds_mb, ...
                          ValidationFrequency=200, ...
                          Plots="training-progress", ...
                          Verbose=true);

%% ---------------------------------------------------------------------- %
%                             Training Options                            %
% ----------------------------------------------------------------------- %

fprintf("Starting Training\n")
[net, info] = trainnet(Train_ds_mb,net,"crossentropy",options);
save("network.mat", "net", "info")

%% ---------------------------------------------------------------------- %
%                             Custom Functions                            %
% ----------------------------------------------------------------------- %

function dataOut = minibatchFcn(dataIn)
    % dataIn is an N×2 cell array: {X, Y}
    X = cat(4, dataIn{:,1});   % concatenate along 4th dim (batch)
    Y = categorical([dataIn{:,2}]);

    % Convert to dlarray for GPU efficiency
    X = dlarray(single(X), "SSCB");   % S=spatial dims, C=channel, B=batch

    % Move to GPU if available
    if canUseGPU
        X = gpuArray(X);
    end

    dataOut = {X, Y};
end
