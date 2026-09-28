clear all; clc; clear;

%% Make datastores

path = "/run/user/1000/gvfs/smb-share:server=ripoff.math.colostate.edu,share=eit/Anatomical_Atlas_3D/Babies_GE";
%path = input("Path: ");

[train_ds, val_ds, test_ds] = makeDatastores(path, 0.75, 0.15);
[train_ds, val_ds, test_ds] = makeArrayDatastores(train_ds, val_ds, test_ds);

% Convert to difference images
% Add time dimension to one-hot vectors

train_ds = transform(train_ds,@makeDiffImg);
val_ds = transform(val_ds,@makeDiffImg);
test_ds = transform(test_ds,@makeDiffImg);

raw_train = readall(train_ds);
raw_val = readall(val_ds);
raw_test = readall(test_ds);

save raw_data.mat raw_test raw_val raw_train;

%% Make miniBatchQueues (with contiq subseq and seq-to-seq output)

miniBatchSize = 10;

[mbqTrain, mbqVal, mbqTest] = makeMBQs( ...
    train_ds, val_ds, test_ds, ...
    miniBatchSize, "SSTSBC", "CTB");

%% Train models

models = {'diff_3d_cnn_lstm_seq_to_seq_726k',...
          'diff_3d_cnn_lstm_seq_to_seq_398k',...
          'diff_3d_cnn_lstm_seq_to_seq_183k',...
          'diff_3d_cnn_lstm_seq_to_seq_100k',...
          'diff_3d_cnn_lstm_seq_to_seq_51k',...
          'diff_3d_cnn_lstm_seq_to_seq_30k',};

for i=models
    model_name = i{1};
    
    filename = sprintf('untrained_models/%s.mat', model_name);
    
    % Remove name from loaded strut
    net = load(filename,model_name);
    net = net.(model_name);
    
    options = trainingOptions("adam", ...
        MaxEpochs=250, ...
        Metrics = "accuracy", ...
        InitialLearnRate=0.001, ...
        MiniBatchSize=miniBatchSize, ...
        ValidationData=mbqVal, ...
        ValidationFrequency=50, ...
        Plots="training-progress", ...
        Shuffle="every-epoch" ...
        );
    
    [train_net,train_info] = trainnet(mbqTrain,net,"crossentropy",options);
    
    filename = sprintf('trained_models/%s.mat', model_name);
    save(filename, 'train_net', 'train_info')

    close(train_info)
    reset(mbqTrain)
    reset(mbqVal)

end

%% Test network

labels = ["Regular Intubate", ...
          "Left Intubate", ...
          "Right Intubate", ...
          "Esoph Intubate"];

correct = 0;
total = 0;

allPredicted = [];
allTrue = [];

reset(mbqTest);

while hasdata(mbqTest)

    [X, Y] = next(mbqTest);

    YPred = predict(train_net, X);

    YPred = extractdata(YPred);
    Y = extractdata(Y);

    % Find predicted class for each frame
    [~, predictedClass] = max(YPred, [], 1);

    % Find true class for each frame
    
    % As is explained in makeDiffImg.m, the Y are accidentally 39 frames
    % long, which has been fixed. This didn't impact the training for some
    % reason. Probably something to do with LSTM training in MATLAB.
    
    [~, trueClass] = max(Y(:,:,1:end-1), [], 1);

    % Remove singleton class dimension
    predictedClass = squeeze(predictedClass);
    trueClass = squeeze(trueClass);

    % Count correct predictions
    correct = correct + sum(predictedClass == trueClass, 'all');
    total = total + numel(trueClass);

    % Store for confusion matrix
    allPredicted = [allPredicted; predictedClass(:)];
    allTrue = [allTrue; trueClass(:)];
end

% Overall accuracy
accuracy = correct / total;

fprintf('Accuracy: %.2f%%\n', accuracy * 100);

% Confusion matrix
trueLabels = categorical(allTrue, 1:4, labels);
predictedLabels = categorical(allPredicted, 1:4, labels);

figure;
confusionchart(trueLabels, predictedLabels);
title(sprintf('726k Parameter Model Frame Classification Accuracy: %.2f%%', accuracy * 100));

