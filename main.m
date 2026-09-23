clear all; clc; clear;

%% Make datastores

%path = "/run/user/1000/gvfs/smb-share:server=ripoff.math.colostate.edu,share=eit/Anatomical_Atlas_3D/Babies_GE";
path = input("Path: ");

[train_ds, val_ds, test_ds] = makeDatastores(path, 0.75, 0.15);
[train_ds, val_ds, test_ds] = makeArrayDatastores(train_ds, val_ds, test_ds);

%% Convert to difference images

train_ds_diff = transform(train_ds,@makeDiffImg);
val_ds_diff = transform(val_ds,@makeDiffImg);
test_ds_diff = transform(test_ds,@makeDiffImg);

%% Make miniBatchQueues (with contiq subseq and seq-to-seq output)

miniBatchSize = 10;

[mbqTrain, mbqVal, mbqTest] = makeMBQs( ...
    train_ds, val_ds, test_ds, ...
    miniBatchSize, "SSTSBC", "BCT");

%% Train models

models = {'diff_3d_cnn_lstm_seq_to_seq_1M',...
          'diff_3d_cnn_lstm_seq_to_seq_750k',...
          'diff_3d_cnn_lstm_seq_to_seq_500k',...
          'diff_3d_cnn_lstm_seq_to_seq_250k',...
          'diff_3d_cnn_lstm_seq_to_seq_100k',...
          'diff_3d_cnn_lstm_seq_to_seq_50k',...
          'diff_3d_cnn_lstm_seq_to_seq_25k',};

for i=models
    model_name = i{1};
    
    filename = sprintf('untrained_models/%s.mat', model_name);
    load(filename,"net")
    
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

% TODO: Confusion chart and other better tests
testAccuracy = testnet(train_net, mbqTest, "accuracy");
fprintf('Test Accuracy: %.2f%%\n', testAccuracy);

% 92.80% for 3d_cnn_lstm.mat

