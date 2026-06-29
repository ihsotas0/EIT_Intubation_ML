clear
clc
close all

load Intubation_Data.mat
load network.mat

labels = ["Reg_Intubate", "Left_Intubate", "Right_Intubate", "Esoph_Intubate"];
numClasses = length(labels);

%% ---------------------------------------------------------------------- %
%                              Validation Set                             %
% ----------------------------------------------------------------------- %

Val_ds_mb = transform(Val_ds, @minibatchFcn);

YPred_val = classify(net, Val_ds_mb);
YTrue_val = readall(Val_ds_mb);
YTrue_val = YTrue_val(:,2);   % second cell contains labels
YTrue_val = vertcat(YTrue_val{:});

figure;
confusionchart(YTrue_val, YPred_val, ...
    RowSummary="row-normalized", ...
    ColumnSummary="column-normalized", ...
    XLabel="Predicted", YLabel="True", ...
    Title="Confusion Matrix – Validation Set", ...
    ClassLabels=labels);

valAccuracy = mean(YPred_val == YTrue_val);
disp("Validation Accuracy: " + valAccuracy);

%% ---------------------------------------------------------------------- %
%                                 Test Set                                %
% ----------------------------------------------------------------------- %

Test_ds_mb = transform(Test_ds, @minibatchFcn);

YPred_test = classify(net, Test_ds);
YTrue_test = readall(Test_ds);
YTrue_test = YTrue_test(:,2);
YTrue_test = vertcat(YTrue_test{:});

figure;
confusionchart(YTrue_test, YPred_test, ...
    RowSummary="row-normalized", ...
    ColumnSummary="column-normalized", ...
    XLabel="Predicted", YLabel="True", ...
    Title="Confusion Matrix – Test Set", ...
    ClassLabels=labels);

testAccuracy = mean(YPred_test == YTrue_test);
disp("Test Accuracy: " + testAccuracy);

%% ---------------------------------------------------------------------- %
%                            Per-Class Accuracy                           %
% ----------------------------------------------------------------------- %

classes = categories(YTrue_test);
numClasses = numel(classes);

disp("Per-Class Accuracy:");
for i = 1:numel(labels)
    idx = (YTrue_test == labels(i));
    classAcc = mean(YPred_test(idx) == YTrue_test(idx));
    fprintf("  %s: %.4f\n", labels(i), classAcc);
end

%% ---------------------------------------------------------------------- %
%                             Custom Functions                            %
% ----------------------------------------------------------------------- %

function dataOut = minibatchFcn(dataIn)
    % dataIn is an N×2 cell array: {X, Y}
    X = cat(4, dataIn{:,1});   % concatenate along 4th dim (batch)
    Y = categorical(dataIn{:,2});

    % Convert to dlarray for GPU efficiency
    X = dlarray(single(X), "SSCB");   % S=spatial dims, C=channel, B=batch

    % Move to GPU if available
    if canUseGPU
        X = gpuArray(X);
    end

    dataOut = {X, Y};
end
