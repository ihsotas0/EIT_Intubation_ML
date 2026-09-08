function [train_ds, val_ds, test_ds] = makeVoltageArrayDatastores(train_volt_ds, val_volt_ds, test_volt_ds)
% Unpack and format data
[X_train, Y_train] = formattedVoltageReadall(train_volt_ds);
[X_val, Y_val]     = formattedVoltageReadall(val_volt_ds);
[X_test, Y_test]   = formattedVoltageReadall(test_volt_ds);

% One-hot encodding
labels = ["Reg_Intubate", "Left_Intubate", "Right_Intubate", "Esoph_Intubate"];
numClasses = length(labels);

Y_train = cellfun(@full, ind2vec(Y_train, numClasses), UniformOutput=false);
Y_val = cellfun(@full, ind2vec(Y_val, numClasses), UniformOutput=false);
Y_test = cellfun(@full, ind2vec(Y_test, numClasses), UniformOutput=false);

% Create the arrayDatastores
train_ds = arrayDatastore(cat(2, X_train, Y_train), ...
    IterationDimension=1, ...
    OutputType="same");
val_ds   = arrayDatastore(cat(2, X_val, Y_val), ...
    IterationDimension=1, ...
    OutputType="same");
test_ds  = arrayDatastore(cat(2, X_test, Y_test), ...
    IterationDimension=1, ...
    OutputType="same");
end