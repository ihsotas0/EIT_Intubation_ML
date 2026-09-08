function [train_ads, val_ads, test_ads] = makeArrayDatastores(train_ds, val_ds, test_ds)

raw_readall_train = readall(train_ds);
raw_readall_val = readall(val_ds);
raw_readall_test = readall(test_ds);

% Make cell array of cell arrays into 2D cell array so output of read() is
% 1x2 cell array and not cell array of cell arrays.

raw_readall_train = vertcat(raw_readall_train{:});
raw_readall_val = vertcat(raw_readall_val{:});
raw_readall_test = vertcat(raw_readall_test{:});

train_ads = arrayDatastore(raw_readall_train, ...
    IterationDimension=1, ...
    OutputType="same");

val_ads = arrayDatastore(raw_readall_val, ...
    IterationDimension=1, ...
    OutputType="same");

test_ads = arrayDatastore(raw_readall_test, ...
    IterationDimension=1, ...
    OutputType="same");

end