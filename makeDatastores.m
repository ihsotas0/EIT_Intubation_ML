function [train_ads, val_ads, test_ads] = makeDatastores(filepath,train_size,val_size,seed)

%% Make main fileDatastore

% datastoreRead converts to td-EIT, makes one-hot tensor for labels,
% adds time dimension to output tensor before any transform()'s
ds = fileDatastore(filepath, ...
    "ReadFcn", @datastoreRead, ...
    "IncludeSubfolders", true, ...
    "FileExtensions", ".mat");

% Filter for files with 'Video' substring in /belt/Reconstruction_Images_48x48
folderFilter = contains(ds.Files, fullfile("belt","Reconstruction_Images_48x48"), "IgnoreCase", true);
ds.Files = ds.Files(folderFilter);

substringFilter = contains(ds.Files, 'Video', 'IgnoreCase', true);
ds.Files = ds.Files(substringFilter);

%% Split by subject

% Ensure reproducible training sets
rng(seed)

% Find case names and randomize them
expr         = "(case\d{6}|R\d{4})";
cases        = regexp(ds.Files, expr, "match", "once");
unique_cases = unique(cases);
n_cases      = numel(unique_cases);
random_cases = unique_cases(randperm(n_cases));

% Determine which cases belong to which dataset
n_train = round(train_size * n_cases);
n_val   = round(val_size * n_cases);

train_cases = random_cases(1:n_train);
val_cases   = random_cases(n_train+1 : n_train+n_val);
test_cases  = random_cases(n_train+n_val+1 : end);

is_train = ismember(cases, train_cases);
is_val   = ismember(cases, val_cases);
is_test  = ismember(cases, test_cases);

% Create the three partitioned datastores
train_ds = copy(ds);
train_ds.Files = ds.Files(is_train);

val_ds = copy(ds);
val_ds.Files = ds.Files(is_val);

test_ds = copy(ds);
test_ds.Files = ds.Files(is_test);

%% Convert to in-memory arrayDatastores

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