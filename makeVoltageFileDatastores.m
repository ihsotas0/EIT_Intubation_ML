function [train_ds, val_ds, test_ds] = makeVoltageFileDatastores(filepath,train_size,val_size,seed)
ds = fileDatastore(filepath, ...
    "ReadFcn", @loadVoltageVideoFile, ...
    "IncludeSubfolders", true, ...
    "FileExtensions", ".mat");

% Filter for files with 'Video' substring in /belt/voltages
folderFilter = contains(ds.Files, fullfile("belt","voltages"), "IgnoreCase", true);
ds.Files = ds.Files(folderFilter);

substringFilter = contains(ds.Files, 'Video', 'IgnoreCase', true);
ds.Files = ds.Files(substringFilter);

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
end