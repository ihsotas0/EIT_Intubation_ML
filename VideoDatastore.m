clear
clc
close all

%% ---------------------------------------------------------------------- %
%                             Define Datastore                            %
% ----------------------------------------------------------------------- %

fprintf("Setting up video datastore with all files\n")
setup_start = tic;

% Define location of data
% Jonah machine: '/run/user/1000/gvfs/smb-share:server=ripoff.math.colostate.edu,share=eit/Anatomical_Atlas_3D/Babies_GE';
atlas_dir = '/run/user/1000/gvfs/smb-share:server=ripoff.math.colostate.edu,share=eit/Anatomical_Atlas_3D/Babies_GE'

% Create a base datastore to find the list of files
volt_ds = fileDatastore(atlas_dir, ...
                        "ReadFcn", @LoadVideo, ...
                        "IncludeSubfolders", true, ...
                        "FileExtensions", ".mat");

% Filter for files with 'Video' substring in /belt/voltages
folder_filter = contains(volt_ds.Files, fullfile("belt","voltages"), "IgnoreCase", true);
volt_ds.Files = volt_ds.Files(folder_filter);

substring_filter = contains(volt_ds.Files, 'Video', 'IgnoreCase', true);
volt_ds.Files = volt_ds.Files(substring_filter);

setup_time = toc(setup_start);
fprintf("It took %.2f seconds to setup and filter the datastore\n", setup_time)

%% ---------------------------------------------------------------------- %
%                             Split By Subject                            %
% ----------------------------------------------------------------------- %

fprintf("Splitting into three datastores by subject name\n")
split_start = tic;

% Set percentages for training and validation. Test will be leftovers.
size_train = 0.70;
size_val   = 0.20;

% Ensure reproducible training sets
rng(1)

% Find case names and randomize them
expr       = "(case\d{6}|R\d{4})";
cases       = regexp(volt_ds.Files, expr, "match", "once");
unique_cases = unique(cases);
n_cases      = numel(unique_cases);
random_cases = unique_cases(randperm(n_cases));

% Determine which cases belong to which dataset
n_train = round(size_train * n_cases);
n_val   = round(size_val * n_cases);
n_test  = n_cases - n_train - n_val;

train_cases = random_cases(1:n_train);
val_cases   = random_cases(n_train+1 : n_train+n_val);
test_cases  = random_cases(n_train+n_val+1 : end);

is_train = ismember(cases, train_cases);
is_val   = ismember(cases, val_cases);
is_test  = ismember(cases, test_cases);

% Create the three partitioned datastores
train_ds = copy(volt_ds);
train_ds.Files = volt_ds.Files(is_train);

val_ds = copy(volt_ds);
val_ds.Files = volt_ds.Files(is_val);

test_ds = copy(volt_ds);
test_ds.Files = volt_ds.Files(is_test);

split_time = toc(split_start);
fprintf("It took %.2f seconds to split the datastore into three\n", split_time)

%% ---------------------------------------------------------------------- %
%                           Save the Datastores                           %
% ----------------------------------------------------------------------- %

% Save only the partitioned datastores
fprintf("Saving datastores\n")
save("intubation_video_datastores.mat", "train_ds", "val_ds", "test_ds")

%% ---------------------------------------------------------------------- %
%                           Load & Save the Data                          %
% ----------------------------------------------------------------------- %

fprintf("Reading all data\n")
all_train = vertcat(readall(train_ds,"UseParallel", true));
all_val   = vertcat(readall(val_ds,  "UseParallel", true));
all_test  = vertcat(readall(test_ds, "UseParallel", true));

fprintf("Saving data\n")
save("intubation_video_data.mat", "all_train", "all_val", "all_test")

%% ---------------------------------------------------------------------- %
%                             Custom Functions                            %
% ----------------------------------------------------------------------- %

function data = LoadVideo(full_filepath)
    % Check voltage data for noise and add it
    try 
        % Load the voltage with no noise and add noise
        voltage = load(full_filepath, "Umeas_NoNoise").Umeas_NoNoise;
        use_GE  = load(full_filepath, "flags").flags.use_GE;

        % Does this add noise distributions i.i.d. along each axis?
        % Does noise need to be added along pattern axis (axis 2)?
        if use_GE == 1
            voltage = awgn(voltage, 55, "measured");
        else
            voltage = awgn(voltage, 100, "measured");
        end
    catch
        % Load the voltage that already has noise
        voltage = load(full_filepath, "Umeas").Umeas;
    end

    % Normalize the Data from [-1, 1]
    old_min = min(voltage,[],"all");
    old_max = max(voltage,[],"all");
    new_min = -1;
    new_max = 1;
    voltage = (voltage - old_min) / (old_max - old_min) * (new_max - new_min) + new_min;

    data{1} = voltage;

    % Parse label from filename
    file_parts  = split(full_filepath, filesep);
    filename    = file_parts{end};
    name_parts  = split(filename, "-");
    label       = name_parts{3};
    label_parts = split(label, "_");

    if numel(label_parts) > 2
        label = strcat(label_parts{1}, "_", label_parts{2});
    end

    % Convert the string label to a numeric
    labels = ["Reg_Intubate", "Left_Intubate", "Right_Intubate", "Esoph_Intubate"];
    data{2} = find(label == labels);
end