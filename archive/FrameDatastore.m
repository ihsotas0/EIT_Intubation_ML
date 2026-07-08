clear
clc
close all

%% ---------------------------------------------------------------------- %
%                             Define Datastore                            %
% ----------------------------------------------------------------------- %

fprintf("Setting up the datastore with all files\n")
setupStart = tic;

% Define location of data
atlas_dir = '/run/user/1000/gvfs/smb-share:server=ripoff.math.colostate.edu,share=eit/Anatomical_Atlas_3D/Babies_GE';

% Create a base datastore to find the list of files
volt_ds = fileDatastore(atlas_dir, ...
                        "ReadFcn", @LoadFrame, ...
                        "ReadMode","partialfile", ...
                        "IncludeSubfolders", true, ...
                        "FileExtensions", ".mat");

% Filter to keep only folders of voltage data on belts
keep          = contains(volt_ds.Files, fullfile("belt","voltages"), "IgnoreCase", true);
volt_ds.Files = volt_ds.Files(keep);

% Filter to keep only files whose names contain "Video"
keep          = contains(volt_ds.Files, 'Video', 'IgnoreCase', true);
volt_ds.Files = volt_ds.Files(keep);

setupTime= toc(setupStart);
fprintf("   It took %.2f seconds to setup and filter the datastore\n", setupTime)

%% ---------------------------------------------------------------------- %
%                             Split By Subject                            %
% ----------------------------------------------------------------------- %

fprintf("Splitting into three datastores by subject name\n")
splitStart = tic;

% Set percentages for training and validation. Test will be leftovers.
sizeTrain = 0.70;
sizeVal   = 0.20;

% Set the "randomness" off so if I make changes, each network has the same training sets for comparison
rng(1)

% Find subject names & randomize them
expr       = "(case\d{6}|R\d{4})";
sbjs       = regexp(volt_ds.Files, expr, "match", "once");
uniqueSbjs = unique(sbjs);
nSbjs      = numel(uniqueSbjs);
randomSbjs = uniqueSbjs(randperm(nSbjs));

% Determine which subjects belong to which dataset
nTrain = round(sizeTrain * nSbjs);
nVal   = round(sizeVal * nSbjs);
nTest  = nSbjs - nTrain - nVal;

trainSbjs = randomSbjs(1:nTrain);
valSbjs   = randomSbjs(nTrain+1 : nTrain+nVal);
testSbjs  = randomSbjs(nTrain+nVal+1 : end);

isTrain = ismember(sbjs, trainSbjs);
isVal   = ismember(sbjs, valSbjs);
isTest  = ismember(sbjs, testSbjs);

% Create the three partitioned datastores
Train_ds = copy(volt_ds);
Train_ds.Files = volt_ds.Files(isTrain);

Val_ds = copy(volt_ds);
Val_ds.Files = volt_ds.Files(isVal);

Test_ds = copy(volt_ds);
Test_ds.Files = volt_ds.Files(isTest);

splitTime = toc(splitStart);
fprintf("   It took %.2f seconds to split the datastore into three\n", splitTime)

%% ---------------------------------------------------------------------- %
%                           Save the Datastores                           %
% ----------------------------------------------------------------------- %

% Save the total datastore and the final datastores
fprintf("Saving datastores\n")
save("Intubation_Datastores.mat", "volt_ds", "Train_ds", "Val_ds", "Test_ds")


%% ---------------------------------------------------------------------- %
%                           Load & Save the Data                          %
% ----------------------------------------------------------------------- %

% Save the total datastore and the final datastores
fprintf("Reading all data\n")
allTrain = vertcat(readall(Train_ds,"UseParallel", true));
allVal   = vertcat(readall(Val_ds,  "UseParallel", true));
allTest  = vertcat(readall(Test_ds, "UseParallel", true));

%%si
fprintf("Saving data\n")
save("Intubation_Data.mat", "allTrain", "allVal", "allTest", "-v7.3")

%% ---------------------------------------------------------------------- %
%                             Custom Functions                            %
% ----------------------------------------------------------------------- %

function [data, userdata, done] = LoadFrame(full_filepath, userdata)
    % Initialize userdata on first call
    if isempty(userdata)
        % Load voltage data based on if it has noise or not
        try 
            % Load the voltage with no noise and add noise
            userdata.volt = load(full_filepath, "Umeas_NoNoise").Umeas_NoNoise;
            use_GE        = load(full_filepath, "flags").flags.use_GE;

            % Add noise to the voltages
            if use_GE == 1
                userdata.volt = awgn(userdata.volt, 55, "measured");
            else
                userdata.volt = awgn(userdata.volt, 100, "measured");
            end
        catch
            % Load the voltage that already has noise
            userdata.volt = load(full_filepath, "Umeas").Umeas;
        end
        
        % Set up frame information
        userdata.numFrames = size(userdata.volt, 3);
        userdata.frame     = 1;   % start at frame 1

        % Normalize the data from [-1, 1]
        oldMin = min(userdata.volt,[],"all");
        oldMax = max(userdata.volt,[],"all");
        newMin = -1;
        newMax =  1;
        userdata.volt = (userdata.volt - oldMin) / (oldMax - oldMin) * (newMax - newMin) + newMin;

        % % Standardize the data with zero mean and one std
        % userdata.volt = normalize(userdata.volt, 3);
    end
    
    % Total volts is 32x31xframes, so we output only the one frame
    data{1} = squeeze(userdata.volt(:,:,userdata.frame));

    % --- Parse label from filename ---
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
    % data{2} = double(find(label == labels));
    data{2} = categorical(label, ["Reg_Intubate", "Left_Intubate", "Right_Intubate", "Esoph_Intubate"]);

    % Increase the counter and determine if we're done reading the file
    userdata.frame = userdata.frame + 1;
    if userdata.frame > userdata.numFrames
        done = true;
    else
        done = false;
    end

end