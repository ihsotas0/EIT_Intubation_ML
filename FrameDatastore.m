classdef FrameDatastore < matlab.io.Datastore & matlab.io.datastore.MiniBatchable
    properties
        Files
        FrameIndex
        CurrentIndex = 1
    
        MiniBatchSize = 1
    end

    properties (SetAccess = protected)
        NumObservations
    end



    methods
        function ds = FrameDatastore(files, frameIdx)
            ds.Files = files;
            ds.FrameIndex = frameIdx;
            ds.NumObservations = numel(files);
        end

        function tf = hasdata(ds)
            tf = ds.CurrentIndex <= numel(ds.Files);
        end

        function reset(ds)
            ds.CurrentIndex = 1;
        end

        function [data, info] = read(ds)
            file = ds.Files{ds.CurrentIndex};
            frame = ds.FrameIndex(ds.CurrentIndex);

            % Load the data as {voltage, label}
            data = LoadData(file, frame);

            info.Size = {size(data{1}), size(data{2})};In m
            info.FileName = file;

            ds.CurrentIndex = ds.CurrentIndex + 1;
        end

        % ============================================================
        % Required by MiniBatchable
        % ============================================================
        function subds = partition(ds,n,ii)
            subds = copy(ds);
            subds.Files = partition(ds.Files,n,ii);
            reset(subds);         
        end

        function n = numpartitions(ds)
            % Each element is its own partition
            n = numel(ds.Files);
        end

        % function dsCopy = copy(ds)
        %     dsCopy = FrameDatastore(ds.Files, ds.FrameIndex);
        %     dsCopy.CurrentIndex = ds.CurrentIndex;
        % end

        function tf = isDone(ds)
            tf = ~ds.hasdata;
        end
    end

    methods (Hidden = true)
        function frac = progress(ds)
            % Determine percentage of data read from datastore
            if hasdata(ds) 
               frac = (ds.CurrentIndex-1)/...
                             numel(ds.Files); 
            else 
              frac = 1;  
            end 
        end
    end
end


%% ---------------------------------------------------------------------- %
%                             Custom Functions                            %
% ----------------------------------------------------------------------- %

function out = LoadData(full_filepath, frameIndex)
    % --- Load voltage data ---
    try 
        % Load the voltage with no noise and add noise
        voltage = load(full_filepath, "Umeas_NoNoise").Umeas_NoNoise;
        use_GE  = load(full_filepath, "flags").flags.use_GE;

        % voltage is 32x31xframes, so we take only one slice
        voltage = squeeze(voltage(:,:,frameIndex));

        if use_GE == 1
            voltage = awgn(voltage, 55, "measured");
        else
            voltage = awgn(voltage, 100, "measured");
        end
    catch
        % Load the voltage that already has noise
        voltage = load(full_filepath, "Umeas").Umeas;

        % voltage is 32x31xframes, so we take only one slice
        voltage = squeeze(voltage(:,:,frameIndex));
    end

    % --- Normalize the Data from [-1, 1] ---
    oldMin = min(voltage,[],"all");
    oldMax = max(voltage,[],"all");
    newMin = -1;
    newMax = 1;
    voltage = (voltage - oldMin) / (oldMax - oldMin) * (newMax - newMin) + newMin;

    out{1} = voltage;

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
    out{2} = find(label == labels);

end