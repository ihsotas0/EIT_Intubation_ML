function data = loadVideo(full_filepath)
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