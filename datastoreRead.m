function data = datastoreRead(path)

% Get absolute conductivity reconstructions
abs_top = load(path).abs_top;
abs_bot = load(path).abs_bot;

% NaN -> 0
abs_top(isnan(abs_top)) = 0;
abs_bot(isnan(abs_bot)) = 0;

% Dimensions: SSTS (height-width-time-depth)
video = cat(4, abs_top, abs_bot);

% td-EIT: Subtract the first frame from the rest, return the remaining frames
data{1} = video(:,:,2:end,:)-video(:,:,1,:);

% Get label from file name -> one-hot encoding -> convert to tensor with time
labels = ["Reg_Intubate", "Left_Intubate", "Right_Intubate", "Esoph_Intubate"];
onehot = zeros(4,1);

% Make one-hot
for i=1:4
    onehot(i) = double(contains(path, labels(i)));
end

% Dimensions: CT (label-time)
data{2} = repmat(onehot,1,size(video,3)-1);

end