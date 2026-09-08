function data = datastoreRead(path)

% Get absolute conductivity reconstructions
abs_top = load(path).abs_top;
abs_bot = load(path).abs_bot;

% NaN -> 0
abs_top(isnan(abs_top)) = 0;
abs_bot(isnan(abs_bot)) = 0;

% Dims: [space-x, space-y, time, space-z]
data{1} = cat(4, abs_top, abs_bot);

% Get label from file name -> one-hot encoding
labels = ["Reg_Intubate", "Left_Intubate", "Right_Intubate", "Esoph_Intubate"];
onehot = zeros(1,4);

for i=1:4
    onehot(i) = double(contains(path, labels(i)));
end

data{2} = onehot;

end