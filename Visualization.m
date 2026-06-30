% clear
% clc
% close all

load intubation_video_datastores.mat
labels = ["Reg_Intubate", "Left_Intubate", "Right_Intubate", "Esoph_Intubate"];

% data = cell(50,1);
% label_list = zeros(50,1);
% 
% for i = 1:20
%     data{i} = read(train_ds);
%     label_list(i) = data{i}{2};
% end
% 
% [~, ia, ~] = unique(label_list,'stable');
% 
% select_data = data(ia);

testdata = select_data{4}{1};
testdata = testdata(:,1,:);
% [32,1,39] -> [32, 39]
testdata = reshape(testdata, [32,39]);

surf(testdata)
xlabel("Frame") 
ylabel("Electrode")
zlabel("Voltage")