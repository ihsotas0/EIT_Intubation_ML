function [X_cell, Y_cell] = formatReadall(raw_data)
    X_cell = cell(length(raw_data), 1);
    Y_cell = cell(length(raw_data), 1);
    for i = 1:length(raw_data)
        X_cell{i} = raw_data{i}{1};
        Y_cell{i} = raw_data{i}{2};
    end
end
