function [X_cell, Y_cell] = formatReadall(raw_data)
    % Use .mat file saved from readall(VideoDatastore) to make
    % ArrayDatastore. Load data into memory for faster training without
    % need for SMB network drive.
    %
    % Syntax to use output in datastore format:
    % train_ds = arrayDatastore(cat(2, X_train, Y_train), ...
    %                           "IterationDimension", 1, ...
    %                           "OutputType","same");

    X_cell = cell(length(raw_data), 1);
    Y_cell = cell(length(raw_data), 1);
    for i = 1:length(raw_data)
        X_cell{i} = raw_data{i}{1};
        Y_cell{i} = raw_data{i}{2};
    end
end
