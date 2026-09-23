function [x_out, y_out] = makeOneHotSeqs(x_in, y_in)

% x_in: N-by-1 cell array of 48x48x2x38 variables
% y_in: N-by-1 cell array of 1x4 variables

% 38 is hard coded because size(x_in{1},4) didn't work, for some reason
y_in_seq = cellfun(@(x) repmat(x,1,1,38), y_in, 'UniformOutput',false);

x_out = cat(5, x_in{:}); % variables are shaped 48x48x2x38
y_out = cat(1, y_in_seq{:}); % one-hot variables are shaped 1x4x38 (time)

end