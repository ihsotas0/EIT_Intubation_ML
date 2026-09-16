function [x_out, y_out] = makeContigSubseqs(x_in, y_in, range)

% x_in: N-by-1 cell array of 48x48x2x38 variables
% y_in: N-by-1 cell array of 1x4 variables

% Make all contiguous subsequences of length in range (eg. 4-38) from each
% x_in tensor (using the 38 long dim as the dim for splicing). Put all of
% the sequences in x_in_expanded, which is a M*N-by-1 cell array of
% tensors, where M is the number of contiguous subsequences made from each
% original tensor. x_in_expanded{1:M} should be all the contiguous
% subsequences made from x_in{1} and so on. Copy the tensors in y_in to
% make a M*N-by-1 cell array called y_in_expanded, where y_in_expanded{1:M}
% = y{1}, and so on.

N = numel(x_in);

% Number of subsequences per input tensor
M = sum((range) <= range(end) .* (range(end) - (range) + 1));

% Preallocate
x_in_expanded = cell(M * N, 1);
y_in_expanded = cell(M * N, 1);

idx = 1;

for n = 1:N
    x = x_in{n};   % 48 x 48 x 2 x 38
    y = y_in{n};   % 1 x 4

    for seqLen = range
        % Starting indices for all subsequences of this length
        for startIdx = 1:(range(end) - seqLen + 1)
            endIdx = startIdx + seqLen - 1;

            % Slice along the 4th (38-long) dimension
            x_in_expanded{idx} = x(:, :, :, startIdx:endIdx);

            % Copy corresponding y
            y_in_expanded{idx} = y;

            idx = idx + 1;
        end
    end
end

x_out = cat(5, x_in_expanded{:}); % variables are shaped 48x48x2x38
y_out = cat(1, y_in_expanded{:}); % one-hot variables are shaped 1x4

end