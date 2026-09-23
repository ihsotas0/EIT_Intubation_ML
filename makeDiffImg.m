function data_out = makeDiffImg(data_in)

video = data_in{1};

data_out{1} = video(:,:,2:end,:)-video(:,:,1,:);

% Add time dimension to one-hot vectors (transpose them so dims are 4x38,
% CT; mbq collate adds B to N+1 dim, so CTB).
data_out{2} = repmat(data_in{2}.',1,size(data_in{1},3));

end