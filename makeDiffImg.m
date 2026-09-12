function data_out = makeDiffImg(data_in)

video = data_in{1};

data_out{1} = video(:,:,2:end,:)-video(:,:,1,:);
data_out{2} = data_in{2};

end