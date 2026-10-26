
function visualizeModelOutput(data, sampleIdx)
% TODO: Fix video frame viewer to show top AND bottom slices
% TODO: Document code to set up a "data" object

    if nargin < 2
        sampleIdx = 1;
    end

    labels = ["Reg Intubate", "Left Intubate", ...
              "Right Intubate", "Esoph Intubate"];

    assert(sampleIdx >= 1 && sampleIdx <= size(data,1), ...
        'sampleIdx must be between 1 and %d.', size(data,1));

    % Extract and convert dlarray data
    video = gather(extractdata(data{sampleIdx,1}));
    truth = double(gather(extractdata(data{sampleIdx,2})));
    pred  = double(gather(extractdata(data{sampleIdx,3})));

    % Remove singleton batch dimension: H x W x C x T
    video = squeeze(video);

    if ndims(video) ~= 4
        error('Expected video dimensions H x W x C x T after squeeze.');
    end

    numFrames  = size(video,4);
    numClasses = size(truth,1);

    if numClasses ~= numel(labels)
        error('Number of labels (%d) must match number of classes (%d).', ...
              numel(labels), numClasses);
    end

    % Create figure and layout
    fig = figure('Name', sprintf('Sample %d', sampleIdx), ...
        'Color', 'w', 'NumberTitle', 'off');

    tl = tiledlayout(fig, 2, 2, ...
        'TileSpacing', 'compact', 'Padding', 'compact');

    % 1. Video
    ax1 = nexttile(tl,[2 1]);
    hImg = imagesc(ax1, video(:,:,1,1));
    axis(ax1, 'image');
    axis(ax1, 'off');
    colormap(ax1, "default");
    title(ax1, sprintf('td-EIT Video: Sample %d, Frame 1/%d', ...
        sampleIdx, numFrames));

    % 2. Ground truth
    ax2 = nexttile(tl);
    imagesc(ax2, 1:numFrames, 1:numClasses, truth);
    axis(ax2, 'tight');
    set(ax2, 'YTick', 1:numClasses, 'YTickLabel', labels);
    ylabel(ax2, 'Class');
    xlabel(ax2, 'Frame');
    title(ax2, 'Ground-truth classification');
    colorbar(ax2);

    % 3. Predictions
    ax3 = nexttile(tl);
    imagesc(ax3, 1:numFrames, 1:numClasses, pred);
    axis(ax3, 'tight');
    set(ax3, 'YTick', 1:numClasses, 'YTickLabel', labels);
    ylabel(ax3, 'Class');
    xlabel(ax3, 'Frame');
    title(ax3, 'Model predictions');
    colorbar(ax3);

    % Frame cursors
    hold(ax2, 'on');
    gtCursor = xline(ax2, 1, 'w', 'LineWidth', 1.5);
    hold(ax2, 'off');

    hold(ax3, 'on');
    predCursor = xline(ax3, 1, 'w', 'LineWidth', 1.5);
    hold(ax3, 'off');

    % Slider for video frame navigation
    slider = uicontrol(fig, 'Style', 'slider', ...
        'Units', 'normalized', ...
        'Position', [0.25 0.015 0.5 0.035], ...
        'Min', 1, 'Max', numFrames, 'Value', 1, ...
        'SliderStep', [1/max(numFrames-1,1), ...
                       min(10/max(numFrames-1,1),1)]);

    slider.Callback = @(src,~) updateFrame(round(src.Value));
    fig.KeyPressFcn = @keyPress;

    function updateFrame(t)
        t = max(1, min(numFrames, t));
        slider.Value = t;
        hImg.CData = video(:,:,1,t);

        title(ax1, sprintf('td-EIT Video: Sample %d, Frame %d/%d', ...
            sampleIdx, t, numFrames));

        gtCursor.Value = t;
        predCursor.Value = t;
        drawnow limitrate;
    end

    function keyPress(~, event)
        switch event.Key
            case 'rightarrow'
                updateFrame(round(slider.Value) + 1);
            case 'leftarrow'
                updateFrame(round(slider.Value) - 1);
        end
    end
end
