function visualizeVoltageTensor(dataCell)    
tensor = dataCell{1};
label_raw = dataCell{2};

labels = ["Reg Intubate", "Left Intubate", "Right Intubate", "Esoph Intubate"];    

% For one-hot encoded labels
[~,index] = max(label_raw);

label_text = labels(index);

[H, W, num_frames] = size(tensor);
state.current_frame = 1;

% Create figure
hFig = figure('Name', 'Tensor Visualizer', ...
              'WindowKeyPressFcn', @keyPressCallback, ...
              'KeyPressFcn', @keyPressCallback, ... % Backup listener
              'NumberTitle', 'off', ...
              'Position', [100, 100, 600, 650], ...
              'MenuBar', 'none');

% Create axes and image
% Add KeyPressFcn to the axes itself, just in case the axes steals focus
hAx = axes('Parent', hFig, 'Position', [0.1 0.15 0.8 0.75]); 
           
hIm = imagesc(hAx, tensor(:,:,1));
hIm.ButtonDownFcn = @(src,evt) uicontrol(hFig); % ensures click gives focus; keeps figure WindowKeyPressFcn active

colormap(hAx, gray); 
axis(hAx, 'image');

hTitle = title(hAx, sprintf('Label: %s  |  Frame: %d / %d', label_text, state.current_frame, num_frames), ...
               'FontSize', 12, 'FontWeight', 'bold');
xlabel(hAx, sprintf('Pattern (%d)', W));
ylabel(hAx, sprintf('Electrode (%d)', H));

% Slider
hSlider = uicontrol('Parent', hFig, 'Style', 'slider', ...
    'Min', 1, 'Max', num_frames, 'Value', 1, ...
    'Position', [50 40 500 20], ...
    'Callback', @sliderCallback);

uicontrol('Parent', hFig, 'Style', 'text', ...
    'String', 'Click the image first! Then use Left/Right arrows (or A/D) to step. Q to close.', ...
    'Position', [50 15 500 20], 'BackgroundColor', hFig.Color, 'FontSize', 10);

% --- Nested Callback Functions ---

function updateDisplay()
    % Update image data
    hIm.CData = tensor(:,:,state.current_frame);
    
    % Update title text
    hTitle.String = sprintf('Label: %s  |  Frame: %d / %d', label_text, state.current_frame, num_frames);
    
    % Sync slider (only if it actually needs to move to prevent infinite loops)
    if hSlider.Value ~= state.current_frame
        hSlider.Value = state.current_frame;
    end
    
    drawnow; 
end

function keyPressCallback(~, event)
    % Safety check for event object
    if nargin < 2 || isempty(event)
        return;
    end
    
    % Extract key safely (handles both object and struct event types)
    try
        key = event.Key;
    catch
        key = '';
    end
    
    % Right / Forward (strcmpi makes it case-insensitive)
    if strcmpi(key, 'rightarrow') || strcmpi(key, 'd')
        state.current_frame = min(state.current_frame + 1, num_frames);
    % Left / Backward
    elseif strcmpi(key, 'leftarrow') || strcmpi(key, 'a')
        state.current_frame = max(state.current_frame - 1, 1);
    % Close Figure
    elseif strcmpi(key, 'q') || strcmpi(key, 'escape')
        close(hFig);
        return;
    end
    
    updateDisplay();
end

function sliderCallback(src, ~)
    state.current_frame = round(src.Value);
    updateDisplay();
end

% Initial draw to ensure everything renders on startup
updateDisplay();
end