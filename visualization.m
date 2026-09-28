% Code to make a 'hist' table is not in a script or documented yet, but
% will be soon. Involves outerjoins, column deletions, and variable
% renaming of TrainingHistory and ValidationHistory from train_info data.

runs = {'726','398','183','100','51','30'};
colors = lines(numel(runs));

figure();
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

smoothWindow = 500;

% Training Loss
nexttile;
hold on;
for k = 1:numel(runs)
    y = hist.("Loss" + runs{k});
    ySmooth = smoothdata(y, 'movmean', smoothWindow);
    plot(hist.Iteration, ySmooth, ...
        'Color', colors(k,:), ...
        'DisplayName', runs{k} + "k params");
end
hold off;
set(gca,'YScale','log');
set(gca,'XScale','log');
xlabel('Epoch');
ylabel('Loss');
title('Training Loss');
legend('Location','best');
grid on;

% Validation Loss
nexttile;
hold on;
for k = 1:numel(runs)
    y = hist.("ValLoss" + runs{k});
    x = hist.Iteration;
    valid = isfinite(y);
    % smoothWindow/50 because every 50 train has 1 val.
    ySmooth = smoothdata(y(valid), 'movmean', smoothWindow/50);

    plot(x(valid), ySmooth, ...
        '.-', ...
        'Color', colors(k,:), ...
        'MarkerSize', 3, ...
        'DisplayName', runs{k} + "k params");
end
hold off;
set(gca,'YScale','log');
set(gca,'XScale','log');
xlabel('Epoch');
ylabel('Loss');
title('Validation Loss');
legend('Location','best');
grid on;

% Training Accuracy
nexttile;
hold on;
for k = 1:numel(runs)
    y = hist.("Accuracy" + runs{k});
    
    ySmooth = smoothdata(y, 'movmean', smoothWindow);
    plot(hist.Iteration, ySmooth, ...
        'Color', colors(k,:), ...
        'DisplayName', runs{k} + "k params");
end
hold off;
%set(gca,'YScale','log');
set(gca,'XScale','log');
xlabel('Epoch');
ylabel('Accuracy (%)');
title('Training Accuracy');
legend('Location','best');
grid on;

% Validation Accuracy
nexttile;
hold on;
for k = 1:numel(runs)
    y = hist.("ValAcc" + runs{k});
    x = hist.Iteration;
    valid = isfinite(y);
    % smoothWindow/50 because every 50 train has 1 val.
    ySmooth = smoothdata(y(valid), 'movmean', smoothWindow/50);

    plot(x(valid), ySmooth, ...
        '.-', ...
        'Color', colors(k,:), ...
        'MarkerSize', 3, ...
        'DisplayName', runs{k} + "k params");
end
hold off;
%set(gca,'YScale','log');
set(gca,'XScale','log');
xlabel('Epoch');
ylabel('Accuracy (%)');
title('Validation Accuracy');
legend('Location','best');
grid on;
