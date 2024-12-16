clc; close all; clearvars;
%%
APMS_table = [];
for i = 44:44
    vid_num = num2str(i);
    if i < 10
        vid_num = "0" + vid_num;
    end
    outTable = ExtractAPMs(char(vid_num), true);
    APMS_table = [APMS_table; outTable(end,1:end-2)];
    APMS_table.Row{i} = char(vid_num);
end
writetable(APMS_table, 'results/APMs_tot.csv', 'WriteRowNames',true)
%% load tracking data
function outTable = ExtractAPMs(vid_num, plot_flag)
if nargin < 2
    plot_flag = false;
end
% vid_num = '03';
vid_path = ['../cholec80/videos/video' vid_num '.mp4'];
vidObj = VideoReader(vid_path);
track_res_path = ['results/tracker_bytetrack_res_video' vid_num '.csv'];
T = readtable(track_res_path);
%% calculate and plot heatmaps
width = vidObj.Width;
height = vidObj.Height;

classes = unique(T.class);
tools_heatmaps = cell(length(classes), 1);
heat = zeros(height, width);

% acquire heatmap matrix for each tool
if plot_flag
    figure;
end
for i = 1:length(classes)
    class_lines = find(ismember(T.class, classes{i}))';
    for n = class_lines
        if ~strcmp(T.class{n}, classes{i})
            continue;
        end
        x_min = T.bbox_x(n);
        y_min = T.bbox_y(n);
        x_max = T.bbox_x(n) + T.bbox_w(n);
        y_max = T.bbox_y(n) + T.bbox_h(n);
        for x = x_min:x_max
            for y = y_min:y_max
                y_id = min(y+1, size(heat,1));
                x_id = min(x+1, size(heat,2));
                heat(y_id, x_id) = heat(y_id, x_id) + 1;
            end
        end
    end
    tools_heatmaps{i,1} = heat;

    % plot heatmap
    if plot_flag
        nexttile;
        imagesc(heat);
        title(classes{i});
        colormap('hot');
        colorbar;
        axis image; axis off;
    end
end
if plot_flag
    nexttile;
    tot_heat = sum(heat, 3);
    imagesc(tot_heat);
    title('Total heatmap');
    colormap('hot');
    colorbar;
    axis image; axis off;
    sgtitle(['Heatmaps of laparoscopic tools in video' vid_num]);
    saveas(gcf, ['results/heatmaps_video' vid_num '.png']);
end
%% calculate working area

tools_working_area = nan(length(classes),1);
working_area_high_threshold = 0.975*100;
working_area_low_threshold = 0.025*100;

for i = 1:length(classes)
    class_lines = ismember(T.class, classes{i});

    % calculate working area of each tool by 97.5%-2.5% of the locations
    x_pos = T.trace_x(class_lines);
    y_pos = T.trace_y(class_lines);
    % calculate 97.5 percentile positions
    x_97_5 = prctile(x_pos, working_area_high_threshold);
    y_97_5 = prctile(y_pos, working_area_high_threshold);
    % calculate 2.5 percentile positions
    x_2_5 = prctile(x_pos, working_area_low_threshold);
    y_2_5 = prctile(y_pos, working_area_low_threshold);
    wa = round((x_97_5 - x_2_5) * (y_97_5 - y_2_5));
    tools_working_area(i) = wa;
end

%% calculate path length
order = 3; frame_len = 31; % golay filter parameters
tools_pl = nan(length(classes), 1);
if plot_flag
    figure; hold on;
end
for i = 1:length(classes)
    class_lines = ismember(T.class, classes{i});
    x_pos = T.trace_x(class_lines);
    y_pos = T.trace_y(class_lines);
    % apply sacitzky golay filter
    if length(x_pos) > frame_len
        x_pos = sgolayfilt(x_pos, order, frame_len);
        y_pos = sgolayfilt(y_pos, order, frame_len);
    end
    % plot path traces
    if plot_flag
        if strcmpi(classes{i}, 'SpecimenBag') || strcmpi(classes{i}, 'Irrigator')
%             plot(x_pos(1:2*60*vidObj.FrameRate), y_pos(1:2*60*vidObj.FrameRate), 'DisplayName', classes{i});
            plot(x_pos, y_pos, 'LineWidth', 2, 'DisplayName', classes{i});
        end
    end
    % ------------------- for tuning of salitzky golay parameters
    % %     figure;
    % %     for ord = 2:4
    % %         for len = [11 15 21 31]
    % %             sgf = sgolayfilt(x_pos, ord, len);
    % %             nexttile;
    % %             plot(x_pos,':')
    % %             hold on
    % %             plot(sgf,'.-')
    % %             title(['order: ' num2str(ord) ' len: ' num2str(len)]);
    % %         end
    % %     end
    % -----------------
    frames_id = T.frame(class_lines);
    % find continuously sections of tool
    frames_id_diff = diff(frames_id);
    % init path length
    path_length = 0;
    curr_id = 1;
    % loop over all segments of tool to calculate path length
    while curr_id < length(frames_id)
        next_segment_id = curr_id + find(frames_id_diff(curr_id:end) > 1, 1) - 1;
        x_diff = diff(x_pos(curr_id:next_segment_id));
        y_diff = diff(y_pos(curr_id:next_segment_id));
        % add path distance
        curr_segment_path_length = sum(sqrt(x_diff.^2 + y_diff.^2));
        path_length = path_length + curr_segment_path_length;
        curr_id = next_segment_id + 1;
    end
    tools_pl(i) = path_length / (vidObj.Duration / 60);
end
if plot_flag
    hold off;
    legend;
    grid on; axis ij;
    xlim([0, vidObj.Width]); ylim([0 vidObj.Height]);
    title('Path trace');
end
%% calculate average velocity, acceleration, jerk
order = 3; frame_len = 31; % golay filter parameters
tools_avg_vel = nan(length(classes), 1);
tools_avg_acc = nan(length(classes), 1);
tools_avg_jerk = nan(length(classes), 1);
for i = 1:length(classes)
    class_lines = ismember(T.class, classes{i});
    vel = T.velocity(class_lines);
    acc = T.acceleration(class_lines);
    jerk = T.jitter(class_lines);

    % apply sacitzky golay filter
    if length(vel) > frame_len
        vel = sgolayfilt(vel, order, frame_len);
    end
    if length(acc) > frame_len
        acc = sgolayfilt(acc, order, frame_len);
    end
    if length(jerk) > frame_len
        jerk = sgolayfilt(jerk, order, frame_len);
    end

    avg_vel = mean(abs(vel),'omitnan');
    avg_acc = mean(abs(acc), 'omitnan');
    avg_jerk = mean(abs(jerk), 'omitnan');
    tools_avg_vel(i) = avg_vel;
    tools_avg_acc(i) = avg_acc;
    tools_avg_jerk(i) = avg_jerk;
end

%% calculate time distribution of each tool
tools_time_duration = nan(length(classes), 1);
tools_time_distribution = nan(length(classes), 1);
for i = 1:length(classes)
    class_lines = ismember(T.class, classes{i});
    % total duration of tool appearance in seconds
    t_duration = nnz(class_lines) / vidObj.FrameRate;
    % total duration of tool appearance in relation
    t_normalized = t_duration / vidObj.Duration;

    tools_time_duration(i) = t_duration;
    tools_time_distribution(i) = t_normalized;
end
if plot_flag
    figure;
    bar(categorical(classes), tools_time_distribution);
    title(['Time distribution of detected tools in video' vid_num])
    saveas(gcf, ['results/time_dist_video' vid_num '.png']);
end
%% write APM to output
outArray = [tools_working_area,...
    tools_pl,...
    tools_avg_vel,...
    tools_avg_acc,...
    tools_avg_jerk,...
    tools_time_duration,...
    tools_time_distribution];
overall_working_area = tools_time_distribution' * tools_working_area;
overall_pl = tools_time_distribution' * tools_pl;
overall_avg_vel = tools_time_distribution' * tools_avg_vel;
overall_avg_acc = tools_time_distribution' * tools_avg_acc;
overall_avg_jerk = tools_time_distribution' * tools_avg_jerk;
outArray = [outArray;...
    overall_working_area, overall_pl, overall_avg_vel, overall_avg_acc, overall_avg_jerk, nan, nan];
APM_names = {'WorkingArea', 'PathLength', 'AvgVel', 'AvgAcc', 'AvgJerk', 'TimeDuration','TimeDist'};
outTable = array2table(outArray, 'VariableNames', APM_names, 'RowNames',[classes; 'Total']);
writetable(outTable, ['results/APMs_video' vid_num '.csv'], 'WriteRowNames',true);
end