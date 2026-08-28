clear;

% Summarize the maximum-cluster null distribution for Game 2.
result_root = ['/sharedata/public/LOL_Project/GameHyperscanning2022/', ...
    'process/r_Game2/r_results/', ...
    'permutation_whole_run_residual_rmMotionCSFWM/'];
perm_num = 1000;
max_cluster_distribution = zeros(perm_num, 1);

for i = 1:perm_num
    fprintf('Reading cluster report %d/%d\n', i, perm_num);
    report_file = fullfile(result_root, ['randpairs_', num2str(i)], ...
        'p0001_clusterize.txt');
    if ~exist(report_file, 'file')
        error('Missing cluster report: %s', report_file);
    end

    report = importdata(report_file);
    if isstruct(report) && isfield(report, 'data') && ~isempty(report.data)
        max_cluster_distribution(i) = report.data(1, 1);
    elseif isnumeric(report) && ~isempty(report)
        max_cluster_distribution(i) = report(1, 1);
    else
        % No suprathreshold cluster in this null map.
        max_cluster_distribution(i) = 0;
    end
end

max_cluster_distribution_sorted = sort(max_cluster_distribution);
cluster_extent_threshold = ceil(prctile(max_cluster_distribution, 95));

save(fullfile(result_root, 'maxcluster_distribution.mat'), ...
    'max_cluster_distribution', ...
    'max_cluster_distribution_sorted', ...
    'cluster_extent_threshold');

distribution_file = fullfile(result_root, 'maxcluster_distribution.txt');
fid = fopen(distribution_file, 'w');
if fid == -1
    error('Cannot open output file: %s', distribution_file);
end
fprintf(fid, '%d\n', max_cluster_distribution);
fclose(fid);

threshold_file = fullfile(result_root, ...
    'cluster_extent_threshold_p05.txt');
fid = fopen(threshold_file, 'w');
if fid == -1
    error('Cannot open output file: %s', threshold_file);
end
fprintf(fid, '%d\n', cluster_extent_threshold);
fclose(fid);

fprintf('Cluster-level p < .05 extent threshold: %d voxels\n', ...
    cluster_extent_threshold);
