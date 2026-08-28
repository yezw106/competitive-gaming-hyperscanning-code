clear;

% Compute voxel-wise correlations for 1,000 pseudo-dyad groups in Game 2.
base_dir = '/sharedata/public/LOL_Project/GameHyperscanning2022/process/r_Game2';
tc_dir = fullfile(base_dir, 'whole_run_residual_rmMotionCSFWM');
pair_list_dir = fullfile(base_dir, 'randpairLists');
out_root = fullfile(base_dir, 'r_results', ...
    'permutation_whole_run_residual_rmMotionCSFWM');
subjects_file = fullfile(base_dir, 'subjects.txt');
perm_num = 1000;

if ~exist(out_root, 'dir')
    mkdir(out_root);
end

% Read all subject IDs and load each subject's dumped voxel time series once.
subjects_raw = importdata(subjects_file);
if isstruct(subjects_raw)
    subjects_raw = subjects_raw.textdata;
end
if ischar(subjects_raw)
    subjects = cellstr(subjects_raw);
else
    subjects = subjects_raw;
end
subjects = cellfun(@strtrim, subjects, 'UniformOutput', false);

n_subjects = numel(subjects);
subject_tc = cell(n_subjects, 1);
for s = 1:n_subjects
    subject_id = subjects{s};
    tc_file = fullfile(tc_dir, [subject_id, '.txt']);
    if ~exist(tc_file, 'file')
        error('Missing time-series file: %s', tc_file);
    end
    fprintf('Loading %s (%d/%d)\n', subject_id, s, n_subjects);
    subject_tc{s} = load(tc_file);
end

for p = 1:perm_num
    pair_file = fullfile(pair_list_dir, ...
        ['randpairs_', num2str(p), '.txt']);
    if ~exist(pair_file, 'file')
        error('Missing pseudo-dyad list: %s', pair_file);
    end

    pairs_raw = importdata(pair_file);
    if isstruct(pairs_raw)
        pairs_raw = pairs_raw.textdata;
    end
    if ischar(pairs_raw)
        pairs = cellstr(pairs_raw);
    else
        pairs = pairs_raw;
    end
    pairs = cellfun(@strtrim, pairs, 'UniformOutput', false);

    out_dir = fullfile(out_root, ['randpairs_', num2str(p)]);
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    fprintf('Permutation group %d/%d\n', p, perm_num);
    for i = 1:numel(pairs)
        pair_id = pairs{i};
        if numel(pair_id) ~= 10
            error('Invalid pseudo-dyad ID in %s: %s', pair_file, pair_id);
        end

        sub1 = pair_id(1:5);
        sub2 = pair_id(6:10);
        idx1 = find(strcmp(subjects, sub1), 1);
        idx2 = find(strcmp(subjects, sub2), 1);
        if isempty(idx1) || isempty(idx2)
            error('Subject not listed in %s: %s', subjects_file, pair_id);
        end

        sub1_tc = subject_tc{idx1};
        sub2_tc = subject_tc{idx2};
        if size(sub1_tc, 1) ~= size(sub2_tc, 1)
            error('Voxel-count mismatch for pseudo-dyad %s.', pair_id);
        end
        if ~isequal(sub1_tc(:, 1:3), sub2_tc(:, 1:3))
            error('Voxel-coordinate mismatch for pseudo-dyad %s.', pair_id);
        end

        % Columns 1-3 are voxel coordinates. Retain the common temporal
        % overlap and discard any extra trailing volumes from the longer run.
        last_col = min(size(sub1_tc, 2), size(sub2_tc, 2));
        if last_col < 5
            error('Fewer than two time points for pseudo-dyad %s.', pair_id);
        end

        n_voxels = size(sub1_tc, 1);
        r_matrix = zeros(n_voxels, 4);
        z_matrix = zeros(n_voxels, 4);
        r_matrix(:, 1:3) = sub1_tc(:, 1:3);
        z_matrix(:, 1:3) = sub1_tc(:, 1:3);

        for voxel = 1:n_voxels
            series1 = sub1_tc(voxel, 4:last_col);
            series2 = sub2_tc(voxel, 4:last_col);
            r_pair = corrcoef(series1, series2);
            r_value = r_pair(1, 2);
            r_matrix(voxel, 4) = r_value;
            z_matrix(voxel, 4) = atanh(r_value);
        end

        r_file = fullfile(out_dir, [pair_id, '_r.txt']);
        z_file = fullfile(out_dir, [pair_id, '_z.txt']);

        fid = fopen(r_file, 'w');
        if fid == -1
            error('Cannot open output file: %s', r_file);
        end
        fprintf(fid, '%.6f\t%.6f\t%.6f\t%.6f\n', r_matrix');
        fclose(fid);

        fid = fopen(z_file, 'w');
        if fid == -1
            error('Cannot open output file: %s', z_file);
        end
        fprintf(fid, '%.6f\t%.6f\t%.6f\t%.6f\n', z_matrix');
        fclose(fid);
    end
end
