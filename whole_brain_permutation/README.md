# Game2 whole-run voxel-wise inter-brain correlation permutation pipeline

## Purpose

These scripts perform the Continuous Gaming (Game2) whole-run voxel-wise
inter-brain correlation permutation analysis and cluster-level correction.

## Analysis procedure

1. Extract masked voxel time series from each participant's preprocessed Game2
   dataset.
2. Calculate spatially corresponding voxel-wise Pearson correlations for every
   interacting dyad.
3. Calculate the same correlations for 1,000 groups of pseudo-dyads.
4. Average the Pearson-r maps across interacting dyads and separately within
   each pseudo-dyad group.
5. At every voxel, fit a normal distribution to the 1,000 pseudo-group means
   and evaluate the interacting-group mean under that distribution.
6. Retain voxels with fitted CDF > .999 and store positive values as
   (CDF - .999) * 10000.
7. Convert each pseudo-group mean map to an empirical voxel-wise CDF map.
8. Threshold each pseudo-group CDF map at .999, form clusters using AFNI NN=2,
   and record its largest cluster.
9. Use the 95th percentile of the 1,000 maximum cluster sizes as the
   cluster-level p < .05 extent threshold.
10. Apply the extent threshold to the interacting-dyad fitted-CDF map.

Files named permutation_1000_p contain CDF or 1-p values. Larger values
therefore indicate stronger evidence against the pseudo-dyad null distribution.

## Required inputs

- /sharedata/public/LOL_Project/GameHyperscanning2022/process/r_Game2/subjects.txt
- /sharedata/public/LOL_Project/GameHyperscanning2022/process/r_Game2/pairs.txt
- /sharedata/public/LOL_Project/GameHyperscanning2022/process/r_Game2/randpairLists/randpairs_1.txt through randpairs_1000.txt
- /sharedata/public/LOL_Project/GameHyperscanning2022/process/r_Game2/scripts/MNI152_2.5mm_brain.nii.gz
- One participant dataset at:
  /sharedata/public/LOL_Project/GameHyperscanning2022/process/GHxxx/residual_Game2_rmMotionCSFWM.nii.gz

A given analysis run must use the same 1,000 pseudo-dyad lists throughout all
steps. The optional list generator is only needed when these files do not
already exist.

## Software requirements

- Linux
- AFNI
- MATLAB with the Statistics and Machine Learning Toolbox

## Running the scripts

Run the commands from this directory. After transfer to Linux, make the shell
scripts executable:

    chmod +x ./*.sh

Run the pseudo-dyad list generator only when randpairLists are absent:

    matlab -nodisplay -nosplash -r "run('step00_optional_generate_pseudo_dyad_lists.m'); exit"

Run the analysis in this order:

    bash step01_extract_voxel_timeseries.sh

    matlab -nodisplay -nosplash -r "run('step02_compute_interacting_dyad_correlations.m'); exit"
    matlab -nodisplay -nosplash -r "run('step03_compute_pseudo_dyad_correlations.m'); exit"

    bash step04_reconstruct_interacting_dyad_maps.sh
    bash step05_reconstruct_pseudo_dyad_maps.sh

    bash step06_average_interacting_dyad_maps.sh
    bash step07_average_pseudo_dyad_maps.sh

    matlab -nodisplay -nosplash -r "run('step08_fit_real_map_voxelwise_cdf.m'); exit"
    bash step09_reconstruct_real_fitted_cdf_map.sh

    matlab -nodisplay -nosplash -r "run('step10_compute_pseudo_group_empirical_cdf.m'); exit"
    bash step11_reconstruct_pseudo_group_cdf_maps.sh
    bash step12_cluster_pseudo_group_cdf_maps.sh
    matlab -nodisplay -nosplash -r "run('step13_summarize_max_cluster_distribution.m'); exit"

    bash step14_apply_cluster_extent_to_real_map.sh

## Main outputs

Interacting-dyad mean map:

    r_results/paired_whole_run_residual_rmMotionCSFWM/paired_meanr.nii
    r_results/paired_whole_run_residual_rmMotionCSFWM/paired_meanr.txt

Voxel-thresholded fitted-CDF map:

    r_results/permutation_whole_run_residual_rmMotionCSFWM/
    permutation_1000_p_fit_threshold0.001_scale10000.nii

Maximum-cluster distribution and extent threshold:

    r_results/permutation_whole_run_residual_rmMotionCSFWM/maxcluster_distribution.mat
    r_results/permutation_whole_run_residual_rmMotionCSFWM/maxcluster_distribution.txt
    r_results/permutation_whole_run_residual_rmMotionCSFWM/cluster_extent_threshold_p05.txt

Cluster-corrected fitted-CDF map:

    r_results/permutation_whole_run_residual_rmMotionCSFWM/
    permutation_1000_p_fit_threshold0.001_scale10000_cluster_corrected.nii.gz

## Data-handling details

- Correlations are calculated over the common temporal overlap within each
  dyad. Extra trailing volumes from the longer time series are not used.
- The permutation statistic is the mean Pearson-r map.
- Pair-level Fisher-z maps are also saved but are not used in the permutation
  mean maps.
- NN=2 treats face- and edge-connected voxels as belonging to the same cluster.
- With 17 dyads per pseudo group, the reconstruction step creates 17,000
  pseudo-dyad NIfTI correlation maps.
