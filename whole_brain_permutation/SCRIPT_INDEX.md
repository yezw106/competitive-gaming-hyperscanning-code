# Script index

| Script | Function | Main input | Main output |
|---|---|---|---|
| step00_optional_generate_pseudo_dyad_lists.m | Generates 1,000 pseudo-dyad groups while excluding interacting dyads in either participant order | subjects.txt; pairs.txt | randpairLists/randpairs_1.txt to randpairs_1000.txt |
| step01_extract_voxel_timeseries.sh | Extracts masked voxel time series for all participants | residual_Game2_rmMotionCSFWM.nii.gz; subjects.txt; brain mask | whole_run_residual_rmMotionCSFWM/GHxxx.txt |
| step02_compute_interacting_dyad_correlations.m | Computes voxel-wise Pearson-r and Fisher-z values for interacting dyads | participant time-series files; pairs.txt | paired result *_r.txt and *_z.txt files |
| step03_compute_pseudo_dyad_correlations.m | Computes voxel-wise Pearson-r and Fisher-z values for every pseudo-dyad group | participant time-series files; randpairLists | pseudo-group *_r.txt and *_z.txt files |
| step04_reconstruct_interacting_dyad_maps.sh | Reconstructs interacting-dyad Pearson-r maps | interacting-dyad *_r.txt files | interacting-dyad *_r.nii files |
| step05_reconstruct_pseudo_dyad_maps.sh | Reconstructs pseudo-dyad Pearson-r maps | pseudo-dyad *_r.txt files | pseudo-dyad *_r.nii files |
| step06_average_interacting_dyad_maps.sh | Averages interacting-dyad Pearson-r maps | interacting-dyad *_r.nii files | paired_meanr.nii and paired_meanr.txt |
| step07_average_pseudo_dyad_maps.sh | Averages Pearson-r maps within each pseudo group | pseudo-dyad *_r.nii files | randpair_meanr_i.nii and randpair_meanr_i.txt |
| step08_fit_real_map_voxelwise_cdf.m | Fits each voxel's pseudo-group distribution and evaluates the interacting-group mean | paired_meanr.txt; 1,000 pseudo-group mean files | fitted and voxel-thresholded CDF text map |
| step09_reconstruct_real_fitted_cdf_map.sh | Reconstructs the fitted CDF text map | fitted and voxel-thresholded CDF text map | fitted CDF NIfTI map |
| step10_compute_pseudo_group_empirical_cdf.m | Converts every pseudo-group mean to an empirical voxel-wise CDF map | 1,000 pseudo-group mean files | permutation_1000_p.txt in each pseudo-group directory |
| step11_reconstruct_pseudo_group_cdf_maps.sh | Reconstructs each pseudo-group CDF map | pseudo-group permutation_1000_p.txt files | pseudo-group permutation_1000_p.nii files |
| step12_cluster_pseudo_group_cdf_maps.sh | Thresholds and clusters each pseudo-group CDF map using NN=2 | pseudo-group CDF NIfTI maps | p0001_clusterize.txt in each pseudo-group directory |
| step13_summarize_max_cluster_distribution.m | Collects maximum cluster sizes and calculates the 95th-percentile extent threshold | 1,000 cluster reports | maximum-cluster distribution and extent-threshold files |
| step14_apply_cluster_extent_to_real_map.sh | Applies the extent threshold to the interacting-dyad fitted-CDF map | fitted CDF map; extent threshold | cluster labels, report, and corrected CDF map |
