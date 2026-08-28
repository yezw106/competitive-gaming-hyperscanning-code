# Social interaction index and ROI brain-behavior analysis

## Purpose

This package constructs the minimap-based social interaction index and the
visual and motor control variables used in the ROI analyses. It then fits the
primary 60-s models, repeats the analysis with 30-s windows, evaluates temporal
dependence with circular-shift permutations.

## Scripts

1. `01_construct_analysis_table.py`
   - Reads sliding-window ROI inter-brain correlation summaries, YOLO-derived
     behavioral events and minimap positions, and MRI-aligned gameplay videos.
   - Computes luminance, color-histogram, and visual-motion similarity; click
     and movement-speed coupling; and the three social-interaction components.
   - Standardizes and averages the three components to create the dyad-window
     social interaction index.
2. `02_fit_primary_60s_models.py`
   - Fits the four-ROI primary models with dyad fixed effects and dyad-clustered
     standard errors.
   - Computes FDR-adjusted tests, incremental explained variance, ROI slope
     contrasts, and component-level sensitivity analyses.
3. `03_fit_30s_sensitivity_models.py`
   - Repeats the main analysis for non-overlapping 30-s windows.
4. `04_run_temporal_permutations.py`
   - Runs 10,000 within-dyad circular-shift permutations using the
     Freedman-Lane procedure.
## Social interaction index

The index is the equal-weight mean of three standardized dyad-level measures:

- `co_presence_ratio`: proportion of samples in a window in which valid player
  and opponent minimap coordinates are simultaneously available.
- `interaction_dynamics`: mean absolute rate of change in player-opponent
  minimap distance, weighted by the co-presence ratio.
- `state_transition_rate`: entries per minute into the close-interaction state,
  defined by the lower tertile of the pooled valid player-opponent distance
  distribution.

The measures are computed from each participant's gameplay record and then
averaged within dyads for each analysis window. Standardization is performed
across all dyad-window observations before the components are averaged.

## Required inputs

The construction script requires:

- A sliding-window ROI inter-brain correlation summary CSV.
- One behavioral-event CSV per participant with columns
  `time, px, py, ex, ey, click, Q, W, E, R, D, F`.
- One minimap-position CSV per participant with columns
  `time_sec, player_cx, player_cy, enemy_cx, enemy_cy`.
- MRI-aligned gameplay recordings.

The statistical scripts require the analysis tables generated for the
corresponding 60-s or 30-s window specification. Input and output paths are
defined near the top of each script and should be set for the local directory
layout before execution.

## Software requirements

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Execution

Set the input paths and runtime variables described in
`01_construct_analysis_table.py`, then run scripts 01 and 02 in order.
For the main analysis, set `VISUAL_SAMPLE_SEC=1.0` and provide the
non-overlapping 60-s ROI summary. For the temporal-scale sensitivity analysis,
construct the corresponding non-overlapping 30-s analysis table before running
`03_fit_30s_sensitivity_models.py`. Run
`04_run_temporal_permutations.py` after the primary 60-s model.

The raw fMRI, behavioral, positional, and video data are not included.

