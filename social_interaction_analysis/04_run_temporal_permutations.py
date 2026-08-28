from pathlib import Path
import os

import numpy as np
import pandas as pd
import patsy
from statsmodels.stats.multitest import multipletests


ROOT = Path(os.environ.get("ANALYSIS_ROOT", Path(__file__).resolve().parent))
SOURCE_DIR = (
    ROOT
    / "behavior_time_effect_analysis"
    / "sliding_window_direct_similarity_model_w60_s60_4roi_visual1s"
)
INPUT_CSV = SOURCE_DIR / "direct_similarity_brain_behavior_dataset_4roi.csv"
OUT_DIR = (
    ROOT
    / "behavior_time_effect_analysis"
    / "circular_shift_permutation_w60_s60_4roi"
)
N_PERMUTATIONS = 10000
SEED = 20260624

ROI_MAP = {
    "mpfc": "mPFC",
    "precuneus_p0005_intersect": "Precuneus",
    "rtpj_p0005_neurosynth_intersect": "rTPJ",
    "ba17": "BA17",
}
ROI_ORDER = ["mPFC", "Precuneus", "rTPJ", "BA17"]
CONTRAST_ROIS = ["mPFC", "rTPJ", "BA17"]
CONTROLS = [
    "z_luminance_similarity",
    "z_color_hist_similarity",
    "z_visual_motion_similarity",
    "z_click_coupling_z",
    "z_movement_speed_coupling_z",
    "time_z",
    "time_z2",
]
COMPLETE = ["brain_ibc", "social_interaction_index", *CONTROLS]


def empirical_p(null, observed, alternative):
    if alternative == "greater":
        extreme = null >= observed
    elif alternative == "less":
        extreme = null <= observed
    else:
        center = null.mean()
        extreme = np.abs(null - center) >= abs(observed - center)
    return (np.count_nonzero(extreme) + 1) / (len(null) + 1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_CSV)
    data["roi_label_corrected"] = data["roi"].map(ROI_MAP)
    data = data.dropna(subset=COMPLETE).copy()

    template = (
        data[data["roi_label_corrected"] == "Precuneus"]
        .sort_values(["pair", "window_index"])
        .reset_index(drop=True)
    )
    design = np.asarray(
        patsy.dmatrix(
            "1 + " + " + ".join(CONTROLS) + " + C(pair)",
            template,
            return_type="dataframe",
        )
    )
    q, _ = np.linalg.qr(design, mode="reduced")

    def residualize(values):
        return values - q @ (q.T @ values)

    x_res = residualize(
        template["social_interaction_index"].to_numpy(dtype=float)
    )
    denominator = np.dot(x_res, x_res)

    y_res = {}
    observed = {}
    for roi in ROI_ORDER:
        roi_data = (
            data[data["roi_label_corrected"] == roi]
            .sort_values(["pair", "window_index"])
            .reset_index(drop=True)
        )
        if not np.array_equal(
            roi_data[["pair", "window_index"]].to_numpy(),
            template[["pair", "window_index"]].to_numpy(),
        ):
            raise ValueError(f"Alignment failed: {roi}")
        y_res[roi] = residualize(roi_data["brain_ibc"].to_numpy(dtype=float))
        observed[roi] = np.dot(x_res, y_res[roi]) / denominator

    pair_indices = [
        group.index.to_numpy()
        for _, group in template.groupby("pair", sort=True)
    ]
    rng = np.random.default_rng(SEED)
    null = {roi: np.empty(N_PERMUTATIONS) for roi in ROI_ORDER}

    for permutation in range(N_PERMUTATIONS):
        shifts = [
            int(rng.integers(1, len(indices))) for indices in pair_indices
        ]
        for roi in ROI_ORDER:
            shifted = np.empty_like(y_res[roi])
            for indices, amount in zip(pair_indices, shifts):
                shifted[indices] = np.roll(y_res[roi][indices], amount)
            # Freedman-Lane: add shifted reduced-model residuals to reduced
            # fitted values, then refit. The fitted part is annihilated by M_W.
            y_star_res = residualize(shifted)
            null[roi][permutation] = (
                np.dot(x_res, y_star_res) / denominator
            )
        if (permutation + 1) % 1000 == 0:
            print(f"Completed {permutation + 1}/{N_PERMUTATIONS}", flush=True)

    main_rows = []
    for roi in ROI_ORDER:
        values = null[roi]
        main_rows.append(
            {
                "roi": roi,
                "observed_beta": observed[roi],
                "null_mean": values.mean(),
                "null_sd": values.std(ddof=1),
                "null_ci2.5": np.quantile(values, 0.025),
                "null_ci97.5": np.quantile(values, 0.975),
                "p_directional": empirical_p(
                    values,
                    observed[roi],
                    "greater" if roi == "Precuneus" else "two-sided",
                ),
                "p_two_sided": empirical_p(values, observed[roi], "two-sided"),
                "n_permutations": N_PERMUTATIONS,
            }
        )
    main_result = pd.DataFrame(main_rows)

    contrast_rows = []
    for roi in CONTRAST_ROIS:
        values = null[roi] - null["Precuneus"]
        observed_diff = observed[roi] - observed["Precuneus"]
        contrast_rows.append(
            {
                "contrast": f"{roi} minus Precuneus",
                "roi_minus_precuneus": roi,
                "observed_beta_difference": observed_diff,
                "null_mean": values.mean(),
                "null_sd": values.std(ddof=1),
                "null_ci2.5": np.quantile(values, 0.025),
                "null_ci97.5": np.quantile(values, 0.975),
                "p_directional_less_than_zero": empirical_p(
                    values, observed_diff, "less"
                ),
                "p_two_sided": empirical_p(
                    values, observed_diff, "two-sided"
                ),
                "n_permutations": N_PERMUTATIONS,
            }
        )
    contrast_result = pd.DataFrame(contrast_rows)
    contrast_result["p_directional_fdr_3"] = multipletests(
        contrast_result["p_directional_less_than_zero"], method="fdr_bh"
    )[1]
    contrast_result["p_two_sided_fdr_3"] = multipletests(
        contrast_result["p_two_sided"], method="fdr_bh"
    )[1]

    null_df = pd.DataFrame(
        {
            "permutation": np.arange(1, N_PERMUTATIONS + 1),
            **{f"beta_{roi}": null[roi] for roi in ROI_ORDER},
            **{
                f"diff_{roi}_minus_Precuneus":
                    null[roi] - null["Precuneus"]
                for roi in CONTRAST_ROIS
            },
        }
    )
    main_result.to_csv(
        OUT_DIR / "freedman_lane_circular_shift_social_beta_results.csv",
        index=False,
    )
    contrast_result.to_csv(
        OUT_DIR / "freedman_lane_circular_shift_roi_contrasts.csv",
        index=False,
    )
    null_df.to_csv(
        OUT_DIR / "freedman_lane_circular_shift_null_distributions.csv",
        index=False,
    )
    print("\nFreedman-Lane ROI results")
    print(main_result.to_string(index=False))
    print("\nFreedman-Lane ROI contrasts")
    print(contrast_result.to_string(index=False))


if __name__ == "__main__":
    main()

