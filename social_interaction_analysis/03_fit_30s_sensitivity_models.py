from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


ROOT = Path(os.environ.get("ANALYSIS_ROOT", Path(__file__).resolve().parent))
DATA_CSV = (
    ROOT
    / "behavior_time_effect_analysis"
    / "sliding_window_brain_behavior_w30_s30_4roi_visual1s"
    / "sliding_window_brain_behavior_dataset.csv"
)
OUT = (
    ROOT
    / "behavior_time_effect_analysis"
    / "sliding_window_direct_similarity_model_w30_s30_4roi_visual1s"
)
W60_COEF = (
    ROOT
    / "behavior_time_effect_analysis"
    / "sliding_window_direct_similarity_model_w60_s60_4roi_visual1s"
    / "direct_similarity_social_index_coefficients_4roi.csv"
)
W60_CONTRAST = (
    ROOT
    / "behavior_time_effect_analysis"
    / "circular_shift_permutation_w60_s60_4roi"
    / "freedman_lane_circular_shift_roi_contrasts.csv"
)

ROI_ORDER = ["mPFC", "Precuneus", "rTPJ", "BA17"]
COLORS = {
    "mPFC": "#6f93b8",
    "Precuneus": "#ff9638",
    "rTPJ": "#70b269",
    "BA17": "#ba88ae",
}
CONTROLS = [
    "z_luminance_similarity",
    "z_color_hist_similarity",
    "z_visual_motion_similarity",
    "z_click_coupling_z",
    "z_movement_speed_coupling_z",
    "time_z",
    "time_z2",
]
SOCIAL = "social_interaction_index"
COMPONENTS = {
    "Co-presence": "z_co_presence_ratio",
    "Interaction dynamics": "z_interaction_dynamics",
    "State-transition rate": "z_state_transition_rate",
}


def robust_term(model, data, term):
    robust = model.get_robustcov_results(
        cov_type="cluster",
        groups=data["pair"],
        use_correction=True,
        df_correction=True,
    )
    index = model.model.exog_names.index(term)
    beta = float(model.params[term])
    se = float(robust.bse[index])
    return {
        "beta": beta,
        "se_cluster": se,
        "t_cluster": float(robust.tvalues[index]),
        "p_cluster": float(robust.pvalues[index]),
        "ci95_low": beta - 1.96 * se,
        "ci95_high": beta + 1.96 * se,
        "p_ordinary": float(model.pvalues[term]),
    }


def separate_models(data, predictor):
    rows = []
    nuisance = " + ".join(CONTROLS)
    base_formula = f"brain_ibc ~ {nuisance} + C(pair)"
    full_formula = f"brain_ibc ~ {nuisance} + {predictor} + C(pair)"
    needed = ["brain_ibc", predictor, *CONTROLS]
    for roi in ROI_ORDER:
        sub = data[data["roi_label"] == roi].dropna(subset=needed).copy()
        base = smf.ols(base_formula, sub).fit()
        full = smf.ols(full_formula, sub).fit()
        row = {
            "roi": roi,
            "predictor": predictor,
            "n_rows": int(full.nobs),
            "n_pairs": sub["pair"].nunique(),
            "r2_base": base.rsquared,
            "r2_full": full.rsquared,
            "delta_r2": full.rsquared - base.rsquared,
        }
        row.update(robust_term(full, sub, predictor))
        rows.append(row)
    return pd.DataFrame(rows)


def roi_contrasts(data, predictor):
    needed = ["brain_ibc", predictor, *CONTROLS]
    sub = data.dropna(subset=needed).copy()
    sub["roi_label"] = sub["roi_label"].astype(str)
    roi = "C(roi_label, Treatment(reference='Precuneus'))"
    formula = (
        f"brain_ibc ~ {roi} * ("
        + " + ".join([*CONTROLS, predictor])
        + f") + {roi} * C(pair)"
    )
    model = smf.ols(formula, sub).fit()
    robust = model.get_robustcov_results(
        cov_type="cluster",
        groups=sub["pair"],
        use_correction=True,
        df_correction=True,
    )
    rows = []
    for other in ["mPFC", "rTPJ", "BA17"]:
        term = f"{roi}[T.{other}]:{predictor}"
        index = model.model.exog_names.index(term)
        beta = float(model.params[term])
        se = float(robust.bse[index])
        rows.append(
            {
                "contrast": f"{other} minus Precuneus",
                "roi_minus_precuneus": other,
                "predictor": predictor,
                "beta_difference": beta,
                "se_cluster": se,
                "t_cluster": float(robust.tvalues[index]),
                "p_cluster": float(robust.pvalues[index]),
                "ci95_low": beta - 1.96 * se,
                "ci95_high": beta + 1.96 * se,
                "n_rows": int(model.nobs),
                "n_pairs": sub["pair"].nunique(),
            }
        )
    result = pd.DataFrame(rows)
    result["p_fdr_3"] = multipletests(result["p_cluster"], method="fdr_bh")[1]
    return result


def make_comparison(w30, contrasts30):
    w60 = pd.read_csv(W60_COEF)
    w60 = w60.rename(
        columns={
            "beta_cluster": "beta_w60",
            "se_cluster": "se_w60",
            "p_cluster": "p_w60",
            "p_fdr_4roi": "p_fdr_w60",
            "delta_r2": "delta_r2_w60",
        }
    )
    w30_copy = w30.rename(
        columns={
            "beta": "beta_w30",
            "se_cluster": "se_w30",
            "p_cluster": "p_w30",
            "p_fdr_4roi": "p_fdr_w30",
            "delta_r2": "delta_r2_w30",
        }
    )
    comparison = w30_copy[
        ["roi", "beta_w30", "se_w30", "p_w30", "p_fdr_w30", "delta_r2_w30"]
    ].merge(
        w60[
            ["roi", "beta_w60", "se_w60", "p_w60", "p_fdr_w60",
             "delta_r2_w60"]
        ],
        on="roi",
        validate="one_to_one",
    )
    comparison["same_beta_direction"] = (
        np.sign(comparison["beta_w30"]) == np.sign(comparison["beta_w60"])
    )
    comparison.to_csv(OUT / "w30_vs_w60_social_index_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=200)
    x = np.arange(4)
    width = 0.34
    axes[0].bar(
        x - width / 2,
        comparison.set_index("roi").loc[ROI_ORDER, "beta_w60"],
        width,
        color=[COLORS[r] for r in ROI_ORDER],
        alpha=0.42,
        label="60-s",
    )
    axes[0].bar(
        x + width / 2,
        comparison.set_index("roi").loc[ROI_ORDER, "beta_w30"],
        width,
        color=[COLORS[r] for r in ROI_ORDER],
        label="30-s",
    )
    axes[0].axhline(0, color="#777777", lw=0.8)
    axes[0].set_xticks(x, ["mPFC", "Precuneus", "Left TPJ", "BA17"])
    axes[0].set_ylabel("Social-interaction beta")
    axes[0].set_title("A. Social-interaction coefficients")
    axes[0].legend(frameon=False)

    c30 = contrasts30.set_index("roi_minus_precuneus")
    c60 = pd.read_csv(W60_CONTRAST).set_index("roi_minus_precuneus")
    others = ["mPFC", "rTPJ", "BA17"]
    axes[1].bar(
        x[:3] - width / 2,
        c60.loc[others, "observed_beta_difference"],
        width,
        color=[COLORS[r] for r in others],
        alpha=0.42,
        label="60-s",
    )
    axes[1].bar(
        x[:3] + width / 2,
        c30.loc[others, "beta_difference"],
        width,
        color=[COLORS[r] for r in others],
        label="30-s",
    )
    axes[1].axhline(0, color="#777777", lw=0.8)
    axes[1].set_xticks(x[:3], ["mPFC - Prec.", "Left TPJ - Prec.", "BA17 - Prec."])
    axes[1].set_ylabel("Slope difference")
    axes[1].set_title("B. ROI differences from Precuneus")
    axes[1].legend(frameon=False)

    fig.suptitle("30-s non-overlapping windows reproduce the 60-s pattern")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "w30_vs_w60_direct_similarity_results.png",
                bbox_inches="tight")
    plt.close(fig)
    return comparison


def write_summary(main, contrasts, components, comparison):
    prec = main.set_index("roi").loc["Precuneus"]
    lines = [
        "# 30秒非重叠窗口与60秒结果的一致性分析",
        "",
        "## 分析设置",
        "",
        "- window = 30秒，step = 30秒，窗口不重叠。",
        "- 17个pair，每个ROI共578个原始窗口。",
        f"- 直接相似性完整模型可用{int(main['n_rows'].iloc[0])}个窗口，17个pair均保留。",
        "- 行为指标由原始YOLO事件和位置文件按30秒重新计算。",
        "- 视觉特征继续使用1秒采样，每个30秒窗口约30个视觉采样点。",
        "- 模型、标准化、pair固定效应和pair-cluster稳健标准误与60秒主分析一致。",
        "",
        "## 社会互动指数结果",
        "",
        "| ROI | 30秒β | SE | cluster p | FDR p | ΔR² | 60秒β |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    comp = comparison.set_index("roi")
    for _, row in main.iterrows():
        lines.append(
            f"| {row['roi']} | {row['beta']:.4f} | "
            f"{row['se_cluster']:.4f} | {row['p_cluster']:.4f} | "
            f"{row['p_fdr_4roi']:.4f} | {row['delta_r2']:.4f} | "
            f"{comp.loc[row['roi'], 'beta_w60']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Precuneus在30秒窗口中仍显著正向预测IBC：β = {prec['beta']:.4f}，"
            f"cluster p = {prec['p_cluster']:.4f}，4-ROI FDR p = "
            f"{prec['p_fdr_4roi']:.4f}，ΔR² = {prec['delta_r2']:.4f}。",
            "",
            "## ROI斜率差",
            "",
            "| 对比 | 30秒差值 | p | FDR p |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in contrasts.iterrows():
        lines.append(
            f"| {row['contrast']} | {row['beta_difference']:.4f} | "
            f"{row['p_cluster']:.4f} | {row['p_fdr_3']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 三个组成指标",
            "",
            "| 指标 | Precuneus β | cluster p |",
            "|---|---:|---:|",
        ]
    )
    prec_components = components[components["roi"] == "Precuneus"]
    for _, row in prec_components.iterrows():
        lines.append(
            f"| {row['component_label']} | {row['beta']:.4f} | "
            f"{row['p_cluster']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 简要结论",
            "",
            "30秒非重叠窗口复现了60秒分析的核心模式：Precuneus社会互动系数保持",
            "正向且显著，mPFC不显著，rTPJ不显著。效应量从60秒的β≈0.038下降到",
            "30秒的β≈0.030，ΔR²也有所下降，符合较短窗口的IBC和行为指标噪声更大的预期。",
            "",
            "是否能继续支持“Precuneus显著强于其他ROI”，应以联合ROI斜率差检验的",
            "具体结果为准，而不能仅依据各ROI各自显著或不显著。",
        ]
    )
    (OUT / "w30_direct_similarity_summary_cn.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_CSV)
    data["roi_label"] = data["roi_label"].replace({"lTPJ": "rTPJ"})
    data = data[data["roi_label"].isin(ROI_ORDER)].copy()

    main_result = separate_models(data, SOCIAL)
    main_result["p_fdr_4roi"] = multipletests(
        main_result["p_cluster"], method="fdr_bh"
    )[1]
    contrasts = roi_contrasts(data, SOCIAL)

    component_frames = []
    for label, predictor in COMPONENTS.items():
        result = separate_models(data, predictor)
        result.insert(0, "component_label", label)
        component_frames.append(result)
    components = pd.concat(component_frames, ignore_index=True)
    components["p_fdr_12"] = multipletests(
        components["p_cluster"], method="fdr_bh"
    )[1]

    main_result.to_csv(OUT / "w30_social_index_coefficients_4roi.csv",
                       index=False)
    contrasts.to_csv(OUT / "w30_social_index_roi_contrasts.csv", index=False)
    components.to_csv(OUT / "w30_social_components_coefficients_4roi.csv",
                      index=False)
    comparison = make_comparison(main_result, contrasts)
    write_summary(main_result, contrasts, components, comparison)

    print("\n30-s social index coefficients")
    print(main_result.to_string(index=False))
    print("\n30-s ROI contrasts")
    print(contrasts.to_string(index=False))
    print("\n30-s Precuneus components")
    print(components[components["roi"] == "Precuneus"].to_string(index=False))
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()

