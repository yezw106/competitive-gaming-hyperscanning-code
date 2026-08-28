from pathlib import Path
import os

import cv2
import numpy as np
import pandas as pd


WORKDIR = Path(os.environ.get("ANALYSIS_ROOT", Path(__file__).resolve().parent))
OUT_DIR = Path(os.environ.get(
    "OUT_DIR",
    str(WORKDIR / "behavior_time_effect_analysis" / "sliding_window_brain_behavior"),
))

BRAIN_SUMMARY = Path(os.environ.get(
    "BRAIN_SUMMARY",
    str(WORKDIR / "voxelwise_ibc_roi_sliding_window_summary.csv"),
))
BEHAVIOR_DIR = Path(os.environ.get(
    "BEHAVIOR_DIR",
    str(WORKDIR / "inputs" / "behavior"),
))
POSITIONS_DIR = Path(os.environ.get(
    "POSITIONS_DIR",
    str(WORKDIR / "inputs" / "positions"),
))
VIDEO_DIR = Path(os.environ.get(
    "VIDEO_DIR",
    str(WORKDIR / "inputs" / "videos"),
))

VISUAL_SAMPLE_SEC = float(os.environ.get("VISUAL_SAMPLE_SEC", "1.0"))
FRAME_SIZE = (128, 72)
PROXIMITY_QUANTILE = 0.33

ROI_MAP = {
    "mpfc": "mPFC",
    "precuneus_p0005_intersect": "Precuneus",
    "rtpj_p0005_neurosynth_intersect": "rTPJ",
    "ltpj_p0005_neurosynth_intersect": "rTPJ",
    "ba17": "BA17",
}
ROI_ORDER = ["mPFC", "Precuneus", "rTPJ", "BA17"]


def zscore(series):
    x = pd.to_numeric(series, errors="coerce")
    sd = x.std(skipna=True, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return x * np.nan
    return (x - x.mean(skipna=True)) / sd


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 3:
        return np.nan
    x = x[:n]
    y = y[:n]
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    x = x[ok]
    y = y[ok]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)


def load_brain():
    df = pd.read_csv(BRAIN_SUMMARY)
    df = df[df["roi"].isin(ROI_MAP)].copy()
    df["roi_label"] = df["roi"].map(ROI_MAP)
    df["brain_ibc"] = pd.to_numeric(df["mean_fisher_z"], errors="coerce")
    df["pair_dash"] = df["subj1"] + "-" + df["subj2"]
    df["window_center_s"] = (df["start_s"] + df["end_s"]) / 2.0
    return df


def unique_pair_windows(brain):
    return brain[["pair", "pair_dash", "subj1", "subj2", "window_index", "start_s", "end_s", "window_center_s"]].drop_duplicates().reset_index(drop=True)


def load_subject_behavior(subjects):
    out = {}
    for subj in subjects:
        cols = ["time", "px", "py", "ex", "ey", "click", "Q", "W", "E", "R", "D", "F"]
        df = pd.read_csv(BEHAVIOR_DIR / f"{subj}_game2_behavior.csv", usecols=cols)
        skill_cols = ["Q", "W", "E", "R", "D", "F"]
        df["enemy_visible"] = (df["ex"] >= 0) & (df["ey"] >= 0)
        df["skill_event"] = df[skill_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        df["action_event"] = df["skill_event"] + pd.to_numeric(df["click"], errors="coerce").fillna(0)
        out[subj] = df
    return out


def load_subject_positions(subjects):
    out = {}
    for subj in subjects:
        cols = ["time_sec", "player_cx", "player_cy", "enemy_cx", "enemy_cy"]
        df = pd.read_csv(POSITIONS_DIR / f"{subj}_game2_positions.csv", usecols=cols)
        valid = (
            (df["player_cx"] >= 0)
            & (df["player_cy"] >= 0)
            & (df["enemy_cx"] >= 0)
            & (df["enemy_cy"] >= 0)
        )
        df["both_visible"] = valid
        df["distance"] = np.nan
        df.loc[valid, "distance"] = np.sqrt(
            (df.loc[valid, "player_cx"] - df.loc[valid, "enemy_cx"]) ** 2
            + (df.loc[valid, "player_cy"] - df.loc[valid, "enemy_cy"]) ** 2
        )
        out[subj] = df
    return out


def compute_proximity_threshold(positions, subjects, max_end_by_subject):
    values = []
    for subj in subjects:
        df = positions[subj]
        max_end = max_end_by_subject[subj]
        d = df[(df["time_sec"] >= 0) & (df["time_sec"] < max_end)]["distance"].dropna().to_numpy(dtype=float)
        if len(d):
            values.append(d)
    return float(np.quantile(np.concatenate(values), PROXIMITY_QUANTILE))


def row_durations(w):
    time = w["time"].to_numpy(dtype=float)
    if len(time) == 0:
        return np.array([])
    dt = np.empty(len(time), dtype=float)
    diffs = np.diff(time)
    good = diffs[(diffs > 0) & (diffs < 2.0)]
    typical = np.nanmedian(good) if len(good) else 1 / 30
    dt[:-1] = np.where((diffs > 0) & (diffs < 2.0), diffs, typical)
    dt[-1] = typical
    return dt


def behavior_window_metrics(df, start_s, end_s):
    w = df[(df["time"] >= start_s) & (df["time"] < end_s)].copy()
    dur_min = (end_s - start_s) / 60.0
    if len(w) == 0:
        return {
            "click_rate_min": np.nan,
            "skill_rate_min": np.nan,
            "opponent_contingent_skill": np.nan,
        }
    click_rate = pd.to_numeric(w["click"], errors="coerce").fillna(0).sum() / dur_min
    skill_rate = w["skill_event"].sum() / dur_min

    dt = row_durations(w)
    visible = w["enemy_visible"].to_numpy(dtype=bool)
    skill = w["skill_event"].to_numpy(dtype=float)
    dur_vis = dt[visible].sum()
    dur_not = dt[~visible].sum()
    rate_vis = skill[visible].sum() / (dur_vis / 60.0) if dur_vis > 0 else np.nan
    rate_not = skill[~visible].sum() / (dur_not / 60.0) if dur_not > 0 else np.nan
    contingent = rate_vis - rate_not if np.isfinite(rate_vis) and np.isfinite(rate_not) else np.nan
    return {
        "click_rate_min": float(click_rate),
        "skill_rate_min": float(skill_rate),
        "opponent_contingent_skill": float(contingent) if np.isfinite(contingent) else np.nan,
    }


def click_series(df, start_s, end_s, bin_sec=5.0):
    w = df[(df["time"] >= start_s) & (df["time"] < end_s)].copy()
    n_bins = int(np.ceil((end_s - start_s) / bin_sec))
    series = np.zeros(n_bins)
    if len(w) == 0:
        return series
    click = pd.to_numeric(w["click"], errors="coerce").fillna(0).to_numpy()
    bins = np.floor((w["time"].to_numpy() - start_s) / bin_sec).astype(int)
    ok = (bins >= 0) & (bins < n_bins) & (click > 0)
    if ok.any():
        counts = np.bincount(bins[ok], weights=click[ok], minlength=n_bins)
        series[: len(counts)] = counts[:n_bins]
    return series


def position_window_metrics(df, start_s, end_s, threshold):
    w = df[(df["time_sec"] >= start_s) & (df["time_sec"] < end_s)].copy()
    dur_min = (end_s - start_s) / 60.0
    if len(w) == 0:
        return {
            "co_presence_ratio": np.nan,
            "proximity_ratio": np.nan,
            "interaction_dynamics": np.nan,
            "state_transition_rate": np.nan,
            "map_speed_mean": np.nan,
            "map_path_per_min": np.nan,
        }
    co_presence = float(w["both_visible"].mean())
    d = w["distance"].dropna().to_numpy(dtype=float)
    proximity_ratio = float(np.mean(d <= threshold)) if len(d) else np.nan

    valid = w.dropna(subset=["distance"]).copy()
    interaction_dyn = np.nan
    if len(valid) > 2:
        dist = valid["distance"].to_numpy(dtype=float)
        time = valid["time_sec"].to_numpy(dtype=float)
        dt = np.diff(time)
        dd = np.diff(dist)
        ok = (dt > 0) & (dt < 2.0) & np.isfinite(dd)
        if ok.any():
            interaction_dyn = float(co_presence * np.mean(np.abs(dd[ok]) / dt[ok]))

    state = ((w["both_visible"]) & (w["distance"] <= threshold)).to_numpy(dtype=bool)
    time = w["time_sec"].to_numpy(dtype=float)
    starts = 0
    for i, on in enumerate(state):
        if on and (i == 0 or (not state[i - 1]) or (time[i] - time[i - 1] > 2.0)):
            starts += 1
    transition_rate = float(starts / dur_min)

    player_valid = (w["player_cx"] >= 0) & (w["player_cy"] >= 0)
    p = w.loc[player_valid, ["time_sec", "player_cx", "player_cy"]].copy()
    speed_mean = np.nan
    path_per_min = np.nan
    if len(p) > 2:
        dt = p["time_sec"].diff().to_numpy(dtype=float)
        dx = p["player_cx"].diff().to_numpy(dtype=float)
        dy = p["player_cy"].diff().to_numpy(dtype=float)
        step = np.sqrt(dx ** 2 + dy ** 2)
        ok = (dt > 0) & (dt < 2.0) & np.isfinite(step)
        if ok.any():
            speed_mean = float(np.nanmean(step[ok] / dt[ok]))
            path_per_min = float(np.nansum(step[ok]) / dur_min)

    return {
        "co_presence_ratio": co_presence,
        "proximity_ratio": proximity_ratio,
        "interaction_dynamics": interaction_dyn,
        "state_transition_rate": transition_rate,
        "map_speed_mean": speed_mean,
        "map_path_per_min": path_per_min,
    }


def speed_series(df, start_s, end_s, bin_sec=1.0):
    w = df[(df["time_sec"] >= start_s) & (df["time_sec"] < end_s)].copy()
    n_bins = int(np.ceil((end_s - start_s) / bin_sec))
    series = np.full(n_bins, np.nan)
    player_valid = (w["player_cx"] >= 0) & (w["player_cy"] >= 0)
    p = w.loc[player_valid, ["time_sec", "player_cx", "player_cy"]].copy()
    if len(p) < 3:
        return series
    t = p["time_sec"].to_numpy(dtype=float)
    dt = p["time_sec"].diff().to_numpy(dtype=float)
    dx = p["player_cx"].diff().to_numpy(dtype=float)
    dy = p["player_cy"].diff().to_numpy(dtype=float)
    step = np.sqrt(dx ** 2 + dy ** 2)
    speed = step / dt
    ok = (dt > 0) & (dt < 2.0) & np.isfinite(speed)
    bins = np.floor((t - start_s) / bin_sec).astype(int)
    tmp = pd.DataFrame({"bin": bins[ok], "speed": speed[ok]})
    tmp = tmp[(tmp["bin"] >= 0) & (tmp["bin"] < n_bins)]
    if len(tmp):
        binned = tmp.groupby("bin")["speed"].mean()
        series[binned.index.to_numpy(dtype=int)] = binned.to_numpy(dtype=float)
    return series


def pair_mean(m1, m2):
    keys = sorted(set(m1) | set(m2))
    out = {}
    for k in keys:
        vals = [m1.get(k, np.nan), m2.get(k, np.nan)]
        vals = [v for v in vals if np.isfinite(v)]
        out[k] = float(np.mean(vals)) if vals else np.nan
    return out


def build_behavior_controls(pair_windows, behavior, positions, threshold):
    rows = []
    for _, row in pair_windows.iterrows():
        s1, s2 = row["subj1"], row["subj2"]
        start_s, end_s = row["start_s"], row["end_s"]
        b = pair_mean(
            behavior_window_metrics(behavior[s1], start_s, end_s),
            behavior_window_metrics(behavior[s2], start_s, end_s),
        )
        p = pair_mean(
            position_window_metrics(positions[s1], start_s, end_s, threshold),
            position_window_metrics(positions[s2], start_s, end_s, threshold),
        )
        click_coupling = safe_corr(
            click_series(behavior[s1], start_s, end_s),
            click_series(behavior[s2], start_s, end_s),
        )
        speed_coupling = safe_corr(
            speed_series(positions[s1], start_s, end_s),
            speed_series(positions[s2], start_s, end_s),
        )
        out = row.to_dict()
        out.update(b)
        out.update(p)
        out["click_coupling_z"] = np.arctanh(np.clip(click_coupling, -0.999999, 0.999999)) if np.isfinite(click_coupling) else np.nan
        out["movement_speed_coupling_z"] = np.arctanh(np.clip(speed_coupling, -0.999999, 0.999999)) if np.isfinite(speed_coupling) else np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def extract_frame_features(frame, prev_gray=None):
    frame = cv2.resize(frame, FRAME_SIZE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 80, 160)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).flatten()
    if hist.sum() > 0:
        hist = hist / hist.sum()
    motion = np.nan if prev_gray is None else float(np.mean(np.abs(gray.astype(float) - prev_gray.astype(float))))
    return {
        "luminance_mean": float(gray.mean()),
        "luminance_std": float(gray.std(ddof=1)),
        "saturation_mean": float(hsv[:, :, 1].mean()),
        "edge_density": float((edges > 0).mean()),
        "visual_motion": motion,
        "hist": hist,
        "gray": gray,
    }


def build_subject_visual_cache(subjects, max_end_by_subject, cache_csv, hist_npz):
    if cache_csv.exists() and hist_npz.exists():
        features = pd.read_csv(cache_csv)
        hist_data = np.load(hist_npz, allow_pickle=True)
        hists = {k: hist_data[k] for k in hist_data.files}
        return features, hists

    rows = []
    hists = {}
    for si, subj in enumerate(subjects, start=1):
        video = VIDEO_DIR / f"{subj}_game2.mp4"
        if not video.exists():
            raise FileNotFoundError(video)
        print(f"[visual subject] {si}/{len(subjects)} {subj}", flush=True)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video}")
        prev_gray = None
        subj_hists = []
        times = np.arange(0.5, max_end_by_subject[subj], VISUAL_SAMPLE_SEC)
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t * 1000.0))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            feat = extract_frame_features(frame, prev_gray)
            prev_gray = feat["gray"]
            hist_idx = len(subj_hists)
            subj_hists.append(feat["hist"])
            rows.append(
                {
                    "subject": subj,
                    "time": t,
                    "hist_idx": hist_idx,
                    "luminance_mean": feat["luminance_mean"],
                    "luminance_std": feat["luminance_std"],
                    "saturation_mean": feat["saturation_mean"],
                    "edge_density": feat["edge_density"],
                    "visual_motion": feat["visual_motion"],
                }
            )
        cap.release()
        hists[subj] = np.asarray(subj_hists)
    features = pd.DataFrame(rows)
    features.to_csv(cache_csv, index=False, encoding="utf-8-sig")
    np.savez_compressed(hist_npz, **hists)
    return features, hists


def visual_window_subject(features, hists, subj, start_s, end_s):
    w = features[(features["subject"] == subj) & (features["time"] >= start_s) & (features["time"] < end_s)].copy()
    if len(w) == 0:
        return None
    return {
        "n_visual_samples": len(w),
        "luminance_mean": float(w["luminance_mean"].mean()),
        "luminance_std": float(w["luminance_std"].mean()),
        "saturation_mean": float(w["saturation_mean"].mean()),
        "edge_density": float(w["edge_density"].mean()),
        "visual_motion_mean": float(w["visual_motion"].mean(skipna=True)),
        "lum_series": w["luminance_mean"].to_numpy(dtype=float),
        "motion_series": w["visual_motion"].to_numpy(dtype=float),
        "hist_series": hists[subj][w["hist_idx"].to_numpy(dtype=int)],
    }


def build_visual_controls(pair_windows, features, hists):
    rows = []
    for _, row in pair_windows.iterrows():
        f1 = visual_window_subject(features, hists, row["subj1"], row["start_s"], row["end_s"])
        f2 = visual_window_subject(features, hists, row["subj2"], row["start_s"], row["end_s"])
        if f1 is None or f2 is None:
            continue
        n_hist = min(len(f1["hist_series"]), len(f2["hist_series"]))
        hist_sim = np.nanmean([cosine_similarity(f1["hist_series"][i], f2["hist_series"][i]) for i in range(n_hist)]) if n_hist else np.nan
        rows.append(
            {
                "pair": row["pair"],
                "pair_dash": row["pair_dash"],
                "window_index": row["window_index"],
                "n_visual_samples_min": min(f1["n_visual_samples"], f2["n_visual_samples"]),
                "luminance_mean_pair": np.nanmean([f1["luminance_mean"], f2["luminance_mean"]]),
                "luminance_std_pair": np.nanmean([f1["luminance_std"], f2["luminance_std"]]),
                "saturation_mean_pair": np.nanmean([f1["saturation_mean"], f2["saturation_mean"]]),
                "edge_density_pair": np.nanmean([f1["edge_density"], f2["edge_density"]]),
                "visual_motion_mean_pair": np.nanmean([f1["visual_motion_mean"], f2["visual_motion_mean"]]),
                "luminance_similarity": safe_corr(f1["lum_series"], f2["lum_series"]),
                "visual_motion_similarity": safe_corr(f1["motion_series"], f2["motion_series"]),
                "color_hist_similarity": hist_sim,
            }
        )
    return pd.DataFrame(rows)


def add_composites(df):
    out = df.copy()
    cols = [
        "co_presence_ratio",
        "interaction_dynamics",
        "state_transition_rate",
        "click_coupling_z",
        "movement_speed_coupling_z",
        "luminance_similarity",
        "color_hist_similarity",
        "visual_motion_similarity",
    ]
    for c in cols:
        out[f"z_{c}"] = zscore(out[c])
    out["social_interaction_index"] = out[["z_co_presence_ratio", "z_interaction_dynamics", "z_state_transition_rate"]].mean(axis=1, skipna=True)
    out["time_z"] = zscore(out["window_center_s"])
    out["time_z2"] = out["time_z"] ** 2
    return out


def build_dataset():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    brain = load_brain()
    pair_windows = unique_pair_windows(brain)
    subjects = sorted(set(pair_windows["subj1"]) | set(pair_windows["subj2"]))
    max_end = {}
    for subj in subjects:
        max_end[subj] = float(pair_windows[(pair_windows["subj1"] == subj) | (pair_windows["subj2"] == subj)]["end_s"].max())

    behavior = load_subject_behavior(subjects)
    positions = load_subject_positions(subjects)
    threshold = compute_proximity_threshold(positions, subjects, max_end)
    print(f"Proximity threshold: {threshold:.6f}", flush=True)

    beh = build_behavior_controls(pair_windows, behavior, positions, threshold)
    sample_label = f"{int(VISUAL_SAMPLE_SEC)}s" if float(VISUAL_SAMPLE_SEC).is_integer() else f"{VISUAL_SAMPLE_SEC:g}s"
    feat_cache = OUT_DIR / f"subject_visual_features_{sample_label}.csv"
    hist_cache = OUT_DIR / f"subject_visual_histograms_{sample_label}.npz"
    features, hists = build_subject_visual_cache(subjects, max_end, feat_cache, hist_cache)
    visual = build_visual_controls(pair_windows, features, hists)

    controls = beh.merge(visual, on=["pair", "pair_dash", "window_index"], how="left")
    controls = add_composites(controls)
    data = brain.merge(
        controls.drop(columns=["subj1", "subj2", "start_s", "end_s", "window_center_s"], errors="ignore"),
        on=["pair", "pair_dash", "window_index"],
        how="left",
    )
    data = add_composites(data)
    return data, controls


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data, controls = build_dataset()

    data_csv = OUT_DIR / "sliding_window_brain_behavior_dataset.csv"
    controls_csv = OUT_DIR / "sliding_window_pair_behavior_visual_controls.csv"
    data.to_csv(data_csv, index=False, encoding="utf-8-sig")
    controls.to_csv(controls_csv, index=False, encoding="utf-8-sig")

    print("Saved:")
    for p in [data_csv, controls_csv]:
        print(p)


if __name__ == "__main__":
    main()

