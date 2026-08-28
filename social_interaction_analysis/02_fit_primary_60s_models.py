import warnings
import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
warnings.filterwarnings('ignore')

ROOT = Path(os.environ.get('ANALYSIS_ROOT', Path(__file__).resolve().parent))
DATA3 = ROOT / 'behavior_time_effect_analysis' / 'sliding_window_brain_behavior_w60_s60_3roi_visual1s' / 'sliding_window_brain_behavior_dataset.csv'
BA17 = ROOT / 'voxelwise_ibc_ba17_sliding_window_summary.csv'
OUT = ROOT / 'behavior_time_effect_analysis' / 'sliding_window_direct_similarity_model_w60_s60_4roi_visual1s'
OUT.mkdir(parents=True, exist_ok=True)
ROI_ORDER = ['mPFC', 'Precuneus', 'rTPJ', 'BA17']
ROI_COLORS = {'mPFC': '#4C78A8', 'Precuneus': '#F58518', 'rTPJ': '#54A24B', 'BA17': '#B279A2'}
control_vars = ['z_luminance_similarity','z_color_hist_similarity','z_visual_motion_similarity','z_click_coupling_z','z_movement_speed_coupling_z','time_z','time_z2']
base_formula = 'brain_ibc ~ ' + ' + '.join(control_vars + ['C(pair)'])
full_formula = 'brain_ibc ~ ' + ' + '.join(control_vars + ['social_interaction_index','C(pair)'])
component_terms = {'co_presence': 'z_co_presence_ratio', 'interaction_dynamics': 'z_interaction_dynamics', 'state_transition_rate': 'z_state_transition_rate'}
component_labels = {'co_presence': 'Co-presence', 'interaction_dynamics': 'Interaction dynamics', 'state_transition_rate': 'State-transition rate'}
model_vars_base = control_vars + ['brain_ibc', 'pair']
model_vars_full = model_vars_base + ['social_interaction_index']

def fdr_bh(pvals):
    arr = np.asarray(pvals, dtype=float)
    out = np.full(arr.shape, np.nan)
    mask = np.isfinite(arr)
    if mask.sum():
        out[mask] = multipletests(arr[mask], method='fdr_bh')[1]
    return out

def fit_ols_cluster(formula, data):
    m = smf.ols(formula, data=data).fit()
    r = m.get_robustcov_results(cov_type='cluster', groups=data['pair'], use_correction=True, df_correction=True)
    names = m.model.exog_names
    return m, pd.Series(r.params, index=names), pd.Series(r.bse, index=names), pd.Series(r.tvalues, index=names), pd.Series(r.pvalues, index=names)

def ci95(beta, se):
    return beta - 1.96 * se, beta + 1.96 * se

def p_text(p):
    if pd.isna(p):
        return 'p = NA'
    if p < 0.001:
        return 'p < .001'
    return ('p = {:.3f}'.format(p)).replace('0.', '.')

def fdr_text(p):
    if pd.isna(p):
        return 'NA'
    if p < 0.001:
        return '<.001'
    return ('={:.3f}'.format(p)).replace('0.', '.')

def fmtp(p):
    return p_text(p)

print('Reading and merging data')
df3 = pd.read_csv(DATA3)
ba17 = pd.read_csv(BA17)
keys = ['pair', 'window_index', 'start_s', 'end_s']
template = df3[df3['roi_label'].astype(str).str.lower().eq('precuneus')].copy()
merged = template.merge(ba17[keys + ['roi','subj1','subj2','n_tr_subj1','n_tr_subj2','n_common_tr','start_tr_1based','end_tr_1based','n_segment_tr','mean_r','mean_fisher_z','r_map','z_map']], on=keys, suffixes=('', '_ba17'), how='inner')
if len(merged) != len(ba17):
    raise ValueError('BA17 merge failed: {} vs {}'.format(len(merged), len(ba17)))
ba_rows = merged[template.columns].copy()
for col in ['roi','subj1','subj2','n_tr_subj1','n_tr_subj2','n_common_tr','start_tr_1based','end_tr_1based','n_segment_tr','mean_r','mean_fisher_z','r_map','z_map']:
    src = col + '_ba17'
    if src in merged.columns:
        ba_rows[col] = merged[src].values
ba_rows['roi'] = 'ba17'
ba_rows['roi_label'] = 'BA17'
ba_rows['brain_ibc'] = ba_rows['mean_fisher_z']
df4 = pd.concat([df3, ba_rows], ignore_index=True)
df4['roi_label'] = df4['roi_label'].astype(str)
df4 = df4.sort_values(['roi_label', 'pair', 'window_index']).reset_index(drop=True)
df4.to_csv(OUT / 'direct_similarity_brain_behavior_dataset_4roi.csv', index=False, encoding='utf-8-sig')

qc_rows = []
for roi in ROI_ORDER:
    g = df4[df4['roi_label'].eq(roi)]
    qc_rows.append({'roi': roi, 'rows_total': len(g), 'pairs_total': g['pair'].nunique(), 'windows_total': g['window_index'].nunique(), 'rows_complete_main_model': g[model_vars_full].dropna().shape[0], 'pairs_complete_main_model': g[model_vars_full].dropna()['pair'].nunique(), 'min_visual_samples': g['n_visual_samples_min'].min(), 'max_visual_samples': g['n_visual_samples_min'].max()})
qc = pd.DataFrame(qc_rows)
qc.to_csv(OUT / 'direct_similarity_data_qc.csv', index=False, encoding='utf-8-sig')

main_rows = []
partial_rows = []
for roi in ROI_ORDER:
    g = df4[df4['roi_label'].eq(roi)][model_vars_full].dropna().copy()
    base_m, _, _, _, _ = fit_ols_cluster(base_formula, g)
    full_m, params, bse, tvals, pvals = fit_ols_cluster(full_formula, g)
    term = 'social_interaction_index'
    beta = params[term]
    se = bse[term]
    lo, hi = ci95(beta, se)
    main_rows.append({'roi': roi, 'n_rows': len(g), 'n_pairs': g['pair'].nunique(), 'predictor': term, 'beta_cluster': beta, 'se_cluster': se, 'ci95_low_cluster': lo, 'ci95_high_cluster': hi, 't_cluster': tvals[term], 'p_cluster': pvals[term], 'p_ordinary_ols': full_m.pvalues.get(term, np.nan), 'r2_base': base_m.rsquared, 'r2_full': full_m.rsquared, 'delta_r2': full_m.rsquared - base_m.rsquared, 'aic_base': base_m.aic, 'aic_full': full_m.aic})
    y_res = smf.ols(base_formula, data=g).fit().resid
    x_res = smf.ols('social_interaction_index ~ ' + ' + '.join(control_vars + ['C(pair)']), data=g).fit().resid
    partial_rows.append(pd.DataFrame({'roi': roi, 'x_resid': x_res, 'y_resid': y_res, 'pair': g['pair'].values}))
main_df = pd.DataFrame(main_rows)
main_df['p_fdr_4roi'] = fdr_bh(main_df['p_cluster'])
main_df.to_csv(OUT / 'direct_similarity_social_index_coefficients_4roi.csv', index=False, encoding='utf-8-sig')
partial_df = pd.concat(partial_rows, ignore_index=True)
partial_df.to_csv(OUT / 'direct_similarity_social_index_partial_residuals_4roi.csv', index=False, encoding='utf-8-sig')
print(main_df[['roi','n_rows','n_pairs','beta_cluster','p_cluster','p_fdr_4roi','delta_r2']].to_string(index=False))

# ROI contrasts with explicit dummies and ROI-specific covariate slopes
st = df4[['brain_ibc','roi_label','pair'] + control_vars + ['social_interaction_index']].dropna().copy()
for roi in ['mPFC','rTPJ','BA17']:
    st['is_' + roi] = (st['roi_label'].eq(roi)).astype(int)
    st[roi + '_x_social'] = st['is_' + roi] * st['social_interaction_index']
    for v in control_vars:
        st[roi + '_x_' + v] = st['is_' + roi] * st[v]
rhs = ['social_interaction_index'] + control_vars + ['is_mPFC','is_rTPJ','is_BA17']
rhs += [r + '_x_social' for r in ['mPFC','rTPJ','BA17']]
rhs += [r + '_x_' + v for r in ['mPFC','rTPJ','BA17'] for v in control_vars]
rhs += ['C(pair)', 'C(pair):is_mPFC', 'C(pair):is_rTPJ', 'C(pair):is_BA17']
contrast_formula = 'brain_ibc ~ ' + ' + '.join(rhs)
cm, cparams, cbse, ct, cp = fit_ols_cluster(contrast_formula, st)
contrast_rows = []
for roi in ['mPFC','rTPJ','BA17']:
    term = roi + '_x_social'
    beta = cparams.get(term, np.nan)
    se = cbse.get(term, np.nan)
    contrast_rows.append({'contrast': roi + ' minus Precuneus social slope', 'roi_minus_precuneus': roi, 'beta_diff': beta, 'se_cluster': se, 'ci95_low_cluster': beta - 1.96 * se, 'ci95_high_cluster': beta + 1.96 * se, 't_cluster': ct.get(term, np.nan), 'p_cluster': cp.get(term, np.nan), 'n_rows': len(st), 'n_pairs': st['pair'].nunique()})
contrast_df = pd.DataFrame(contrast_rows)
contrast_df['p_fdr_3contrasts'] = fdr_bh(contrast_df['p_cluster'])
contrast_df.to_csv(OUT / 'direct_similarity_social_index_roi_contrasts_vs_precuneus.csv', index=False, encoding='utf-8-sig')
print(contrast_df[['contrast','beta_diff','p_cluster','p_fdr_3contrasts']].to_string(index=False))

comp_rows = []
for comp_name, term in component_terms.items():
    full_comp_formula = 'brain_ibc ~ ' + ' + '.join(control_vars + [term, 'C(pair)'])
    for roi in ROI_ORDER:
        g = df4[df4['roi_label'].eq(roi)][model_vars_base + [term]].dropna().copy()
        base_m, _, _, _, _ = fit_ols_cluster(base_formula, g)
        full_m, params, bse, tvals, pvals = fit_ols_cluster(full_comp_formula, g)
        beta = params[term]
        se = bse[term]
        lo, hi = ci95(beta, se)
        comp_rows.append({'component': comp_name, 'component_label': component_labels[comp_name], 'predictor': term, 'roi': roi, 'n_rows': len(g), 'n_pairs': g['pair'].nunique(), 'beta_cluster': beta, 'se_cluster': se, 'ci95_low_cluster': lo, 'ci95_high_cluster': hi, 't_cluster': tvals[term], 'p_cluster': pvals[term], 'p_ordinary_ols': full_m.pvalues.get(term, np.nan), 'r2_base': base_m.rsquared, 'r2_full': full_m.rsquared, 'delta_r2': full_m.rsquared - base_m.rsquared})
comp_df = pd.DataFrame(comp_rows)
comp_df['p_fdr_12tests'] = fdr_bh(comp_df['p_cluster'])
comp_df['p_fdr_within_component_4roi'] = np.nan
for comp_name in comp_df['component'].unique():
    idx = comp_df['component'].eq(comp_name)
    comp_df.loc[idx, 'p_fdr_within_component_4roi'] = fdr_bh(comp_df.loc[idx, 'p_cluster'])
comp_df['p_fdr_within_roi_3components'] = np.nan
for roi in ROI_ORDER:
    idx = comp_df['roi'].eq(roi)
    comp_df.loc[idx, 'p_fdr_within_roi_3components'] = fdr_bh(comp_df.loc[idx, 'p_cluster'])
comp_df.to_csv(OUT / 'direct_similarity_social_components_coefficients_4roi.csv', index=False, encoding='utf-8-sig')
print(comp_df[['component','roi','n_rows','beta_cluster','p_cluster','p_fdr_12tests','delta_r2']].to_string(index=False))

comp_contrast_rows = []
for comp_name, term in component_terms.items():
    cs = df4[['brain_ibc','roi_label','pair'] + control_vars + [term]].dropna().copy()
    for roi in ['mPFC','rTPJ','BA17']:
        cs['is_' + roi] = (cs['roi_label'].eq(roi)).astype(int)
        cs[roi + '_x_term'] = cs['is_' + roi] * cs[term]
        for v in control_vars:
            cs[roi + '_x_' + v] = cs['is_' + roi] * cs[v]
    rhs = [term] + control_vars + ['is_mPFC','is_rTPJ','is_BA17']
    rhs += [r + '_x_term' for r in ['mPFC','rTPJ','BA17']]
    rhs += [r + '_x_' + v for r in ['mPFC','rTPJ','BA17'] for v in control_vars]
    rhs += ['C(pair)', 'C(pair):is_mPFC', 'C(pair):is_rTPJ', 'C(pair):is_BA17']
    formula = 'brain_ibc ~ ' + ' + '.join(rhs)
    m, params, bse, tv, pv = fit_ols_cluster(formula, cs)
    for roi in ['mPFC','rTPJ','BA17']:
        iname = roi + '_x_term'
        beta = params.get(iname, np.nan)
        se = bse.get(iname, np.nan)
        comp_contrast_rows.append({'component': comp_name, 'component_label': component_labels[comp_name], 'roi_minus_precuneus': roi, 'contrast': roi + ' minus Precuneus ' + component_labels[comp_name] + ' slope', 'beta_diff': beta, 'se_cluster': se, 'ci95_low_cluster': beta - 1.96 * se, 'ci95_high_cluster': beta + 1.96 * se, 't_cluster': tv.get(iname, np.nan), 'p_cluster': pv.get(iname, np.nan), 'n_rows': len(cs), 'n_pairs': cs['pair'].nunique()})
comp_contrast_df = pd.DataFrame(comp_contrast_rows)
comp_contrast_df['p_fdr_9contrasts'] = fdr_bh(comp_contrast_df['p_cluster'])
idx_ba17 = comp_contrast_df['roi_minus_precuneus'].eq('BA17')
comp_contrast_df['p_fdr_ba17_vs_precuneus_3components'] = np.nan
comp_contrast_df.loc[idx_ba17, 'p_fdr_ba17_vs_precuneus_3components'] = fdr_bh(comp_contrast_df.loc[idx_ba17, 'p_cluster'])
comp_contrast_df.to_csv(OUT / 'direct_similarity_social_components_roi_contrasts_vs_precuneus.csv', index=False, encoding='utf-8-sig')

corr_vars = ['z_luminance_similarity','z_color_hist_similarity','z_visual_motion_similarity','z_click_coupling_z','z_movement_speed_coupling_z','social_interaction_index'] + list(component_terms.values())
df4[df4['roi_label'].eq('Precuneus')][corr_vars].corr().to_csv(OUT / 'direct_similarity_predictor_correlations.csv', encoding='utf-8-sig')

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), dpi=180)
ax = axes[0,0]
for roi in ROI_ORDER:
    tmp = df4[df4['roi_label'].eq(roi)].groupby('window_index')['brain_ibc'].agg(['mean','sem']).reset_index()
    color = ROI_COLORS[roi]
    xvals = tmp['window_index'].to_numpy(float)
    yvals = tmp['mean'].to_numpy(float)
    sevals = tmp['sem'].to_numpy(float)
    ax.plot(xvals, yvals, marker='o', ms=3, lw=1.6, color=color, label=roi)
    ax.fill_between(xvals, yvals-sevals, yvals+sevals, color=color, alpha=0.16, linewidth=0)
ax.axhline(0, color='0.75', lw=0.8)
ax.set_xlabel('Window index (60 s non-overlap)')
ax.set_ylabel('IBC (Fisher z)')
ax.set_title('A. ROI IBC dynamics')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend(frameon=False, ncol=2, fontsize=8)

ax = axes[0,1]
md = main_df.set_index('roi').loc[ROI_ORDER].reset_index()
x = np.arange(len(md))
colors = [ROI_COLORS[r] for r in md['roi']]
ax.bar(x, md['beta_cluster'], color=colors, alpha=0.85)
ax.errorbar(x, md['beta_cluster'], yerr=1.96*md['se_cluster'], fmt='none', ecolor='black', capsize=4, lw=1)
ax.axhline(0, color='0.45', lw=0.9)
ax.set_xticks(x, md['roi'], rotation=20, ha='right')
ax.set_ylabel('Social index beta')
ax.set_title('B. Added social interaction index')
for i, row in md.iterrows():
    yy = row['ci95_high_cluster'] if row['beta_cluster'] >= 0 else row['ci95_low_cluster']
    va = 'bottom' if row['beta_cluster'] >= 0 else 'top'
    pad = 0.004 if row['beta_cluster'] >= 0 else -0.004
    ax.text(i, yy + pad, p_text(row['p_cluster']) + '\nFDR ' + fdr_text(row['p_fdr_4roi']), ha='center', va=va, fontsize=7)

ax = axes[1,0]
ax.bar(x, md['delta_r2'], color=colors, alpha=0.85)
ax.set_xticks(x, md['roi'], rotation=20, ha='right')
ax.set_ylabel('Delta R-squared')
ax.set_title('C. Incremental explained variance')
for i, row in md.iterrows():
    ax.text(i, row['delta_r2'] + max(md['delta_r2'].max()*0.03, 0.0005), ('{:.3f}'.format(row['delta_r2'])).replace('0.','.'), ha='center', va='bottom', fontsize=8)

ax = axes[1,1]
cd = contrast_df.set_index('roi_minus_precuneus').loc[['mPFC','rTPJ','BA17']].reset_index()
x2 = np.arange(len(cd))
cols2 = [ROI_COLORS[r] for r in cd['roi_minus_precuneus']]
ax.bar(x2, cd['beta_diff'], color=cols2, alpha=0.85)
ax.errorbar(x2, cd['beta_diff'], yerr=1.96*cd['se_cluster'], fmt='none', ecolor='black', capsize=4, lw=1)
ax.axhline(0, color='0.45', lw=0.9)
ax.set_xticks(x2, [r + ' - Prec.' for r in cd['roi_minus_precuneus']], rotation=20, ha='right')
ax.set_ylabel('Slope difference')
ax.set_title('D. ROI differences in social slope')
for i, row in cd.iterrows():
    yy = row['ci95_high_cluster'] if row['beta_diff'] >= 0 else row['ci95_low_cluster']
    va = 'bottom' if row['beta_diff'] >= 0 else 'top'
    pad = 0.004 if row['beta_diff'] >= 0 else -0.004
    ax.text(i, yy + pad, p_text(row['p_cluster']), ha='center', va=va, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'direct_similarity_social_index_4roi.png', bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(12, 8.0), dpi=180, sharey=True)
for ax, roi in zip(axes.ravel(), ROI_ORDER):
    sub = comp_df[comp_df['roi'].eq(roi)].set_index('component').loc[list(component_terms.keys())].reset_index()
    xx = np.arange(len(sub))
    ax.bar(xx, sub['beta_cluster'], color=ROI_COLORS[roi], alpha=0.85)
    ax.errorbar(xx, sub['beta_cluster'], yerr=1.96*sub['se_cluster'], fmt='none', ecolor='black', capsize=4, lw=1)
    ax.axhline(0, color='0.45', lw=0.9)
    ax.set_xticks(xx, sub['component_label'], rotation=18, ha='right')
    ax.set_title(roi)
    ax.set_ylabel('Component beta')
    for i, row in sub.iterrows():
        yy = row['ci95_high_cluster'] if row['beta_cluster'] >= 0 else row['ci95_low_cluster']
        va = 'bottom' if row['beta_cluster'] >= 0 else 'top'
        pad = 0.004 if row['beta_cluster'] >= 0 else -0.004
        ax.text(i, yy + pad, p_text(row['p_cluster']), ha='center', va=va, fontsize=7)
fig.suptitle('Social interaction components beyond direct visual/behavioral similarity controls', y=1.02, fontsize=12)
fig.tight_layout()
fig.savefig(OUT / 'direct_similarity_social_components_4roi.png', bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
sub = comp_contrast_df[comp_contrast_df['roi_minus_precuneus'].eq('BA17')].set_index('component').loc[list(component_terms.keys())].reset_index()
xx = np.arange(len(sub))
ax.bar(xx, sub['beta_diff'], color=ROI_COLORS['BA17'], alpha=0.85)
ax.errorbar(xx, sub['beta_diff'], yerr=1.96*sub['se_cluster'], fmt='none', ecolor='black', capsize=4, lw=1)
ax.axhline(0, color='0.45', lw=0.9)
ax.set_xticks(xx, sub['component_label'], rotation=15, ha='right')
ax.set_ylabel('Slope difference: BA17 - Precuneus')
ax.set_title('BA17 control: component slope differences versus Precuneus')
for i, row in sub.iterrows():
    yy = row['ci95_high_cluster'] if row['beta_diff'] >= 0 else row['ci95_low_cluster']
    va = 'bottom' if row['beta_diff'] >= 0 else 'top'
    pad = 0.004 if row['beta_diff'] >= 0 else -0.004
    ax.text(i, yy + pad, p_text(row['p_cluster']) + '\nFDR ' + fdr_text(row['p_fdr_ba17_vs_precuneus_3components']), ha='center', va=va, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'direct_similarity_ba17_vs_precuneus_component_contrasts.png', bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), dpi=180)
for ax, roi in zip(axes.ravel(), ROI_ORDER):
    sub = partial_df[partial_df['roi'].eq(roi)].dropna()
    ax.scatter(sub['x_resid'], sub['y_resid'], s=18, alpha=0.55, color=ROI_COLORS[roi], edgecolor='none')
    if len(sub) > 2:
        slope, intercept, _, _, _ = stats.linregress(sub['x_resid'], sub['y_resid'])
        xs = np.linspace(sub['x_resid'].min(), sub['x_resid'].max(), 100)
        ax.plot(xs, intercept + slope*xs, color='black', lw=1.2)
    row = main_df[main_df['roi'].eq(roi)].iloc[0]
    ax.axhline(0, color='0.82', lw=0.8)
    ax.axvline(0, color='0.82', lw=0.8)
    ax.set_title('{}: beta={:.3f}, {}'.format(roi, row['beta_cluster'], p_text(row['p_cluster'])))
    ax.set_xlabel('Social index residual')
    ax.set_ylabel('IBC residual')
fig.tight_layout()
fig.savefig(OUT / 'direct_similarity_social_index_partial_residuals_4roi.png', bbox_inches='tight')
plt.close(fig)

# Chinese markdown summary
lines = []
lines.append('# 1秒视觉采样下的直接相似性控制模型分析说明')
lines.append('')
lines.append('## 1. 分析目标')
lines.append('本次分析检验：在控制低阶视觉相似性、低阶行为耦合、时间趋势以及被试对固定差异之后，较高阶的社会互动指标是否仍然能够解释 sliding-window 脑际同步（IBC）的变化。')
lines.append('')
lines.append('与前一版使用复合控制指标不同，这一版只纳入更直接、含义更清楚的相似性/耦合变量：亮度相似性、颜色直方图相似性、视觉运动相似性、点击耦合、移动速度耦合。')
lines.append('')
lines.append('## 2. 数据来源')
lines.append('- 3个功能 ROI 的 IBC 数据：`' + str(DATA3) + '`')
lines.append('- BA17 视觉皮层对照 ROI 的 IBC 数据：`' + str(BA17) + '`')
lines.append('- 行为和视觉控制变量来自 60 秒非重叠 sliding window；视觉特征按 1 秒采样，因此每个 60 秒窗口通常包含 60 个视觉采样点。')
lines.append('- 合并后的 4 ROI 数据保存为：`' + str(OUT / 'direct_similarity_brain_behavior_dataset_4roi.csv') + '`')
lines.append('')
lines.append('## 3. 变量定义')
lines.append('- 因变量：`brain_ibc`，即每个 ROI、每个被试对、每个 60 秒窗口的 voxel-wise IBC 后 ROI 平均 Fisher-z。')
lines.append('- 低阶视觉相似性：`z_luminance_similarity`、`z_color_hist_similarity`、`z_visual_motion_similarity`。')
lines.append('- 低阶行为耦合：`z_click_coupling_z`、`z_movement_speed_coupling_z`。')
lines.append('- 高阶社会互动总体指标：`social_interaction_index`。')
lines.append('- 三个社会互动组成指标：`z_co_presence_ratio`、`z_interaction_dynamics`、`z_state_transition_rate`。')
lines.append('- 时间控制：`time_z` 和 `time_z2`，分别表示线性时间趋势和二次时间趋势。')
lines.append('- 被试对固定效应：`C(pair)`，用于控制不同 dyad 的稳定差异。')
lines.append('')
lines.append('## 4. 统计模型')
lines.append('基础模型：')
lines.append('```text')
lines.append('brain_ibc ~ z_luminance_similarity + z_color_hist_similarity + z_visual_motion_similarity')
lines.append('          + z_click_coupling_z + z_movement_speed_coupling_z')
lines.append('          + time_z + time_z2 + C(pair)')
lines.append('```')
lines.append('完整模型：')
lines.append('```text')
lines.append('brain_ibc ~ z_luminance_similarity + z_color_hist_similarity + z_visual_motion_similarity')
lines.append('          + z_click_coupling_z + z_movement_speed_coupling_z')
lines.append('          + social_interaction_index + time_z + time_z2 + C(pair)')
lines.append('```')
lines.append('每个 ROI 分别拟合基础模型和完整模型。主统计推断采用按被试对聚类的 cluster-robust 标准误，因为同一对被试贡献了多个时间窗，窗口之间不是独立观测。')
lines.append('')
lines.append('## 5. 数据完整性')
lines.append(qc.to_markdown(index=False))
lines.append('')
lines.append('## 6. social_interaction_index 的结果')
lines.append(main_df[['roi','n_rows','n_pairs','beta_cluster','se_cluster','t_cluster','p_cluster','p_fdr_4roi','delta_r2','r2_base','r2_full']].to_markdown(index=False, floatfmt='.4f'))
lines.append('')
for _, row in main_df.iterrows():
    lines.append('- {}：social index beta = {:.3f}, cluster p {}, 4-ROI FDR p {}, ΔR² = {:.3f}。'.format(row['roi'], row['beta_cluster'], fmtp(row['p_cluster']), fmtp(row['p_fdr_4roi']), row['delta_r2']))
lines.append('')
lines.append('## 7. ROI 斜率差异')
lines.append('以下模型直接检验其他 ROI 的 social index 斜率是否不同于 precuneus。')
lines.append(contrast_df.to_markdown(index=False, floatfmt='.4f'))
lines.append('')
lines.append('## 8. 三个社会互动组成指标的结果')
lines.append(comp_df[['component_label','roi','n_rows','n_pairs','beta_cluster','se_cluster','t_cluster','p_cluster','p_fdr_12tests','p_fdr_within_roi_3components','delta_r2']].to_markdown(index=False, floatfmt='.4f'))
lines.append('')
lines.append('## 9. 组成指标的 ROI 差异')
lines.append(comp_contrast_df.to_markdown(index=False, floatfmt='.4f'))
lines.append('')
lines.append('## 10. 结论性概括')
prec_row = main_df[main_df['roi'].eq('Precuneus')].iloc[0]
ba_row = main_df[main_df['roi'].eq('BA17')].iloc[0]
lines.append('在本版模型中，precuneus 的 social_interaction_index 仍然能够在控制直接低阶视觉相似性、低阶行为耦合、时间趋势和被试对固定效应之后预测 IBC（beta = {:.3f}, cluster p {}, FDR p {}, ΔR² = {:.3f}）。'.format(prec_row['beta_cluster'], fmtp(prec_row['p_cluster']), fmtp(prec_row['p_fdr_4roi']), prec_row['delta_r2']))
lines.append('BA17 的同一效应为 beta = {:.3f}, cluster p {}, FDR p {}。因此，当前结果支持 precuneus IBC 与高阶社会互动强度有关，并且这种关联不能简单归因于低阶画面相似性或基础行为耦合；但 ROI 间斜率差异是否显著仍需结合 contrast 表谨慎表述。'.format(ba_row['beta_cluster'], fmtp(ba_row['p_cluster']), fmtp(ba_row['p_fdr_4roi'])))
lines.append('')
lines.append('## 11. 输出文件')
for name in ['direct_similarity_social_index_coefficients_4roi.csv','direct_similarity_social_index_roi_contrasts_vs_precuneus.csv','direct_similarity_social_components_coefficients_4roi.csv','direct_similarity_social_components_roi_contrasts_vs_precuneus.csv','direct_similarity_social_index_4roi.png','direct_similarity_social_components_4roi.png','direct_similarity_ba17_vs_precuneus_component_contrasts.png','direct_similarity_social_index_partial_residuals_4roi.png']:
    lines.append('- `' + str(OUT / name) + '`')
(OUT / 'direct_similarity_model_visual1s_summary_cn.md').write_text('\n'.join(lines), encoding='utf-8')
print('Saved outputs to ' + str(OUT))
print('DONE')

