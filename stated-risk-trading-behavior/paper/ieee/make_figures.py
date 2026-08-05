from __future__ import annotations

from pathlib import Path
import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

MATLAB_BLUE = '#0072BD'
MATLAB_ORANGE = '#D95319'
MATLAB_GREEN = '#77AC30'
DARK_BLUE = '#005A94'
LIGHT_BLUE = '#7FB9DE'
GRID_GRAY = '#D9D9D9'
REFERENCE_GRAY = '#777777'
STEM_GRAY = '#A6A6A6'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create publication figures for the FAR-Trans paper.')
    parser.add_argument(
        '--project-root',
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help='Root directory of the verified FAR-Trans research package.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).resolve().parent / 'figures',
        help='Directory for vector PDF figures.',
    )
    return parser.parse_args()


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 7.5,
    'axes.labelsize': 7.5,
    'xtick.labelsize': 7.0,
    'ytick.labelsize': 7.0,
    'legend.fontsize': 6.8,
    'axes.linewidth': 0.7,
    'lines.linewidth': 1.15,
    'lines.markersize': 4.5,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def clean_axes(ax: plt.Axes) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=2.5, width=0.6)
    ax.grid(axis='y', linewidth=0.45, color=GRID_GRAY, alpha=0.85)
    ax.set_axisbelow(True)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / total
    denom = 1 + z*z/total
    center = (p + z*z/(2*total)) / denom
    half = z * math.sqrt((p*(1-p) + z*z/(4*total))/total) / denom
    return center-half, center+half


def main() -> None:
    args = parse_args()
    root = args.project_root
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    risk = pd.read_csv(root / 'results_final/descriptive_by_risk.csv')
    order = ['Conservative', 'Income', 'Balanced', 'Aggressive']
    risk = risk.set_index('riskLevel').loc[order].reset_index()
    ci = np.array([wilson_interval(int(a), int(n)) for a, n in zip(risk['actions'], risk['exposures'])])
    y = 100*risk['action_rate'].to_numpy()
    yerr = np.vstack((y-100*ci[:,0], 100*ci[:,1]-y))
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    x = np.arange(len(order))
    ax.errorbar(
        x, y, yerr=yerr, fmt='o', color=DARK_BLUE, ecolor=LIGHT_BLUE,
        markerfacecolor=MATLAB_BLUE, markeredgecolor=DARK_BLUE,
        capsize=2.5, capthick=0.8, elinewidth=1.0,
    )
    ax.set_xticks(x, order)
    ax.set_ylabel('Trading within five days (%)')
    ax.set_ylim(0, 9)
    clean_axes(ax)
    fig.savefig(out / 'fig1_response_by_risk.pdf')
    plt.close(fig)

    odds = pd.read_csv(root / 'results_final/action_inference_odds_ratios.csv')
    term_map = {
        "C(riskLevel, Treatment(reference='Conservative'))[T.Income]": 'Income risk category',
        "C(riskLevel, Treatment(reference='Conservative'))[T.Balanced]": 'Balanced risk category',
        "C(riskLevel, Treatment(reference='Conservative'))[T.Aggressive]": 'Aggressive risk category',
        'log_position_value_z': 'Position value',
        'log_days_since_asset_trade_z': 'Time since last trade in the security',
        'log_prior_transactions_z': 'Cumulative prior transaction count',
        'log_prior_transactions_90d_z': 'Transactions during the preceding 90 days',
        'prior_action_rate_smoothed_z': 'Prior response rate after earlier events',
    }
    sel = odds[odds['term'].isin(term_map)].copy()
    sel['label'] = sel['term'].map(term_map)
    label_order = [
        'Income risk category',
        'Balanced risk category',
        'Aggressive risk category',
        'Position value',
        'Time since last trade in the security',
        'Cumulative prior transaction count',
        'Transactions during the preceding 90 days',
        'Prior response rate after earlier events',
    ]
    sel = sel.set_index('label').loc[label_order].reset_index()
    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    ypos = np.arange(len(sel))[::-1]
    ax.errorbar(
        sel['OR'], ypos,
        xerr=np.vstack((sel['OR']-sel['OR_low'], sel['OR_high']-sel['OR'])),
        fmt='o', color=DARK_BLUE, ecolor=LIGHT_BLUE,
        markerfacecolor=MATLAB_BLUE, markeredgecolor=DARK_BLUE,
        capsize=2.2, capthick=0.8, elinewidth=1.0,
    )
    ax.axvline(1.0, color=REFERENCE_GRAY, linestyle='--', linewidth=0.9)
    ax.set_yticks(ypos, sel['label'])
    ax.set_xlabel('Adjusted odds ratio (95% confidence interval)')
    ax.set_xlim(0.45, 1.85)
    ax.grid(axis='x', linewidth=0.45, color=GRID_GRAY, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=2.5, width=0.6)
    fig.savefig(out / 'fig2_selected_odds_ratios.pdf')
    plt.close(fig)

    year = pd.read_csv(root / 'results_final/descriptive_by_year.csv')
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    ax.plot(
        year['year'], 100*year['action_rate'], marker='o',
        color=MATLAB_BLUE, markerfacecolor=MATLAB_BLUE, markeredgecolor=DARK_BLUE,
    )
    ax.set_xticks(year['year'])
    ax.set_ylabel('Trading within five days (%)')
    ax.set_xlabel('Event year')
    ax.set_ylim(0, 11)
    clean_axes(ax)
    fig.savefig(out / 'fig3_response_by_year.pdf')
    plt.close(fig)

    bootstrap = pd.read_csv(root / 'results_final/bootstrap_metric_differences.csv')
    comparison_order = [
        'M1_plus_profile minus M0_event_position',
        'M4_plus_prior_shocks minus M3_profile_behavior',
        'R1_risk_only minus M0_event_position',
        'R2_behavior_plus_risk minus M2_plus_behavior',
    ]
    comparison_labels = {
        'M1_plus_profile minus M0_event_position': 'Customer profiles added',
        'M4_plus_prior_shocks minus M3_profile_behavior': 'Prior extreme-event responses added',
        'R1_risk_only minus M0_event_position': 'Stated risk category added alone',
        'R2_behavior_plus_risk minus M2_plus_behavior': 'Stated risk added after past behavior',
    }
    boot = bootstrap[bootstrap['metric'] == 'PR_diff'].set_index('comparison').loc[comparison_order].reset_index()
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    ypos = np.arange(len(boot))[::-1]
    ax.errorbar(
        boot['mean_diff'], ypos,
        xerr=np.vstack((boot['mean_diff']-boot['ci_low'], boot['ci_high']-boot['mean_diff'])),
        fmt='o', color=DARK_BLUE, ecolor=LIGHT_BLUE,
        markerfacecolor=MATLAB_BLUE, markeredgecolor=DARK_BLUE,
        capsize=2.5, capthick=0.8, elinewidth=1.1,
    )
    ax.axvline(0.0, color=REFERENCE_GRAY, linestyle='--', linewidth=0.9)
    ax.set_yticks(ypos, [comparison_labels[c] for c in boot['comparison']])
    ax.set_xlabel('Change in precision-recall area')
    ax.set_xlim(-0.025, 0.055)
    ax.grid(axis='x', linewidth=0.45, color=GRID_GRAY, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=2.5, width=0.6)
    fig.savefig(out / 'fig4_action_model_performance.pdf')
    plt.close(fig)

    ext = pd.read_csv(root / 'results_external/external_validation_model_metrics.csv')
    scenario_order = ['original', 'official_action_screened', 'proxy_consensus_3of4', 'mechanical_conservative']
    scenario_labels = ['Original sample', 'Official actions removed', 'Market-proxy consensus', 'Conservative mechanical screen']
    series = [
        ('M1_plus_profile', 'Customer profile information', 'o', '-', MATLAB_BLUE),
        ('M3_profile_behavior', 'Profile and historical trading behavior', 's', '--', MATLAB_ORANGE),
        ('M4_plus_prior_shocks', 'Add prior responses to extreme returns', '^', '-.', MATLAB_GREEN),
    ]
    fig, ax = plt.subplots(figsize=(7.1, 2.65))
    x = np.arange(len(scenario_order))
    for model, label, marker, ls, color in series:
        vals = ext[ext['model_short'] == model].set_index('scenario').loc[scenario_order]['PR_AUC'].to_numpy()
        ax.plot(
            x, vals, marker=marker, linestyle=ls, color=color,
            markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.0,
            label=label,
        )
    ax.set_xticks(x, scenario_labels)
    ax.set_ylabel('Precision-recall area under the curve')
    ax.set_ylim(0.19, 0.29)
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.18))
    clean_axes(ax)
    fig.savefig(out / 'fig5_external_validation.pdf')
    plt.close(fig)

    print('Created:', *sorted(str(p) for p in out.glob('*.pdf')), sep='\n')


if __name__ == '__main__':
    main()
