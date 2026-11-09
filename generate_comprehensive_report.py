#!/usr/bin/env python3
"""
HMARL-DRT Comprehensive Report Generator
Phase 3 (CEMRL Integration) vs Phase 4 (Fine-tuned) Analysis
Generates all plots and saves them to runs/comprehensive_report/
"""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import os
from pathlib import Path

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e94560',
    'axes.labelcolor': '#ffffff',
    'axes.titlecolor': '#ffffff',
    'text.color': '#ffffff',
    'xtick.color': '#cccccc',
    'ytick.color': '#cccccc',
    'grid.color': '#2a2a4a',
    'grid.alpha': 0.3,
    'legend.facecolor': '#1a1a2e',
    'legend.edgecolor': '#e94560',
    'legend.labelcolor': '#ffffff',
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'font.size': 10,
})

OUT_DIR = Path('/home/ubuntu/hmarl-drt/runs/comprehensive_report')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load Data ──────────────────────────────────────────────────────────────
phase3 = pd.read_csv('/home/ubuntu/hmarl-drt/runs/phase3_cemrl_integration/training_log.csv')
phase4 = pd.read_csv('/home/ubuntu/hmarl-drt/runs/phase4_finetune/training_log.csv')

with open('/home/ubuntu/hmarl-drt/runs/phase4_finetune/validation_results.json', 'r') as f:
    val_data = json.load(f)

# ── Derived columns ────────────────────────────────────────────────────────
def delta_to_regime(d):
    if d < 0.25: return 0
    elif d < 0.5: return 1
    elif d < 0.75: return 2
    else: return 3

for df in [phase3, phase4]:
    df['zero_violation'] = (df['physics_violations'] == 0).astype(int)
    df['true_regime'] = df['delta'].apply(delta_to_regime)
    df['regime_correct'] = (df['true_regime'] == df['cemrl/regime_id']).astype(int)
    df['reward_ma50'] = df['reward'].rolling(50, min_periods=1).mean()
    df['viol_ma50'] = df['physics_violations'].rolling(50, min_periods=1).mean()
    df['zero_viol_ma50'] = df['zero_violation'].rolling(50, min_periods=1).mean()
    df['regime_acc_ma50'] = df['regime_correct'].rolling(50, min_periods=1).mean()

# ── Helper ─────────────────────────────────────────────────────────────────
def save_fig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    print(f"  ✓ Saved {path}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Training Curves Comparison (Reward + Violations)
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1a: Raw reward
ax = axes[0, 0]
ax.plot(phase3['episode'], phase3['reward'], alpha=0.15, color='#4ecca3', linewidth=0.5)
ax.plot(phase3['episode'], phase3['reward_ma50'], color='#4ecca3', linewidth=2, label='Phase 3 (50-ep MA)')
ax.plot(phase4['episode'], phase4['reward'], alpha=0.15, color='#e94560', linewidth=0.5)
ax.plot(phase4['episode'], phase4['reward_ma50'], color='#e94560', linewidth=2, label='Phase 4 (50-ep MA)')
ax.axhline(y=175, color='yellow', linestyle='--', alpha=0.5, label='90% Peak Threshold')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.set_title('Cumulative Reward per Episode', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# 1b: Physics violations
ax = axes[0, 1]
ax.plot(phase3['episode'], phase3['physics_violations'], alpha=0.15, color='#4ecca3', linewidth=0.5)
ax.plot(phase3['episode'], phase3['viol_ma50'], color='#4ecca3', linewidth=2, label='Phase 3')
ax.plot(phase4['episode'], phase4['physics_violations'], alpha=0.15, color='#e94560', linewidth=0.5)
ax.plot(phase4['episode'], phase4['viol_ma50'], color='#e94560', linewidth=2, label='Phase 4')
ax.set_xlabel('Episode')
ax.set_ylabel('Physics Violations')
ax.set_title('Physics Violations per Episode', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# 1c: Zero-violation rate (rolling)
ax = axes[1, 0]
ax.plot(phase3['episode'], phase3['zero_viol_ma50'], color='#4ecca3', linewidth=2, label='Phase 3')
ax.plot(phase4['episode'], phase4['zero_viol_ma50'], color='#e94560', linewidth=2, label='Phase 4')
ax.set_xlabel('Episode')
ax.set_ylabel('Zero-Violation Rate (50-ep MA)')
ax.set_title('Safety: Zero-Violation Rate', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# 1d: Regime detection accuracy (rolling)
ax = axes[1, 1]
ax.plot(phase3['episode'], phase3['regime_acc_ma50'], color='#4ecca3', linewidth=2, label='Phase 3')
ax.plot(phase4['episode'], phase4['regime_acc_ma50'], color='#e94560', linewidth=2, label='Phase 4')
ax.axhline(y=0.977, color='#e94560', linestyle=':', alpha=0.7, label='Phase 4 Final: 97.7%')
ax.axhline(y=0.407, color='#4ecca3', linestyle=':', alpha=0.7, label='Phase 3 Final: 40.7%')
ax.set_xlabel('Episode')
ax.set_ylabel('Regime Detection Accuracy (50-ep MA)')
ax.set_title('PEARL Regime Detection Accuracy', fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.2)

fig.suptitle('HMARL-DRT: Phase 3 vs Phase 4 Training Comparison', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
save_fig(fig, '01_training_curves.png')

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Gating Weights Evolution
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

agents = ['g_nn', 'g_gp', 'g_fm', 'g_physics']
labels = ['SAC (NN)', 'GP/iPAE', 'Foundation Model', 'PINN Physics']
colors = ['#4ecca3', '#ff6b6b', '#ffd93d', '#6c5ce7']

for idx, (col, label, color) in enumerate(zip(agents, labels, colors)):
    ax = axes[idx // 2, idx % 2]
    ax.plot(phase3['episode'], phase3[col], alpha=0.3, color=color, linewidth=0.5)
    ax.plot(phase3['episode'], phase3[col].rolling(50, min_periods=1).mean(), 
            color=color, linewidth=2, linestyle='-', label='Phase 3')
    ax.plot(phase4['episode'], phase4[col], alpha=0.3, color=color, linewidth=0.5)
    ax.plot(phase4['episode'], phase4[col].rolling(50, min_periods=1).mean(), 
            color=color, linewidth=2, linestyle='--', label='Phase 4')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Gating Weight')
    ax.set_title(f'{label} Gating Weight', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-0.05, 1.05)

fig.suptitle('Agent Gating Weight Evolution Across Training', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
save_fig(fig, '02_gating_weights.png')

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Validation Performance Dashboard
# ══════════════════════════════════════════════════════════════════════════
val_rewards = [e['reward'] for e in val_data['episodes']]
val_violations = [e['violations'] for e in val_data['episodes']]
val_zero = [e['zero_violations'] for e in val_data['episodes']]
val_omega_max = [e['omega_max'] for e in val_data['episodes']]
val_T_max = [e['T_bearing_max'] for e in val_data['episodes']]
val_g_nn = [e['avg_weights'][0] for e in val_data['episodes']]
val_g_gp = [e['avg_weights'][1] for e in val_data['episodes']]
val_g_fm = [e['avg_weights'][2] for e in val_data['episodes']]
val_g_physics = [e['avg_weights'][3] for e in val_data['episodes']]

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# 3a: Validation reward distribution
ax = fig.add_subplot(gs[0, 0])
ax.hist(val_rewards, bins=30, color='#4ecca3', alpha=0.7, edgecolor='white', linewidth=0.5)
ax.axvline(np.mean(val_rewards), color='#e94560', linestyle='--', linewidth=2, 
           label=f'Mean: {np.mean(val_rewards):.1f}')
ax.axvline(np.median(val_rewards), color='#ffd93d', linestyle=':', linewidth=2,
           label=f'Median: {np.median(val_rewards):.1f}')
ax.set_xlabel('Reward')
ax.set_ylabel('Frequency')
ax.set_title('Validation Reward Distribution', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# 3b: Validation violations distribution
ax = fig.add_subplot(gs[0, 1])
ax.hist(val_violations, bins=30, color='#ff6b6b', alpha=0.7, edgecolor='white', linewidth=0.5)
ax.axvline(np.mean(val_violations), color='#e94560', linestyle='--', linewidth=2,
           label=f'Mean: {np.mean(val_violations):.1f}')
ax.axvline(np.median(val_violations), color='#ffd93d', linestyle=':', linewidth=2,
           label=f'Median: {np.median(val_violations):.1f}')
ax.set_xlabel('Violations')
ax.set_ylabel('Frequency')
ax.set_title('Validation Violations Distribution', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# 3c: Zero-violation pie chart
ax = fig.add_subplot(gs[0, 2])
zero_count = sum(val_zero)
non_zero_count = len(val_zero) - zero_count
wedges, texts, autotexts = ax.pie(
    [zero_count, non_zero_count],
    labels=['Zero Violations', 'Has Violations'],
    autopct='%1.1f%%',
    colors=['#4ecca3', '#ff6b6b'],
    startangle=90,
    textprops={'color': 'white', 'fontsize': 9}
)
for t in autotexts:
    t.set_color('white')
ax.set_title('Safety: Zero-Violation Episodes', fontweight='bold')

# 3d: Gating weights (validation)
ax = fig.add_subplot(gs[1, 0])
val_g_means = [np.mean(val_g_nn), np.mean(val_g_gp), np.mean(val_g_fm), np.mean(val_g_physics)]
bars = ax.bar(labels, val_g_means, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, val_g_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{val:.1%}', ha='center', va='bottom', color='white', fontsize=9)
ax.set_ylabel('Mean Weight')
ax.set_title('Validation Gating Weights', fontweight='bold')
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.2, axis='y')

# 3e: Reward vs Violations scatter
ax = fig.add_subplot(gs[1, 1])
sc = ax.scatter(val_violations, val_rewards, c=val_zero, cmap='RdYlGn', alpha=0.6, 
                s=30, edgecolors='white', linewidth=0.3)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Zero Violations', color='white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.get_yticklabels(), color='white')
ax.set_xlabel('Physics Violations')
ax.set_ylabel('Reward')
ax.set_title('Reward vs Violations Trade-off', fontweight='bold')
ax.grid(True, alpha=0.2)

# 3f: Omega max distribution
ax = fig.add_subplot(gs[1, 2])
ax.hist(val_omega_max, bins=30, color='#6c5ce7', alpha=0.7, edgecolor='white', linewidth=0.5)
ax.axvline(2.0, color='#e94560', linestyle='--', linewidth=2, label='Hard Limit (2.0)')
ax.set_xlabel('Max Rotor Speed (rad/s)')
ax.set_ylabel('Frequency')
ax.set_title('Peak Rotor Speed Distribution', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# 3g: Bearing temperature
ax = fig.add_subplot(gs[2, 0])
ax.hist(val_T_max, bins=30, color='#ffd93d', alpha=0.7, edgecolor='white', linewidth=0.5)
ax.axvline(120, color='#e94560', linestyle='--', linewidth=2, label='Safety Limit (120°C)')
ax.set_xlabel('Max Bearing Temp (°C)')
ax.set_ylabel('Frequency')
ax.set_title('Peak Bearing Temperature', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# 3h: Regime distribution in validation
ax = fig.add_subplot(gs[2, 1])
regime_counts = {}
for e in val_data['episodes']:
    for r, c in e['regime_counts'].items():
        regime_counts[f'Regime {r}'] = regime_counts.get(f'Regime {r}', 0) + c
reg_labels = list(regime_counts.keys())
reg_values = list(regime_counts.values())
reg_colors = ['#4ecca3', '#ffd93d', '#ff9f43', '#e94560']
bars = ax.bar(reg_labels, reg_values, color=reg_colors[:len(reg_labels)], alpha=0.8, edgecolor='white')
for bar, val in zip(bars, reg_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(reg_values)*0.01,
            f'{val}', ha='center', va='bottom', color='white', fontsize=9)
ax.set_ylabel('Total Steps')
ax.set_title('Regime Distribution in Validation', fontweight='bold')
ax.grid(True, alpha=0.2, axis='y')

# 3i: Per-class regime accuracy
ax = fig.add_subplot(gs[2, 2])
regime_accs = [1.0, 1.0, 1.0, 0.771]  # from Phase 4 training log
bars = ax.bar(['Healthy (0)', 'Degrading (1)', 'Critical (2)', 'Emergency (3)'], 
              regime_accs, color=reg_colors, alpha=0.8, edgecolor='white')
for bar, val in zip(bars, regime_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.1%}', ha='center', va='bottom', color='white', fontsize=9)
ax.set_ylabel('Accuracy')
ax.set_title('Per-Class Regime Detection Accuracy', fontweight='bold')
ax.set_ylim(0, 1.15)
ax.grid(True, alpha=0.2, axis='y')

fig.suptitle('Phase 4 Validation Performance Dashboard', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
save_fig(fig, '03_validation_dashboard.png')

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Phase Comparison Radar Chart
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

categories = ['Reward\n(mean)', 'Reward\n(max)', 'Zero-Violation\nRate', 
              'Regime\nDetection', 'Gating\nBalance', 'Final 10\nMean Reward']
N = len(categories)

# Normalize values to [0, 1] for radar
p3_vals = [
    phase3['reward'].mean() / 200,           # mean reward
    phase3['reward'].max() / 200,             # max reward
    (phase3['physics_violations'] == 0).mean() * 5,  # zero-viol rate (scaled)
    0.407,                                     # regime detection
    1 - abs(phase3['g_nn'].mean() - 0.25) * 2,  # gating balance (0.25 ideal)
    phase3.tail(10)['reward'].mean() / 200     # final 10 mean
]
p4_vals = [
    phase4['reward'].mean() / 200,
    phase4['reward'].max() / 200,
    (phase4['physics_violations'] == 0).mean() * 5,
    0.977,
    1 - abs(phase4['g_nn'].mean() - 0.25) * 2,
    phase4.tail(10)['reward'].mean() / 200
]

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
p3_vals += p3_vals[:1]
p4_vals += p4_vals[:1]
angles += angles[:1]

ax.plot(angles, p3_vals, 'o-', linewidth=2, color='#4ecca3', label='Phase 3', markersize=8)
ax.fill(angles, p3_vals, alpha=0.1, color='#4ecca3')
ax.plot(angles, p4_vals, 'o-', linewidth=2, color='#e94560', label='Phase 4', markersize=8)
ax.fill(angles, p4_vals, alpha=0.1, color='#e94560')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 1.0)
ax.set_title('Phase 3 vs Phase 4: Multi-Metric Radar Comparison', 
             fontsize=14, fontweight='bold', pad=25)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

save_fig(fig, '04_radar_comparison.png')

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Sample Efficiency (Steps to 90% Performance)
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 5a: Steps to 90% bar chart
ax = axes[0]
p3_90_ep = phase3[phase3['reward'] >= 0.9*phase3['reward'].max()].iloc[0]['episode']
p4_90_ep = phase4[phase4['reward'] >= 0.9*phase4['reward'].max()].iloc[0]['episode']
bars = ax.bar(['Phase 3', 'Phase 4'], [int(p3_90_ep), int(p4_90_ep)], 
              color=['#4ecca3', '#e94560'], alpha=0.8, edgecolor='white', width=0.5)
for bar, val in zip(bars, [int(p3_90_ep), int(p4_90_ep)]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'Episode {val}', ha='center', va='bottom', color='white', fontsize=12, fontweight='bold')
ax.set_ylabel('Episode to Reach 90% Peak Reward')
ax.set_title('Sample Efficiency: Episodes to 90% Performance', fontweight='bold')
ax.grid(True, alpha=0.2, axis='y')

# 5b: Learning curve zoom (first 100 episodes)
ax = axes[1]
ax.plot(phase3['episode'].iloc[:100], phase3['reward'].iloc[:100], 
        color='#4ecca3', linewidth=1.5, alpha=0.7, label='Phase 3')
ax.plot(phase4['episode'].iloc[:100], phase4['reward'].iloc[:100], 
        color='#e94560', linewidth=1.5, alpha=0.7, label='Phase 4')
ax.axhline(y=0.9*phase3['reward'].max(), color='#4ecca3', linestyle='--', alpha=0.5, 
           label=f'Phase 3 90% ({0.9*phase3["reward"].max():.0f})')
ax.axhline(y=0.9*phase4['reward'].max(), color='#e94560', linestyle='--', alpha=0.5,
           label=f'Phase 4 90% ({0.9*phase4["reward"].max():.0f})')
ax.axvline(x=int(p3_90_ep), color='#4ecca3', linestyle=':', alpha=0.5)
ax.axvline(x=int(p4_90_ep), color='#e94560', linestyle=':', alpha=0.5)
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.set_title('Early Learning: First 100 Episodes', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

fig.suptitle('Sample Efficiency Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig(fig, '05_sample_efficiency.png')

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Regime Detection Confusion Matrix (from training log)
# ══════════════════════════════════════════════════════════════════════════
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (df, label, cmap) in enumerate(zip(
    [phase3, phase4], ['Phase 3', 'Phase 4'], ['Greens', 'Reds'])):
    ax = axes[idx]
    cm = confusion_matrix(df['true_regime'], df['cemrl/regime_id'], labels=[0, 1, 2, 3])
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xticklabels(['Healthy', 'Degrading', 'Critical', 'Emergency'], fontsize=8)
    ax.set_yticklabels(['Healthy', 'Degrading', 'Critical', 'Emergency'], fontsize=8)
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{cm_norm[i, j]:.2f}\n({cm[i, j]})', 
                    ha='center', va='center', color='white' if cm_norm[i, j] > 0.5 else 'black',
                    fontsize=8)
    
    ax.set_xlabel('Predicted Regime')
    ax.set_ylabel('True Regime')
    ax.set_title(f'{label} Confusion Matrix\n(Accuracy: {df["regime_correct"].mean():.1%})', 
                 fontweight='bold')

plt.tight_layout()
save_fig(fig, '06_confusion_matrices.png')

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 7: Reward Component Breakdown
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

reward_components = ['r_energy', 'r_rul', 'r_safety', 'r_physics', 'r_curiosity']
comp_labels = ['Energy', 'RUL', 'Safety', 'Physics', 'Curiosity']
comp_colors = ['#4ecca3', '#ffd93d', '#ff6b6b', '#6c5ce7', '#ff9f43']

for idx, (df, label) in enumerate(zip([phase3, phase4], ['Phase 3', 'Phase 4'])):
    ax = axes[idx]
    means = [df[c].mean() for c in reward_components]
    bars = ax.bar(comp_labels, means, color=comp_colors, alpha=0.8, edgecolor='white')
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', color='white', fontsize=8)
    ax.set_ylabel('Mean Component Value')
    ax.set_title(f'{label} Reward Component Breakdown', fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
save_fig(fig, '07_reward_components.png')

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 8: Validation Episode-by-Episode Analysis
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

episodes = list(range(len(val_data['episodes'])))

# 8a: Reward per validation episode
ax = axes[0]
ax.plot(episodes, val_rewards, color='#4ecca3', linewidth=1, alpha=0.7)
ax.fill_between(episodes, val_rewards, alpha=0.1, color='#4ecca3')
ax.axhline(y=np.mean(val_rewards), color='#e94560', linestyle='--', 
           label=f'Mean: {np.mean(val_rewards):.1f}')
ax.axhline(y=np.median(val_rewards), color='#ffd93d', linestyle=':', 
           label=f'Median: {np.median(val_rewards):.1f}')
ax.set_xlabel('Validation Episode')
ax.set_ylabel('Reward')
ax.set_title('Validation: Reward per Episode', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# 8b: Violations per validation episode
ax = axes[1]
ax.plot(episodes, val_violations, color='#ff6b6b', linewidth=1, alpha=0.7)
ax.fill_between(episodes, val_violations, alpha=0.1, color='#ff6b6b')
ax.axhline(y=np.mean(val_violations), color='#e94560', linestyle='--',
           label=f'Mean: {np.mean(val_violations):.1f}')
ax.set_xlabel('Validation Episode')
ax.set_ylabel('Physics Violations')
ax.set_title('Validation: Violations per Episode', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# 8c: Gating weights per validation episode
ax = axes[2]
ax.plot(episodes, val_g_nn, color='#4ecca3', linewidth=1, alpha=0.7, label='SAC (NN)')
ax.plot(episodes, val_g_gp, color='#ff6b6b', linewidth=1, alpha=0.7, label='GP/iPAE')
ax.plot(episodes, val_g_fm, color='#ffd93d', linewidth=1, alpha=0.7, label='FM')
ax.plot(episodes, val_g_physics, color='#6c5ce7', linewidth=1, alpha=0.7, label='PINN Physics')
ax.set_xlabel('Validation Episode')
ax.set_ylabel('Gating Weight')
ax.set_title('Validation: Agent Gating Weights', fontweight='bold')
ax.legend(fontsize=8, ncol=4)
ax.grid(True, alpha=0.2)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
save_fig(fig, '08_validation_episodes.png')

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 9: Summary Metrics Comparison Table (as image)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 8))
ax.axis('off')

# Build comparison table data
table_data = [
    ['Metric', 'Phase 3', 'Phase 4', 'Δ Change', 'Assessment'],
    ['Total Episodes', '1000', '481', '-51.9%', ''],
    ['Max Reward', f'{phase3["reward"].max():.2f}', f'{phase4["reward"].max():.2f}', 
     f'{phase4["reward"].max() - phase3["reward"].max():+.2f}', '≈ Same'],
    ['Mean Reward', f'{phase3["reward"].mean():.2f}', f'{phase4["reward"].mean():.2f}',
     f'{phase4["reward"].mean() - phase3["reward"].mean():+.2f}', '↑ Slight'],
    ['Final 10 Ep Mean Reward', f'{phase3.tail(10)["reward"].mean():.2f}', 
     f'{phase4.tail(10)["reward"].mean():.2f}',
     f'{phase4.tail(10)["reward"].mean() - phase3.tail(10)["reward"].mean():+.2f}', '↑↑ Improved'],
    ['Mean Physics Violations', f'{phase3["physics_violations"].mean():.1f}',
     f'{phase4["physics_violations"].mean():.1f}',
     f'{phase4["physics_violations"].mean() - phase3["physics_violations"].mean():+.1f}', '↓ Reduced'],
    ['Zero-Violation Rate', f'{(phase3["physics_violations"]==0).mean():.1%}',
     f'{(phase4["physics_violations"]==0).mean():.1%}',
     f'{(phase4["physics_violations"]==0).mean() - (phase3["physics_violations"]==0).mean():+.1%}', '↑ Slight'],
    ['Regime Detection Accuracy', f'{phase3["regime_correct"].mean():.1%}',
     f'{phase4["regime_correct"].mean():.1%}',
     f'{phase4["regime_correct"].mean() - phase3["regime_correct"].mean():+.1%}', '🚀 HUGE!'],
    ['Episodes to 90% Reward', f'{int(p3_90_ep)}', f'{int(p4_90_ep)}',
     f'{int(p4_90_ep) - int(p3_90_ep)}', '≈ Same'],
    ['SAC (NN) Weight', f'{phase3["g_nn"].mean():.3f}', f'{phase4["g_nn"].mean():.3f}',
     f'{phase4["g_nn"].mean() - phase3["g_nn"].mean():+.3f}', '↓ Decreased'],
    ['GP/iPAE Weight', f'{phase3["g_gp"].mean():.3f}', f'{phase4["g_gp"].mean():.3f}',
     f'{phase4["g_gp"].mean() - phase3["g_gp"].mean():+.3f}', '↑ Increased'],
    ['FM Weight', f'{phase3["g_fm"].mean():.3f}', f'{phase4["g_fm"].mean():.3f}',
     f'{phase4["g_fm"].mean() - phase3["g_fm"].mean():+.3f}', '↓ Decreased'],
    ['PINN Physics Weight', f'{phase3["g_physics"].mean():.3f}', f'{phase4["g_physics"].mean():.3f}',
     f'{phase4["g_physics"].mean() - phase3["g_physics"].mean():+.3f}', '↑ Increased'],
]

table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.18, 0.12, 0.12, 0.12, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)

# Style header
for j in range(5):
    cell = table[0, j]
    cell.set_facecolor('#e94560')
    cell.set_text_props(color='white', fontweight='bold')

# Style data rows
for i in range(1, len(table_data)):
    for j in range(5):
        cell = table[i, j]
        cell.set_facecolor('#16213e' if i % 2 == 0 else '#1a1a2e')
        cell.set_text_props(color='white')
        if j == 4:  # Assessment column
            val = table_data[i][4]
            if 'HUGE' in val:
                cell.set_text_props(color='#4ecca3', fontweight='bold')
            elif 'Improved' in val or 'Reduced' in val or 'Increased' in val:
                cell.set_text_props(color='#4ecca3')
            elif 'Decreased' in val:
                cell.set_text_props(color='#ffd93d')

ax.set_title('Phase 3 vs Phase 4: Comprehensive Metrics Comparison', 
             fontsize=16, fontweight='bold', pad=20, color='white')

save_fig(fig, '09_metrics_comparison_table.png')

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 10: Validation Summary Card
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')

s = val_data['summary']
val_text = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    PHASE 4 VALIDATION SUMMARY (200 Episodes)            ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  REWARD                                                                 ║
║    • Mean:          {s['mean_reward']:>8.2f} ± {s['std_reward']:>6.2f}                              ║
║    • Median:        {np.median(val_rewards):>8.2f}                                       ║
║    • Range:         [{np.min(val_rewards):>7.1f}, {np.max(val_rewards):>7.1f}]                                   ║
║                                                                        ║
║  SAFETY                                                                 ║
║    • Mean Violations:  {s['mean_violations_per_ep']:>5.1f} per episode                          ║
║    • Zero-Violation:   {s['zero_violation_rate']:.1%} of episodes                            ║
║    • Omega Limit Viol: {s['omega_hard_limit_violation_count']:>6d} total                           ║
║    • Temp Limit Viol:  {s['T_hard_limit_violation_count']:>6d} total (limit 120°C)                    ║
║                                                                        ║
║  GATING WEIGHTS                                                         ║
║    • SAC (NN):       {s['mean_sac_weight']:.3f}                                        ║
║    • GP/iPAE:        {s['mean_gp_weight']:.3f}                                        ║
║    • FM:             {s['mean_fm_weight']:.3f}                                        ║
║    • PINN Physics:   {s['mean_physics_weight']:.3f}                                        ║
║                                                                        ║
║  PHYSICAL LIMITS                                                         ║
║    • Max Omega Mean: {s['max_omega_mean']:.3f} rad/s (limit: 2.0)                          ║
║    • Max Bearing T:  {s['max_T_bearing_mean']:.1f} °C (limit: 120°C)                          ║
║                                                                        ║
║  COMPUTATION                                                            ║
║    • Episodes:       {s['n_episodes']:>3d}                                            ║
║    • Elapsed Time:   {s['elapsed_s']:>5.1f}s ({s['elapsed_s']/60:.1f} min)                                  ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

ax.text(0.5, 0.5, val_text, transform=ax.transAxes, fontsize=10, 
        fontfamily='monospace', ha='center', va='center', color='#4ecca3',
        bbox=dict(boxstyle='round', facecolor='#16213e', edgecolor='#e94560', alpha=0.9))

ax.set_title('Phase 4 Validation Results Summary', fontsize=14, fontweight='bold', pad=10)

save_fig(fig, '10_validation_summary_card.png')

print(f"\n{'='*60}")
print(f"All plots saved to: {OUT_DIR}")
print(f"{'='*60}")
print(f"\nGenerated {10} figures:")
for f in sorted(OUT_DIR.glob('*.png')):
    print(f"  • {f.name}")
