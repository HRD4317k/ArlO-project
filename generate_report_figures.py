#!/usr/bin/env python3
"""
Generate all plots for the CEMRL-HMARL-DRT Supervised Test Report.

Produces figures in docs/report_figures/:
  fig1_training_reward.png      - reward trend over 5000 episodes (rolling avg)
  fig2_energy_violations.png    - energy production & violations over training
  fig3_regime_coverage.png      - regime coverage pie/bar
  fig4_gating_by_regime.png     - gating weights by dominant regime
  fig5_reward_weights.png       - reward-shaping weights by regime
  fig6_rul_accuracy.png         - RUL estimate vs true delta (scatter)
  fig7_losses.png               - agent loss trends
  fig8_degradation_control.png  - degradation Δ change by action mode (R1/R2)
  fig9_energy_by_mode.png       - energy production by action mode
  fig10_adaptation_speed.png    - CEMRL vs baseline adaptation
  fig11_checkpoint_validation.png - reward/energy across all checkpoints
"""

import csv
import json
import os
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("docs/report_figures")
OUT.mkdir(parents=True, exist_ok=True)

# Color palette
C = {
    "sac": "#1f77b4", "ipae": "#ff7f0e", "fm": "#2ca02c", "pinn": "#d62728",
    "combined": "#9467bd", "zero": "#8c564b",
    "r0": "#2ca02c", "r1": "#1f77b4", "r2": "#ff7f0e", "r3": "#d62728",
}


def load_training_log():
    rows = list(csv.DictReader(open("hmarl_drt/runs/fresh_ipae_supervised/training_log.csv")))
    f = lambda r, k: float(r[k])
    return rows, f


def rolling_mean(vals, window=50):
    out = []
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        out.append(float(np.mean(vals[lo:i + 1])))
    return out


def fig1_training_reward(rows, f):
    eps = [f(r, "episode") for r in rows]
    rew = [f(r, "reward") for r in rows]
    rm = rolling_mean(rew, 100)
    plt.figure(figsize=(10, 5))
    plt.plot(eps, rew, alpha=0.15, color=C["sac"], label="per-episode")
    plt.plot(eps, rm, color=C["sac"], lw=2, label="100-ep rolling mean")
    plt.axhline(np.mean(rew), color="k", ls="--", lw=1, label=f"overall mean={np.mean(rew):.0f}")
    plt.xlabel("Episode"); plt.ylabel("Reward"); plt.title("Training Reward over 5000 Episodes")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(OUT / "fig1_training_reward.png", dpi=150); plt.close()


def fig2_energy_violations(rows, f):
    eps = [f(r, "episode") for r in rows]
    energy = rolling_mean([f(r, "r_energy") for r in rows], 100)
    viol = rolling_mean([f(r, "physics_violations") / 10 for r in rows], 100)  # % of 1000
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(eps, energy, color=C["ipae"], lw=2, label="energy production (r_energy)")
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Energy production", color=C["ipae"])
    ax1.tick_params(axis="y", labelcolor=C["ipae"])
    ax2 = ax1.twinx()
    ax2.plot(eps, viol, color=C["pinn"], lw=2, label="violations (%)")
    ax2.set_ylabel("Violations (%)", color=C["pinn"])
    ax2.tick_params(axis="y", labelcolor=C["pinn"])
    plt.title("Energy Production vs Violations over Training")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88)); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "fig2_energy_violations.png", dpi=150); plt.close()


def fig3_regime_coverage(rows, f):
    fracs = [st.mean(f(r, f"regime_frac_{i}") for r in rows) for i in range(4)]
    labels = ["R0 healthy", "R1 degrading", "R2 critical", "R3 emergency"]
    colors = [C["r0"], C["r1"], C["r2"], C["r3"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.pie(fracs, labels=labels, colors=colors, autopct="%.1f%%", startangle=90)
    ax1.set_title("Regime Coverage (fraction of steps)")
    ax2.bar(labels, [x * 100 for x in fracs], color=colors)
    ax2.set_ylabel("% of steps"); ax2.set_title("Regime Coverage")
    ax2.set_ylim(0, 50)
    for i, v in enumerate(fracs):
        ax2.text(i, v * 100 + 1, f"{v*100:.1f}%", ha="center")
    plt.tight_layout(); plt.savefig(OUT / "fig3_regime_coverage.png", dpi=150); plt.close()


def fig4_gating_by_regime(rows, f):
    agents = ["SAC", "iPAE", "FM", "PINN"]
    keys = ["g_nn", "g_gp", "g_fm", "g_physics"]
    colors = [C["sac"], C["ipae"], C["fm"], C["pinn"]]
    regimes = []
    for reg in range(4):
        grp = [r for r in rows if max(range(4), key=lambda i: f(r, f"regime_frac_{i}")) == reg]
        if grp:
            regimes.append((f"R{reg}", [st.mean(f(r, k) for r in grp) for k in keys]))
    x = np.arange(len(regimes)); w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, color) in enumerate(zip(agents, colors)):
        vals = [r[1][i] for r in regimes]
        ax.bar(x + (i - 1.5) * w, vals, w, label=name, color=color)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in regimes])
    ax.set_ylabel("Gating weight"); ax.set_title("Arbiter Gating by Dominant Regime")
    ax.axhline(0.25, color="k", ls="--", lw=1, label="uniform 0.25")
    ax.legend(); ax.grid(alpha=0.3, axis="y"); plt.tight_layout()
    plt.savefig(OUT / "fig4_gating_by_regime.png", dpi=150); plt.close()


def fig5_reward_weights(rows, f):
    wkeys = ["w_energy", "w_rul", "w_safety", "w_physics", "w_curiosity", "w_degradation"]
    wlabels = ["Energy", "RUL", "Safety", "Physics", "Curiosity", "Degradation"]
    regimes = []
    for reg in range(4):
        grp = [r for r in rows if max(range(4), key=lambda i: f(r, f"regime_frac_{i}")) == reg]
        if grp:
            regimes.append((f"R{reg}", [st.mean(f(r, k) for r in grp) for k in wkeys]))
    x = np.arange(len(regimes)); w = 0.13
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (name, color) in enumerate(zip(wlabels, plt.cm.tab10.colors)):
        vals = [r[1][i] for r in regimes]
        ax.bar(x + (i - 2.5) * w, vals, w, label=name, color=color)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in regimes])
    ax.set_ylabel("Reward weight"); ax.set_title("Reward-Shaping Weights by Regime")
    ax.legend(ncol=3, fontsize=8); ax.grid(alpha=0.3, axis="y"); plt.tight_layout()
    plt.savefig(OUT / "fig5_reward_weights.png", dpi=150); plt.close()


def fig6_rul_accuracy(rows, f):
    deltas = [f(r, "delta") for r in rows]
    ruls = [f(r, "rul_estimate") for r in rows]
    # Sample for plot
    idx = np.random.RandomState(42).choice(len(deltas), 2000, replace=False)
    plt.figure(figsize=(8, 6))
    plt.scatter([deltas[i] for i in idx], [ruls[i] for i in idx], s=8, alpha=0.3, color=C["ipae"])
    # True RUL line: RUL = 1000*(1-delta)
    xs = np.linspace(0, 1, 100)
    plt.plot(xs, 1000 * (1 - xs), "k--", lw=2, label="true RUL = 1000(1-δ)")
    corr = st.correlation(deltas, ruls)
    plt.xlabel("Degradation index δ"); plt.ylabel("RUL estimate")
    plt.title(f"iPAE RUL Estimation Accuracy (corr = {corr:.3f})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(OUT / "fig6_rul_accuracy.png", dpi=150); plt.close()


def fig7_losses(rows, f):
    eps = [f(r, "episode") for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plots = [
        ("loss_sac_actor", "SAC actor loss", axes[0, 0], C["sac"]),
        ("loss_sac_critic", "SAC critic loss", axes[0, 1], C["sac"]),
        ("loss_gp", "iPAE loss", axes[1, 0], C["ipae"]),
        ("loss_pinn_policy", "PINN policy loss", axes[1, 1], C["pinn"]),
    ]
    for key, title, ax, color in plots:
        vals = [f(r, key) for r in rows]
        ax.plot(eps, vals, color=color, lw=1)
        ax.set_title(title); ax.set_xlabel("Episode"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "fig7_losses.png", dpi=150); plt.close()


def fig8_degradation_control():
    data = json.load(open("results/ep2300_deepdive_results.json"))
    modes = ["combined", "zero", "sac", "ipae", "fm", "pinn"]
    labels = ["Combined", "Zero", "SAC", "iPAE", "FM", "PINN"]
    colors = [C["combined"], C["zero"], C["sac"], C["ipae"], C["fm"], C["pinn"]]
    r1 = [data["results"]["R1_start"][m]["avg_delta_change"] for m in modes]
    r2 = [data["results"]["R2_start"][m]["avg_delta_change"] for m in modes]
    x = np.arange(len(modes)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, r1, w, label="R1-start", color=C["r1"])
    ax.bar(x + w / 2, r2, w, label="R2-start", color=C["r2"])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Δ change over episode"); ax.set_title("Degradation Control by Action Mode")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    for i in range(len(modes)):
        ax.text(i - w / 2, r1[i] + 0.0003, f"{r1[i]:.4f}", ha="center", fontsize=7)
        ax.text(i + w / 2, r2[i] + 0.0003, f"{r2[i]:.4f}", ha="center", fontsize=7)
    plt.tight_layout(); plt.savefig(OUT / "fig8_degradation_control.png", dpi=150); plt.close()


def fig9_energy_by_mode():
    data = json.load(open("results/ep2300_deepdive_results.json"))
    modes = ["combined", "zero", "sac", "ipae", "fm", "pinn"]
    labels = ["Combined", "Zero", "SAC", "iPAE", "FM", "PINN"]
    colors = [C["combined"], C["zero"], C["sac"], C["ipae"], C["fm"], C["pinn"]]
    r1 = [data["results"]["R1_start"][m]["avg_mean_energy"] for m in modes]
    r2 = [data["results"]["R2_start"][m]["avg_mean_energy"] for m in modes]
    x = np.arange(len(modes)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, r1, w, label="R1-start", color=C["r1"])
    ax.bar(x + w / 2, r2, w, label="R2-start", color=C["r2"])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Mean energy (P_out)"); ax.set_title("Energy Production by Action Mode")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(OUT / "fig9_energy_by_mode.png", dpi=150); plt.close()


def fig10_adaptation_speed():
    data = json.load(open("results/regime_transition_adaptation_v2.json"))
    regimes = list(data["results"].keys())
    base = [data["results"][r]["baseline"]["mean_return"] for r in regimes]
    cemrl = [data["results"][r]["cemrl"]["mean_return"] for r in regimes]
    speed = [data["results"][r]["cemrl"]["adaptation_speed_episodes"] for r in regimes]
    x = np.arange(len(regimes)); w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(x - w / 2, base, w, label="no-CEMRL baseline", color="#8c564b")
    ax1.bar(x + w / 2, cemrl, w, label="CEMRL", color=C["combined"])
    ax1.set_xticks(x); ax1.set_xticklabels(regimes)
    ax1.set_ylabel("Mean return"); ax1.set_title("CEMRL vs no-CEMRL Baseline")
    ax1.legend(); ax1.grid(alpha=0.3, axis="y")
    ax2.bar(x, speed, color=C["combined"])
    ax2.set_xticks(x); ax2.set_xticklabels(regimes)
    ax2.set_ylabel("Adaptation episodes"); ax2.set_title("CEMRL Adaptation Speed (to 90% baseline)")
    ax2.grid(alpha=0.3, axis="y")
    for i, v in enumerate(speed):
        ax2.text(i, v + 0.1, f"{v}", ha="center")
    plt.tight_layout(); plt.savefig(OUT / "fig10_adaptation_speed.png", dpi=150); plt.close()


def fig11_checkpoint_validation():
    data = json.load(open("results/checkpoint_validation_results.json"))
    results = data["results"]
    eps = sorted(int(k.split("_")[1]) for k in results)
    rewards = []
    energies = []
    for ep in eps:
        ck = results[f"ep_{ep:05d}"]
        r1, r2 = ck["R1_start"], ck["R2_start"]
        rewards.append((r1["avg_mean_reward"] + r2["avg_mean_reward"]) / 2)
        energies.append((r1["avg_mean_energy"] + r2["avg_mean_energy"]) / 2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(eps, rewards, "o-", color=C["sac"], lw=1.5)
    ax1.set_xlabel("Checkpoint episode"); ax1.set_ylabel("Validation reward")
    ax1.set_title("Validation Reward across Checkpoints")
    ax1.grid(alpha=0.3)
    ax2.plot(eps, energies, "o-", color=C["ipae"], lw=1.5)
    ax2.set_xlabel("Checkpoint episode"); ax2.set_ylabel("Validation energy")
    ax2.set_title("Validation Energy across Checkpoints")
    ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "fig11_checkpoint_validation.png", dpi=150); plt.close()


def main():
    rows, f = load_training_log()
    print("Generating figures...")
    fig1_training_reward(rows, f)
    fig2_energy_violations(rows, f)
    fig3_regime_coverage(rows, f)
    fig4_gating_by_regime(rows, f)
    fig5_reward_weights(rows, f)
    fig6_rul_accuracy(rows, f)
    fig7_losses(rows, f)
    fig8_degradation_control()
    fig9_energy_by_mode()
    fig10_adaptation_speed()
    fig11_checkpoint_validation()
    print(f"All figures saved to {OUT}/")


if __name__ == "__main__":
    main()
