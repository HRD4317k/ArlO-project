#!/usr/bin/env python3
"""
Analyze degradation-combination validation results.

Computes, for each checkpoint and each action mode:
  - mean degradation slope (delta rate) and its std (variance)
  - % reduction in degradation rate vs the zero-action baseline
  - mean delta change and std
  - mean reward and std
  - steps to critical

Then produces:
  1. A per-checkpoint ranking of all modes by degradation-rate reduction
  2. An aggregated ranking across all checkpoints (mean reduction + std)
  3. Grouped summary: combined / 3-agent / 2-agent / individual / zero
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def reduction_pct(zero_slope, mode_slope):
    """% reduction in degradation rate vs zero baseline. Positive = slower degradation."""
    if zero_slope is None or mode_slope is None or abs(zero_slope) < 1e-12:
        return None
    return (zero_slope - mode_slope) / abs(zero_slope) * 100.0


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/degradation_combinations_results.json"
    data = load_results(path)

    checkpoints = data["checkpoints"]
    modes = data["modes"]
    results = data["results"]

    # ------------------------------------------------------------------
    # 1. Per-checkpoint table: mode -> (slope_mean, slope_std, reduction%)
    # ------------------------------------------------------------------
    print("=" * 100)
    print("DEGRADATION-RATE REDUCTION vs ZERO-ACTION BASELINE (per checkpoint)")
    print("Positive % = slower degradation (better).  slope = d(delta)/step")
    print("=" * 100)

    # Aggregate across checkpoints for final ranking
    agg_slope = defaultdict(list)      # mode -> list of slopes across ckpts
    agg_reduction = defaultdict(list)  # mode -> list of reduction% across ckpts
    agg_change = defaultdict(list)
    agg_reward = defaultdict(list)

    for ck in checkpoints:
        ck_str = str(ck)
        if ck_str not in results:
            continue
        ck_res = results[ck_str]
        zero_slope = ck_res.get("zero", {}).get("delta_slope_mean")

        print(f"\n--- Checkpoint ep_{ck} ---")
        print(f"{'mode':<22} {'slope':>12} {'slope_std':>12} {'reduction%':>12} {'d_change':>10} {'reward':>10}")

        # Sort modes by reduction (best first)
        mode_rows = []
        for mode in modes:
            if mode not in ck_res or ck_res[mode] is None:
                continue
            m = ck_res[mode]
            slope = m.get("delta_slope_mean")
            slope_std = m.get("delta_slope_std")
            change = m.get("delta_change_mean")
            reward = m.get("mean_reward_mean")
            red = reduction_pct(zero_slope, slope)
            mode_rows.append((mode, slope, slope_std, red, change, reward))

            # Aggregate
            if slope is not None:
                agg_slope[mode].append(slope)
            if red is not None:
                agg_reduction[mode].append(red)
            if change is not None:
                agg_change[mode].append(change)
            if reward is not None:
                agg_reward[mode].append(reward)

        mode_rows.sort(key=lambda r: (r[3] is None, -(r[3] if r[3] is not None else -1e9)))
        for mode, slope, slope_std, red, change, reward in mode_rows:
            red_s = f"{red:+.1f}%" if red is not None else "  N/A"
            print(f"{mode:<22} {slope:+.2e} {slope_std:.1e} {red_s:>12} {change:+.4f} {reward:8.1f}")

    # ------------------------------------------------------------------
    # 2. Aggregated ranking across all checkpoints
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("AGGREGATED RANKING (mean across all checkpoints 2500-4000)")
    print("=" * 100)
    print(f"{'mode':<22} {'mean_slope':>12} {'slope_std':>12} {'mean_reduction%':>14} {'red_std':>10} {'mean_reward':>12}")

    ranked = []
    for mode in modes:
        if mode not in agg_slope or not agg_slope[mode]:
            continue
        mean_slope = float(np.mean(agg_slope[mode]))
        std_slope = float(np.std(agg_slope[mode]))
        mean_red = float(np.mean(agg_reduction[mode])) if agg_reduction[mode] else None
        std_red = float(np.std(agg_reduction[mode])) if agg_reduction[mode] else None
        mean_rew = float(np.mean(agg_reward[mode])) if agg_reward[mode] else None
        ranked.append((mode, mean_slope, std_slope, mean_red, std_red, mean_rew))

    ranked.sort(key=lambda r: (r[3] is None, -(r[3] if r[3] is not None else -1e9)))
    for mode, ms, ss, mr, sr, mw in ranked:
        red_s = f"{mr:+.1f}%" if mr is not None else "  N/A"
        red_std_s = f"{sr:.1f}" if sr is not None else "  N/A"
        rew_s = f"{mw:8.1f}" if mw is not None else "  N/A"
        print(f"{mode:<22} {ms:+.2e} {ss:.1e} {red_s:>14} {red_std_s:>10} {rew_s:>12}")

    # ------------------------------------------------------------------
    # 3. Grouped summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("GROUPED SUMMARY (by combination type)")
    print("=" * 100)

    groups = {
        "zero (baseline)": ["zero"],
        "combined (4-agent)": ["combined"],
        "3-agent combos": [m for m in modes if m.startswith("3agent_")],
        "2-agent combos": [m for m in modes if m.startswith("2agent_")],
        "individual agents": [m for m in modes if m in ("sac", "gp", "fm", "pinn")],
    }

    for gname, gmodes in groups.items():
        gmodes = [m for m in gmodes if m in agg_reduction and agg_reduction[m]]
        if not gmodes:
            continue
        all_red = [r for m in gmodes for r in agg_reduction[m]]
        all_slope = [s for m in gmodes for s in agg_slope[m]]
        print(f"\n{gname}:")
        print(f"  mean reduction vs zero: {np.mean(all_red):+.1f}% ± {np.std(all_red):.1f}%")
        print(f"  mean slope: {np.mean(all_slope):+.2e} ± {np.std(all_slope):.1e}")
        # best mode in group
        best = min(gmodes, key=lambda m: np.mean(agg_slope[m]))
        print(f"  best mode: {best} (slope {np.mean(agg_slope[best]):+.2e})")

    # ------------------------------------------------------------------
    # 4. Best overall combinations
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TOP 5 COMBINATIONS BY DEGRADATION-RATE REDUCTION")
    print("=" * 100)
    top5 = ranked[:5]
    for i, (mode, ms, ss, mr, sr, mw) in enumerate(top5, 1):
        red_s = f"{mr:+.1f}%" if mr is not None else "N/A"
        print(f"  {i}. {mode:<22} reduction={red_s}  slope={ms:+.2e}±{ss:.1e}")


if __name__ == "__main__":
    main()
