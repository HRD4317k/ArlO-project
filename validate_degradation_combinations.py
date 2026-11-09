#!/usr/bin/env python3
"""
Comprehensive Degradation-Effect Validation Across Agent Combinations.

Validates how different combinations of agent actions slow degradation, using
a FRESH evaluation dataset (different seeds than training) and reporting
variance across episodes.

Action modes tested:
  - zero        : zero action (baseline degradation progression)
  - combined    : full 4-agent arbiter-blended policy
  - 3-agent combos : all C(4,3)=4 combinations (arbiter re-normalized over subset)
  - 2-agent combos : all C(4,2)=6 combinations (arbiter re-normalized over subset)
  - individual  : each single agent alone (sac, gp, fm, pinn)

For each checkpoint from ep_02500 to ep_04000 (step 100), for each mode, we run
N_EVAL episodes from fresh random seeds (offset far from training seeds) and
report mean + std (variance) of:
  - delta slope (degradation rate)
  - delta change over episode
  - steps to critical (delta >= 0.9)
  - episode length
  - mean reward
  - improvement/worsening rate

Usage:
    python validate_degradation_combinations.py \
        --ckpt-dir runs/hmarl_baseline_true_delta_resume_5000/checkpoints \
        --start 2500 --end 4000 --step 100 \
        --n-episodes 10 --seed-offset 90000 \
        --out results/degradation_combinations_results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hmarl_drt"))

from hmarl_drt.envs.wind_turbine_env import WindTurbineEnv
from hmarl_drt.agents.sac_agent import SACAgent
from hmarl_drt.agents.pinn_agent import PINNAgent
from hmarl_drt.agents.gp_agent import GPAgent
from hmarl_drt.agents.fm_agent import FMAgent
from hmarl_drt.core.embedding import SharedEmbeddingSpace
from hmarl_drt.core.arbiter import Arbiter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_load(module, state_dict, label: str):
    """Load only compatible tensors (shape-matched) from a checkpoint."""
    if not state_dict:
        return
    current = module.state_dict()
    compatible = {k: v for k, v in state_dict.items()
                  if k in current and tuple(current[k].shape) == tuple(v.shape)}
    skipped = set(state_dict.keys()) - set(compatible.keys())
    module.load_state_dict(compatible, strict=False)
    print(f"    Loaded {label}: {len(compatible)} tensors" +
          (f", skipped {len(skipped)}" if skipped else ""))


def make_env_config(cfg: dict) -> dict:
    """Deterministic env config for validation (true delta visible, no masking)."""
    env_cfg = dict(cfg.get("env", cfg))
    env_cfg["mask_true_delta"] = False
    env_cfg["max_steps"] = int(env_cfg.get("max_steps", 1000))
    env_cfg.setdefault("degradation", {})
    env_cfg["degradation"].setdefault("enabled", True)
    env_cfg["degradation"].setdefault("sample_regime_each_episode", False)
    env_cfg["degradation"].setdefault("regime_delta_ranges", {
        "0": [0.0, 0.25],
        "1": [0.25, 0.50],
        "2": [0.50, 0.75],
        "3": [0.75, 0.95],
    })
    env_cfg["degradation"].setdefault("regime_probs", [0.25, 0.25, 0.25, 0.25])
    return env_cfg


def regime_from_delta(delta: float) -> int:
    if delta < 0.25:
        return 0
    if delta < 0.50:
        return 1
    if delta < 0.75:
        return 2
    return 3


# Agent order used by the arbiter: [NN(SAC), GP, FM, Physics(PINN)]
AGENT_ORDER = ["sac", "gp", "fm", "pinn"]
AGENT_INDEX = {name: i for i, name in enumerate(AGENT_ORDER)}


def build_components(ckpt: dict, device: str, shared_dim: int = 64):
    """Instantiate all agents + embedding + arbiter and load checkpoint weights."""
    sd, ad = 8, 3

    sac = SACAgent(state_dim=sd, shared_dim=shared_dim, action_dim=ad, device=device)
    if "sac_actor" in ckpt:
        safe_load(sac.actor, ckpt["sac_actor"], "SAC actor")
    if "sac_critic" in ckpt:
        safe_load(sac.critic, ckpt["sac_critic"], "SAC critic")

    pinn = PINNAgent(state_dim=sd, action_dim=ad, device=device)
    if "pinn" in ckpt:
        safe_load(pinn.pinn, ckpt["pinn"], "PINN")

    gp = GPAgent(state_dim=sd, action_dim=ad, device=device)
    if "gp_network" in ckpt:
        safe_load(gp.network, ckpt["gp_network"], "GP network")

    fm = FMAgent(state_dim=sd, use_api=False, device=device)

    # Build shared embedding with correct dims
    with torch.no_grad():
        dummy = torch.zeros(1, sac.actor.encoder[0].in_features, device=device)
        dims = [
            sac.actor.get_embedding(dummy).shape[-1],
            gp.get_embedding(dummy).shape[-1],
            fm.get_embedding(dummy).shape[-1],
            pinn.get_embedding(dummy).shape[-1],
        ]
    embedding = SharedEmbeddingSpace(dims, shared_dim=shared_dim, n_agents=4).to(device)
    if "embedding" in ckpt:
        safe_load(embedding, ckpt["embedding"], "shared embedding")

    arbiter = Arbiter(state_dim=sd, n_agents=4, hidden_dim=128, device=device)
    if "arbiter" in ckpt:
        safe_load(arbiter.gating, ckpt["arbiter"], "Arbiter gating")

    return sac, gp, fm, pinn, embedding, arbiter


def compute_z_shared(state_np, sac, gp, fm, pinn, embedding, device):
    s_t = torch.FloatTensor(state_np).unsqueeze(0).to(device)
    with torch.no_grad():
        z1 = sac.actor.get_embedding(s_t)
        z2 = gp.get_embedding(s_t)
        z3 = fm.get_embedding(s_t)
        z4 = pinn.get_embedding(s_t)
        zs, _ = embedding([z1, z2, z3, z4])
    return zs.squeeze(0)


def get_agent_actions(state, sac, gp, fm, pinn, embedding, device):
    """Return dict of agent actions in AGENT_ORDER."""
    z_shared = compute_z_shared(state, sac, gp, fm, pinn, embedding, device)
    return {
        "sac": sac.act(state, z_shared),
        "gp": gp.act(state),
        "fm": fm.act(state),
        "pinn": pinn.act(state),
    }


def get_arbiter_weights(arbiter, state, delta, uncertainty, device):
    """
    Compute the arbiter's gating weights exactly as the training loop does:
    raw gating network output + degradation-aware bias (70% learned / 30% bias).
    Returns weights in AGENT_ORDER: [sac, gp, fm, pinn].
    """
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    delta_t = torch.FloatTensor([[delta]]).to(device)
    uncert_t = torch.FloatTensor([[uncertainty]]).to(device)
    with torch.no_grad():
        weights = arbiter.gating(state_t, delta_t, uncert_t, None).squeeze(0).cpu().numpy()
    weights = arbiter._apply_degradation_bias(weights, delta, uncertainty)
    return weights


def combine_actions(actions: dict, subset: list, arbiter, state, delta, uncertainty,
                    safe_low, safe_high, device):
    """
    Combine actions for a subset of agents using the arbiter's gating weights
    (including degradation bias), re-normalized over the subset.

    For the full 4-agent set, this is the standard arbiter blend.
    For subsets, we take the arbiter's gating weights for the included agents
    and re-normalize them to sum to 1 over the subset.
    """
    weights = get_arbiter_weights(arbiter, state, delta, uncertainty, device)

    # Re-normalize over the subset
    subset_idx = [AGENT_INDEX[name] for name in subset]
    sub_weights = weights[subset_idx]
    total = sub_weights.sum()
    if total < 1e-8:
        sub_weights = np.ones(len(subset)) / len(subset)
    else:
        sub_weights = sub_weights / total

    combined = np.zeros(3, dtype=np.float32)
    for w, name in zip(sub_weights, subset):
        combined += w * np.asarray(actions[name], dtype=np.float32)

    return np.clip(combined, safe_low, safe_high)


def run_episode(env, sac, gp, fm, pinn, embedding, arbiter, shaper, device,
                mode: str, init_delta: float, seed: int, safe_low, safe_high):
    """Run one episode under a given action mode. Returns trajectory metrics."""
    state, _ = env.reset(seed=seed, options={"delta_range": (init_delta, init_delta)})
    done = False
    steps = 0

    deltas = []
    regimes = []
    rewards = []

    while not done:
        delta_curr = float(state[5])
        deltas.append(delta_curr)
        regimes.append(regime_from_delta(delta_curr))

        actions = get_agent_actions(state, sac, gp, fm, pinn, embedding, device)

        # GP epistemic uncertainty (used by arbiter degradation bias)
        try:
            uncertainty = gp.get_uncertainty_np(state)
        except Exception:
            uncertainty = 0.0
        uncertainty_norm = float(np.tanh(uncertainty / 100.0))

        if mode == "zero":
            action = np.zeros(3, dtype=np.float32)
        elif mode == "combined":
            action = combine_actions(actions, AGENT_ORDER, arbiter, state, delta_curr,
                                     uncertainty_norm, safe_low, safe_high, device)
        elif mode.startswith("3agent_"):
            # e.g. 3agent_sac_gp_fm
            subset = mode.split("_")[1:]
            action = combine_actions(actions, subset, arbiter, state, delta_curr,
                                     uncertainty_norm, safe_low, safe_high, device)
        elif mode.startswith("2agent_"):
            subset = mode.split("_")[1:]
            action = combine_actions(actions, subset, arbiter, state, delta_curr,
                                     uncertainty_norm, safe_low, safe_high, device)
        elif mode in AGENT_ORDER:
            # single agent
            action = np.asarray(actions[mode], dtype=np.float32)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        next_state, reward, terminated, truncated, _ = env.step(action)

        # Compute the shaped reward (same as training loop) for a meaningful signal
        try:
            rul_est = gp.estimate_rul(state)
            phys_res = pinn.get_ode_residual_value(
                torch.FloatTensor(state).unsqueeze(0).to(device),
                torch.FloatTensor(action).unsqueeze(0).to(device),
            ).item()
            reward, _ = shaper.compute_reward(
                state, action, next_state,
                rul_estimate=rul_est,
                physics_residual=phys_res,
                uncertainty=uncertainty,
            )
        except Exception:
            reward = float(reward)

        rewards.append(float(reward))
        state = next_state
        done = terminated or truncated
        steps += 1

    return {
        "deltas": np.array(deltas),
        "regimes": np.array(regimes),
        "rewards": np.array(rewards),
        "episode_length": steps,
        "terminated_early": bool(terminated),
    }


def compute_metrics(traj):
    """Compute degradation metrics for one trajectory."""
    deltas = traj["deltas"]
    regimes = traj["regimes"]
    rewards = traj["rewards"]
    n = len(deltas)

    if n > 1:
        x = np.arange(n)
        delta_slope = float(np.polyfit(x, deltas, 1)[0])
    else:
        delta_slope = 0.0

    delta_change = float(deltas[-1] - deltas[0])
    start_regime = int(regimes[0])
    end_regime = int(regimes[-1])

    critical_steps = np.where(deltas >= 0.9)[0]
    steps_to_critical = int(critical_steps[0]) if len(critical_steps) > 0 else None

    return {
        "episode_length": int(n),
        "start_regime": start_regime,
        "end_regime": end_regime,
        "start_delta": float(deltas[0]),
        "end_delta": float(deltas[-1]),
        "delta_change": delta_change,
        "delta_slope": delta_slope,
        "mean_reward": float(np.mean(rewards)) if len(rewards) > 0 else 0.0,
        "steps_to_critical": steps_to_critical,
        "terminated_early": bool(traj["terminated_early"]),
        "improved": int(end_regime < start_regime),
        "worsened": int(end_regime > start_regime),
    }


def aggregate(entries):
    """Aggregate per-episode metrics into mean + std (variance)."""
    if not entries:
        return None
    slopes = [e["delta_slope"] for e in entries]
    changes = [e["delta_change"] for e in entries]
    rewards = [e["mean_reward"] for e in entries]
    lengths = [e["episode_length"] for e in entries]
    stc = [e["steps_to_critical"] for e in entries if e["steps_to_critical"] is not None]

    return {
        "num_episodes": len(entries),
        "delta_slope_mean": float(np.mean(slopes)),
        "delta_slope_std": float(np.std(slopes)),
        "delta_change_mean": float(np.mean(changes)),
        "delta_change_std": float(np.std(changes)),
        "mean_reward_mean": float(np.mean(rewards)),
        "mean_reward_std": float(np.std(rewards)),
        "episode_length_mean": float(np.mean(lengths)),
        "episode_length_std": float(np.std(lengths)),
        "improvement_rate": float(np.mean([e["improved"] for e in entries])),
        "worsening_rate": float(np.mean([e["worsened"] for e in entries])),
        "terminated_early_rate": float(np.mean([e["terminated_early"] for e in entries])),
        "steps_to_critical": {
            "count": len(stc),
            "mean": float(np.mean(stc)) if stc else None,
            "std": float(np.std(stc)) if stc else None,
            "min": int(min(stc)) if stc else None,
            "max": int(max(stc)) if stc else None,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default="runs/hmarl_baseline_true_delta_resume_5000/checkpoints")
    parser.add_argument("--config", default="hmarl_drt/configs/hmarl_baseline_true_delta.yaml")
    parser.add_argument("--start", type=int, default=2500)
    parser.add_argument("--end", type=int, default=4000)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=90000,
                        help="Offset from training seeds to use a fresh eval dataset")
    parser.add_argument("--delta-lo", type=float, default=0.25)
    parser.add_argument("--delta-hi", type=float, default=0.45)
    parser.add_argument("--out", default="results/degradation_combinations_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = yaml.safe_load(open(args.config))
    env_cfg = make_env_config(cfg)
    env = WindTurbineEnv(env_cfg)
    safe_low = env.action_space.low
    safe_high = env.action_space.high

    # Dynamic Reward Shaper (same as training loop) for meaningful reward signal
    from hmarl_drt.core.reward_shaper import DynamicRewardShaper
    rew_cfg = cfg.get("reward", {})
    shaper = DynamicRewardShaper(
        initial_weights=rew_cfg.get("initial_weights"),
        kappa=rew_cfg.get("kappa", 0.1),
        lambda_entropy=rew_cfg.get("lambda_entropy", 0.01),
        alpha_rul=rew_cfg.get("alpha_rul", 0.01),
        phase_mid_step=rew_cfg.get("phase_mid_step", 50000),
        phase_late_step=rew_cfg.get("phase_late_step", 150000),
    )

    # Build all action modes
    modes = ["zero", "combined"]
    # 3-agent combos
    import itertools
    for combo in itertools.combinations(AGENT_ORDER, 3):
        modes.append("3agent_" + "_".join(combo))
    # 2-agent combos
    for combo in itertools.combinations(AGENT_ORDER, 2):
        modes.append("2agent_" + "_".join(combo))
    # individual
    modes += AGENT_ORDER

    # Checkpoint list
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_nums = list(range(args.start, args.end + 1, args.step))
    ckpt_paths = []
    for n in ckpt_nums:
        p = ckpt_dir / f"ep_{n:05d}.pt"
        if p.exists():
            ckpt_paths.append((n, p))
        else:
            print(f"  [WARN] Missing checkpoint: {p}")

    if not ckpt_paths:
        print("No checkpoints found!")
        sys.exit(1)

    print(f"Checkpoints to evaluate: {[n for n, _ in ckpt_paths]}")
    print(f"Modes: {modes}")
    print(f"Episodes per mode: {args.n_episodes} (fresh seeds offset {args.seed_offset})")
    print(f"Initial delta range: [{args.delta_lo}, {args.delta_hi}]")

    all_results = {}

    for ckpt_num, ckpt_path in ckpt_paths:
        print(f"\n{'='*70}")
        print(f"CHECKPOINT ep_{ckpt_num}")
        print(f"{'='*70}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sac, gp, fm, pinn, embedding, arbiter = build_components(ckpt, device)

        ckpt_results = {}
        for mode in modes:
            entries = []
            for ep in range(args.n_episodes):
                # Fresh eval dataset: seed far from training seeds
                seed = args.seed_offset + ckpt_num + ep * 1000
                init_delta = np.random.default_rng(seed).uniform(args.delta_lo, args.delta_hi)
                traj = run_episode(env, sac, gp, fm, pinn, embedding, arbiter, shaper, device,
                                   mode, init_delta, seed, safe_low, safe_high)
                entries.append(compute_metrics(traj))
            agg = aggregate(entries)
            ckpt_results[mode] = agg
            if agg:
                print(f"  {mode:>22s}: slope={agg['delta_slope_mean']:+.2e}±{agg['delta_slope_std']:.1e}  "
                      f"Δchange={agg['delta_change_mean']:+.4f}±{agg['delta_change_std']:.4f}  "
                      f"steps2crit={agg['steps_to_critical']['mean'] if agg['steps_to_critical']['mean'] else 'NA'}  "
                      f"reward={agg['mean_reward_mean']:.3f}")

        all_results[str(ckpt_num)] = ckpt_results

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint_dir": str(ckpt_dir),
        "checkpoints": [n for n, _ in ckpt_paths],
        "modes": modes,
        "n_episodes_per_mode": args.n_episodes,
        "seed_offset": args.seed_offset,
        "delta_range": [args.delta_lo, args.delta_hi],
        "results": all_results,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
