#!/usr/bin/env python3
"""
Validate the effect of the arbiter's blended action on RUL and degradation index,
specifically focusing on whether the combined action can *prevent further degradation*
when the system is already in regime 1 (degrading, Δ ∈ [0.25, 0.50)) or
regime 2 (critical, Δ ∈ [0.50, 0.75)).

The script:
1. Loads the checkpoint (default 2500‑episode, can be overridden via env var).
2. Resets the environment with a *targeted* initial Δ in R1 or R2.
3. Runs each action mode (blended, zero, sac, pinn, gp, fm) from the same start state.
4. Records the Δ trajectory, regime, reward and gating weights.
5. Computes summary metrics:
   - Δ slope (linear fit) and Δ acceleration.
   - Fraction of steps spent in each regime.
   - Whether the system improved or worsened.
   - Steps to critical (Δ ≥ 0.9) if reached.
   - Mean reward.

The results are saved as JSON to ``results/degradation_effect_validation.json``.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import yaml
from hmarl_drt.core.embedding import SharedEmbeddingSpace

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hmarl_drt"))

from hmarl_drt.envs.wind_turbine_env import WindTurbineEnv
from hmarl_drt.agents.sac_agent import SACAgent
from hmarl_drt.agents.pinn_agent import PINNAgent
from hmarl_drt.agents.gp_agent import GPAgent
from hmarl_drt.agents.fm_agent import FMAgent
from hmarl_drt.core.arbiter import Arbiter


def safe_load(module, state_dict, label: str):
    if not state_dict:
        return
    current = module.state_dict()
    compatible = {k: v for k, v in state_dict.items()
                  if k in current and tuple(current[k].shape) == tuple(v.shape)}
    skipped = set(state_dict.keys()) - set(compatible.keys())
    module.load_state_dict(compatible, strict=False)
    print(f"  Loaded {label}: {len(compatible)} tensors" +
          (f", skipped {len(skipped)}" if skipped else ""))


def make_env_config(cfg: dict) -> dict:
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


def run_episode(env, sac, gp, fm, pinn, arbiter, embedding, compute_z_shared,
                safe_low, safe_high, device, action_mode="blended", init_delta=None):
    """Run a single episode and collect per‑step data."""
    if init_delta is not None:
        state, _ = env.reset(options={"delta_range": (init_delta, init_delta)})
    else:
        state, _ = env.reset()
    done = False
    steps = 0

    deltas = []
    regimes = []
    rewards = []
    weights_list = []

    while not done:
        delta_curr = float(state[5])
        regime_curr = regime_from_delta(delta_curr)
        deltas.append(delta_curr)
        regimes.append(regime_curr)

        # Compute actions
        z_shared = compute_z_shared(state)
        a_sac = sac.act(state, z_shared)
        a_pinn = pinn.act(state)
        a_gp = gp.act(state)
        a_fm = fm.act(state)

        if action_mode == "blended":
            action_proposals = {"fm": a_fm, "sac": a_sac, "pinn": a_pinn, "gp": a_gp}
            action, weights = arbiter.select_action(
                state, action_proposals,
                {"low": safe_low, "high": safe_high},
                delta=delta_curr, return_weights=True,
            )
            weights_list.append(weights.tolist())
        elif action_mode == "zero":
            action = np.zeros_like(a_sac)
            weights_list.append([0.25, 0.25, 0.25, 0.25])
        elif action_mode == "sac":
            action = a_sac
            weights_list.append([1.0, 0.0, 0.0, 0.0])
        elif action_mode == "pinn":
            action = a_pinn
            weights_list.append([0.0, 0.0, 0.0, 1.0])
        elif action_mode == "gp":
            action = a_gp
            weights_list.append([0.0, 1.0, 0.0, 0.0])
        elif action_mode == "fm":
            action = a_fm
            weights_list.append([0.0, 0.0, 1.0, 0.0])
        else:
            raise ValueError(f"Unknown action_mode: {action_mode}")

        next_state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        state = next_state
        done = terminated or truncated
        steps += 1

    return {
        "deltas": np.array(deltas),
        "regimes": np.array(regimes),
        "rewards": np.array(rewards),
        "weights": np.array(weights_list) if weights_list else None,
        "episode_length": steps,
        "terminated_early": bool(terminated),
    }


def compute_metrics(traj, label):
    """Summarize a trajectory.
    Returns steps_to_critical (first step where Δ ≥ 0.9) instead of GP‑based RUL.
    """
    deltas = traj["deltas"]
    regimes = traj["regimes"]
    rewards = traj["rewards"]
    n = len(deltas)

    if n > 1:
        x = np.arange(n)
        delta_slope = np.polyfit(x, deltas, 1)[0]
    else:
        delta_slope = 0.0

    delta_change = deltas[-1] - deltas[0]

    regime_fractions = {f"R{r}": float((regimes == r).mean()) for r in range(4)}

    start_regime = regimes[0]
    end_regime = regimes[-1]
    improved = int(end_regime < start_regime)
    worsened = int(end_regime > start_regime)

    critical_steps = np.where(deltas >= 0.9)[0]
    steps_to_critical = int(critical_steps[0]) if len(critical_steps) > 0 else None

    mean_reward = float(rewards.mean()) if len(rewards) > 0 else 0.0

    if n > 2:
        delta_accel = float(np.mean(np.diff(deltas, 2)))
    else:
        delta_accel = 0.0

    return {
        "label": label,
        "episode_length": int(n),
        "start_regime": int(start_regime),
        "end_regime": int(end_regime),
        "start_delta": float(deltas[0]),
        "end_delta": float(deltas[-1]),
        "delta_change": float(delta_change),
        "delta_slope": float(delta_slope),
        "delta_accel": delta_accel,
        "mean_reward": mean_reward,
        "regime_fractions": regime_fractions,
        "improved": improved,
        "worsened": worsened,
        "steps_to_critical": steps_to_critical,
        "terminated_early": traj["terminated_early"],
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg_path = os.environ.get("HMARL_VALIDATION_CONFIG", "configs/wind_config.yaml")
    ckpt_path = os.environ.get(
        "HMARL_VALIDATION_CHECKPOINT",
        "runs/archive_pre_fresh/checkpoints/ep_02500.pt",
    )
    cfg = yaml.safe_load(open(cfg_path))
    env_cfg = make_env_config(cfg)
    ckpt = torch.load(ckpt_path, map_location=device) if Path(ckpt_path).exists() else {}
    if not ckpt:
        print(f"  Warning: checkpoint not found at {ckpt_path}; using random init")

    env = WindTurbineEnv(env_cfg)
    sd = int(env.observation_space.shape[0])
    ad = int(env.action_space.shape[0])
    shared_dim = cfg.get("embedding", {}).get("shared_dim", 64)

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

    def build_shared_embedding():
        """Create the shared latent space used by the agents."""
        with torch.no_grad():
            dummy = torch.zeros(1, sac.actor.encoder[0].in_features, device=device)
            dims = [
                sac.actor.get_embedding(dummy).shape[-1],
                gp.get_embedding(dummy).shape[-1],
                fm.get_embedding(dummy).shape[-1],
                pinn.get_embedding(dummy).shape[-1],
            ]
        emb = SharedEmbeddingSpace(dims, shared_dim=shared_dim, n_agents=4).to(device)
        if "embedding" in ckpt:
            safe_load(emb, ckpt["embedding"], "shared embedding")
        return emb

    embedding = build_shared_embedding()

    def compute_z_shared(state_np):
        s_t = torch.FloatTensor(state_np).unsqueeze(0).to(device)
        with torch.no_grad():
            z1 = sac.actor.get_embedding(s_t)
            z2 = gp.get_embedding(s_t)
            z3 = fm.get_embedding(s_t)
            z4 = pinn.get_embedding(s_t)
            zs, _ = embedding([z1, z2, z3, z4])
        return zs.squeeze(0)

    arbiter = Arbiter(state_dim=sd, n_agents=4, hidden_dim=128, device=device)
    if "arbiter" in ckpt:
        safe_load(arbiter.gating, ckpt["arbiter"], "Arbiter gating")

    safe_low = env.action_space.low
    safe_high = env.action_space.high

    N_EVAL = int(os.environ.get("HMARL_VALIDATION_EPISODES", 20))
    delta_ranges = {
        "R1_start": (0.25, 0.45),
        "R2_start": (0.50, 0.70),
    }
    action_modes = ["blended", "zero", "sac", "pinn", "gp", "fm"]
    all_results = {}

    for range_label, (delta_lo, delta_hi) in delta_ranges.items():
        print(f"\n{'='*60}")
        print(f"Starting regime: {range_label}  (Δ ∈ [{delta_lo}, {delta_hi}])")
        print(f"{'='*60}")
        range_results = defaultdict(list)
        for ep in range(N_EVAL):
            init_delta = np.random.uniform(delta_lo, delta_hi)
            for mode in action_modes:
                traj = run_episode(
                    env, sac, gp, fm, pinn, arbiter, embedding,
                    compute_z_shared, safe_low, safe_high, device,
                    action_mode=mode, init_delta=init_delta,
                )
                metrics = compute_metrics(traj, mode)
                range_results[mode].append(metrics)
        aggregated = {}
        for mode in action_modes:
            entries = range_results[mode]
            if not entries:
                continue
            agg = {
                "label": mode,
                "num_episodes": len(entries),
                "avg_episode_length": float(np.mean([e["episode_length"] for e in entries])),
                "avg_delta_change": float(np.mean([e["delta_change"] for e in entries])),
                "avg_delta_slope": float(np.mean([e["delta_slope"] for e in entries])),
                "avg_delta_accel": float(np.mean([e["delta_accel"] for e in entries])),
                # RUL change omitted – use steps_to_critical for degradation impact
                "avg_mean_reward": float(np.mean([e["mean_reward"] for e in entries])),
                "improvement_rate": float(np.mean([e["improved"] for e in entries])),
                "worsening_rate": float(np.mean([e["worsened"] for e in entries])),
                "avg_regime_fractions": {},
                "terminated_early_rate": float(np.mean([e["terminated_early"] for e in entries])),
                "steps_to_critical": [],
            }
            for r in range(4):
                vals = [e["regime_fractions"][f"R{r}"] for e in entries]
                agg["avg_regime_fractions"][f"R{r}"] = float(np.mean(vals))
            stc = [e["steps_to_critical"] for e in entries if e["steps_to_critical"] is not None]
            agg["steps_to_critical"] = {
                "count": len(stc),
                "mean_steps": float(np.mean(stc)) if stc else None,
                "min_steps": int(min(stc)) if stc else None,
                "max_steps": int(max(stc)) if stc else None,
            }
            aggregated[mode] = agg
            print(f"  {mode:>8s}: Δ_change={agg['avg_delta_change']:+.6f}  "
                  f"Δ_slope={agg['avg_delta_slope']:+.2e}  "
                  f"improve={agg['improvement_rate']:.2f}  "
                  f"worsen={agg['worsening_rate']:.2f}  "
                  f"reward={agg['avg_mean_reward']:.4f}")
        all_results[range_label] = aggregated

    summary = {
        "checkpoint": ckpt_path,
        "num_episodes_per_range": N_EVAL,
        "results": all_results,
    }
    print("\n\n=== DEGRADATION EFFECT VALIDATION SUMMARY ===")
    print(json.dumps(summary, indent=2))
    out_path = Path("results/degradation_effect_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
