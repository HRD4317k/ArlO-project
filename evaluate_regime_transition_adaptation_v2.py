#!/usr/bin/env python3
"""
Rigorous Regime-Transition Adaptation-Speed Evaluation.

Baseline: HMARL system WITHOUT the CEMRL stack, trained individually on each
regime (single-regime specialisation). This gives the per-regime baseline
performance (rolling-average return).

CEMRL test: the trained CEMRL system (ep_2300) is run in each target regime.
Adaptation speed = number of episodes until the CEMRL system's ROLLING-AVERAGE
return reaches 90% of that regime's baseline (set by the no-CEMRL training).

This avoids single-episode noise by using rolling averages, and gives a
meaningful absolute target (90% of the regime-specialised baseline).

Output: results/regime_transition_adaptation_v2.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hmarl_drt"))

from hmarl_drt.envs.wind_turbine_env import WindTurbineEnv
from hmarl_drt.agents.sac_agent import SACAgent
from hmarl_drt.agents.pinn_agent import PINNAgent
from hmarl_drt.agents.ipae_agent import IPAEAgent
from hmarl_drt.agents.fm_agent import FMAgent
from hmarl_drt.core.embedding import SharedEmbeddingSpace
from hmarl_drt.core.arbiter import Arbiter
from hmarl_drt.core.cemrl_arbiter import CEMRLArbiter


def safe_load(module, state_dict, label: str):
    if not state_dict:
        return
    current = module.state_dict()
    compatible = {k: v for k, v in state_dict.items()
                  if k in current and tuple(current[k].shape) == tuple(v.shape)}
    module.load_state_dict(compatible, strict=False)
    print(f"  Loaded {label}: {len(compatible)} tensors")


def build_supervised_z_task(regime_id, delta, n_regimes=4, z_dim=32, device="cpu"):
    z = torch.zeros(z_dim, device=device)
    if regime_id is not None and 0 <= regime_id < min(n_regimes, z_dim):
        z[regime_id] = 1.0
    delta_slot = min(n_regimes, z_dim - 1)
    z[delta_slot] = float(delta)
    return z


def safety_urgency(state, omega_min, omega_max, T_max, margin_frac=0.15):
    omega = float(state[0]) if len(state) > 0 else 1.0
    T_bearing = float(state[3]) if len(state) > 3 else 50.0
    center = 0.5 * (omega_min + omega_max)
    half = 0.5 * (omega_max - omega_min)
    om_prox = abs(omega - center) / half if half > 0 else 0.0
    om_urgency = np.clip((om_prox - (1.0 - margin_frac)) / margin_frac, 0.0, 1.0)
    T_urgency = np.clip((T_bearing - (1.0 - margin_frac) * T_max) / (margin_frac * T_max), 0.0, 1.0)
    return float(max(om_urgency, T_urgency))


def make_env_for_regime(cfg, regime_id, max_steps=1000):
    """Create an env forced into a single regime."""
    env_cfg = dict(cfg.get("env", cfg))
    env_cfg["mask_true_delta"] = False
    env_cfg["max_steps"] = int(max_steps)
    env_cfg.setdefault("degradation", {})
    env_cfg["degradation"].setdefault("enabled", True)
    env_cfg["degradation"].setdefault("sample_regime_each_episode", False)
    n_regimes = len(env_cfg["degradation"].get("regime_probs", [0.25, 0.25, 0.25, 0.25]))
    one_hot = [0.0] * n_regimes
    one_hot[regime_id] = 1.0
    env_cfg["degradation"]["regime_probs"] = one_hot
    env_cfg["degradation"]["regime_delta_ranges"] = {
        "0": [0.0, 0.25], "1": [0.25, 0.50], "2": [0.50, 0.75], "3": [0.75, 0.95],
    }
    return WindTurbineEnv(env_cfg)


def run_episode_return_cemrl(env, sac, ipae, fm, pinn, arbiter, embedding, compute_z_shared,
                             safe_low, safe_high, device, z_task_dim=32,
                             omega_min=0.5, omega_max=1.5, T_max=100.0,
                             warning_caution=0.5, safety_margin_frac=0.15):
    """Run one episode with the CEMRL system (adaptive gating)."""
    state, _ = env.reset()
    if hasattr(ipae, "reset_window"):
        ipae.reset_window()
    done = False
    total_reward = 0.0
    while not done:
        delta_curr = float(state[5])
        regime_curr = 0 if delta_curr < 0.25 else (1 if delta_curr < 0.5 else (2 if delta_curr < 0.75 else 3))
        z_shared = compute_z_shared(state)
        a_sac = sac.act(state, z_shared)
        a_ipae = ipae.act(state)
        a_fm = fm.act(state)
        a_pinn = pinn.policy_act(state)
        uncertainty = ipae.get_uncertainty_np(state)
        fm_severity, _ = fm.assess(state)
        is_anomaly = fm_severity == "critical"
        urgency = safety_urgency(state, omega_min, omega_max, T_max, safety_margin_frac)
        caution = max(warning_caution if fm_severity == "warning" else 0.0, urgency)
        z_task = build_supervised_z_task(regime_curr, delta_curr, z_dim=z_task_dim, device=device)
        proposals = {"nn": a_sac, "gp": a_ipae, "fm": a_fm, "physics": a_pinn}
        action, _ = arbiter.select_action(
            state, proposals, {"low": safe_low, "high": safe_high},
            z_task=z_task, delta=delta_curr, uncertainty=uncertainty,
            emergency=is_anomaly, caution=caution, return_weights=True,
        )
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        state = next_state
        done = terminated or truncated
    return total_reward


def run_episode_return_nocemrl(env, sac, ipae, fm, pinn, arbiter, embedding, compute_z_shared,
                               safe_low, safe_high, device):
    """Run one episode with the no-CEMRL system (classic Arbiter)."""
    state, _ = env.reset()
    if hasattr(ipae, "reset_window"):
        ipae.reset_window()
    done = False
    total_reward = 0.0
    while not done:
        delta_curr = float(state[5])
        z_shared = compute_z_shared(state)
        a_sac = sac.act(state, z_shared)
        a_ipae = ipae.act(state)
        a_fm = fm.act(state)
        a_pinn = pinn.policy_act(state)
        uncertainty = ipae.get_uncertainty_np(state)
        proposals = {"nn": a_sac, "gp": a_ipae, "fm": a_fm, "physics": a_pinn}
        action, _ = arbiter.select_action(
            state, proposals, {"low": safe_low, "high": safe_high},
            delta=delta_curr, uncertainty=uncertainty, return_weights=True,
        )
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        state = next_state
        done = terminated or truncated
    return total_reward


def rolling_mean(returns, window):
    """Rolling mean of returns with the given window."""
    out = []
    for i in range(len(returns)):
        lo = max(0, i - window + 1)
        out.append(float(np.mean(returns[lo:i + 1])))
    return out


def episodes_to_reach_target(returns, target, window):
    """Episodes until the rolling-average return reaches target."""
    rm = rolling_mean(returns, window)
    for i, v in enumerate(rm, start=1):
        if v >= target:
            return i
    return len(returns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="hmarl_drt/runs/fresh_ipae_supervised/checkpoints/ep_02300.pt",
                        help="CEMRL checkpoint to evaluate")
    parser.add_argument("--config", default="hmarl_drt/configs/wind_fresh_ipae_supervised.yaml")
    parser.add_argument("--baseline-dir", default="hmarl_drt/runs/baseline_nocemrl_R{regime}/checkpoints",
                        help="Template for per-regime no-CEMRL baseline checkpoint dirs")
    parser.add_argument("--baseline-episode", type=int, default=300,
                        help="Which baseline checkpoint episode to load (final)")
    parser.add_argument("--max-episodes", type=int, default=30,
                        help="Max episodes for CEMRL adaptation measurement")
    parser.add_argument("--target-fraction", type=float, default=0.90,
                        help="Fraction of baseline return to reach")
    parser.add_argument("--window", type=int, default=5,
                        help="Rolling-average window")
    parser.add_argument("--out", default="results/regime_transition_adaptation_v2.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = yaml.safe_load(open(args.config))
    sd = 8
    ad = 3
    shared_dim = cfg.get("embedding", {}).get("shared_dim", 64)
    ipae_cfg = cfg.get("ipae", {})
    z_task_dim = cfg.get("cemrl", {}).get("z_task_dim", 32)
    constr = cfg.get("env", {}).get("constraints", {})
    omega_min = float(constr.get("omega_min", 0.5))
    omega_max = float(constr.get("omega_max", 1.5))
    T_max = float(constr.get("T_max", 100.0))
    warning_caution = float(cfg.get("arbiter", {}).get("warning_caution", 0.5))
    safety_margin_frac = float(cfg.get("arbiter", {}).get("safety_margin_frac", 0.15))

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    # ---- Build shared components (used by both CEMRL and no-CEMRL) ----
    sac = SACAgent(state_dim=sd, shared_dim=shared_dim, action_dim=ad,
                   hidden_dim=cfg["agents"]["sac"]["hidden_dim"], device=device)
    safe_load(sac.actor, ckpt.get("sac_actor"), "SAC actor")
    safe_load(sac.critic, ckpt.get("sac_critic"), "SAC critic")

    pinn = PINNAgent(state_dim=sd, action_dim=ad,
                     J=float(cfg["env"]["physics"]["J"]), B=float(cfg["env"]["physics"]["B"]),
                     R=float(cfg["env"]["physics"]["R"]),
                     omega_min=omega_min, omega_max=omega_max, T_max=T_max,
                     hidden_dim=cfg["agents"]["pinn"]["hidden_dim"], device=device)
    safe_load(pinn.pinn, ckpt.get("pinn"), "PINN")
    safe_load(pinn.policy, ckpt.get("pinn_policy"), "PINN policy")

    ipae = IPAEAgent(state_dim=sd, action_dim=ad,
                     latent_dim=ipae_cfg.get("latent_dim", 8),
                     window_size=ipae_cfg.get("window_size", 10),
                     encoder_hidden=ipae_cfg.get("encoder_hidden", 64),
                     lstm_hidden=ipae_cfg.get("lstm_hidden", 64),
                     lstm_layers=ipae_cfg.get("lstm_layers", 1),
                     rul_hidden=ipae_cfg.get("rul_hidden", [64, 32, 16]),
                     mc_dropout=ipae_cfg.get("mc_dropout", 0.1),
                     device=device)
    safe_load(ipae.network, ckpt.get("gp_network"), "iPAE network")

    fm = FMAgent(state_dim=sd, use_api=False, device=device,
                 omega_min=omega_min, omega_max=omega_max,
                 T_max=T_max, delta_crit=float(constr.get("delta_crit", 0.9)))

    def build_shared_embedding():
        with torch.no_grad():
            dummy = torch.zeros(1, sd, device=device)
            dims = [sac.actor.get_embedding(dummy).shape[-1],
                    ipae.get_embedding(dummy).shape[-1],
                    fm.get_embedding(dummy).shape[-1],
                    pinn.get_embedding(dummy).shape[-1]]
        emb = SharedEmbeddingSpace(dims, shared_dim=shared_dim, n_agents=4).to(device)
        safe_load(emb, ckpt.get("embedding"), "shared embedding")
        return emb

    embedding = build_shared_embedding()

    def compute_z_shared(state_np):
        s_t = torch.FloatTensor(state_np).unsqueeze(0).to(device)
        with torch.no_grad():
            z1 = sac.actor.get_embedding(s_t)
            z2 = ipae.get_embedding(s_t)
            z3 = fm.get_embedding(s_t)
            z4 = pinn.get_embedding(s_t)
            zs, _ = embedding([z1, z2, z3, z4])
        return zs.squeeze(0)

    # CEMRL arbiter (from checkpoint)
    cemrl_arbiter = CEMRLArbiter(state_dim=sd, z_task_dim=z_task_dim, n_agents=4,
                                 hidden_dim=cfg["cemrl"]["cemrl_arbiter"]["hidden_dim"],
                                 device=device)
    safe_load(cemrl_arbiter.gating, ckpt.get("cemrl_arbiter"), "CEMRL arbiter gating")

    safe_low = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    safe_high = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # ---- Phase 1: Load trained no-CEMRL baselines per regime ----
    print("\n" + "=" * 70)
    print("PHASE 1: Load trained no-CEMRL baselines (single-regime specialisation)")
    print("=" * 70)
    baseline_returns = {}
    for regime in range(4):
        # Load the trained no-CEMRL checkpoint for this regime
        base_dir = args.baseline_dir.format(regime=regime)
        base_ckpt_path = Path(base_dir) / f"ep_{args.baseline_episode:05d}.pt"
        if not base_ckpt_path.exists():
            # fall back to final_model.pt
            base_ckpt_path = Path(base_dir) / "final_model.pt"
        print(f"\n  Loading no-CEMRL baseline R{regime} from {base_ckpt_path}")
        if not base_ckpt_path.exists():
            print(f"  WARNING: baseline checkpoint not found for R{regime}, skipping")
            continue
        base_ckpt = torch.load(base_ckpt_path, map_location=device, weights_only=False)

        # Build a fresh no-CEMRL system from the baseline checkpoint
        b_sac = SACAgent(state_dim=sd, shared_dim=shared_dim, action_dim=ad,
                         hidden_dim=cfg["agents"]["sac"]["hidden_dim"], device=device)
        safe_load(b_sac.actor, base_ckpt.get("sac_actor"), "baseline SAC actor")
        safe_load(b_sac.critic, base_ckpt.get("sac_critic"), "baseline SAC critic")

        b_pinn = PINNAgent(state_dim=sd, action_dim=ad,
                           J=float(cfg["env"]["physics"]["J"]), B=float(cfg["env"]["physics"]["B"]),
                           R=float(cfg["env"]["physics"]["R"]),
                           omega_min=omega_min, omega_max=omega_max, T_max=T_max,
                           hidden_dim=cfg["agents"]["pinn"]["hidden_dim"], device=device)
        safe_load(b_pinn.pinn, base_ckpt.get("pinn"), "baseline PINN")
        safe_load(b_pinn.policy, base_ckpt.get("pinn_policy"), "baseline PINN policy")

        b_ipae = IPAEAgent(state_dim=sd, action_dim=ad,
                           latent_dim=ipae_cfg.get("latent_dim", 8),
                           window_size=ipae_cfg.get("window_size", 10),
                           encoder_hidden=ipae_cfg.get("encoder_hidden", 64),
                           lstm_hidden=ipae_cfg.get("lstm_hidden", 64),
                           lstm_layers=ipae_cfg.get("lstm_layers", 1),
                           rul_hidden=ipae_cfg.get("rul_hidden", [64, 32, 16]),
                           mc_dropout=ipae_cfg.get("mc_dropout", 0.1),
                           device=device)
        safe_load(b_ipae.network, base_ckpt.get("gp_network"), "baseline iPAE network")

        b_fm = FMAgent(state_dim=sd, use_api=False, device=device,
                       omega_min=omega_min, omega_max=omega_max,
                       T_max=T_max, delta_crit=float(constr.get("delta_crit", 0.9)))

        def build_baseline_embedding():
            with torch.no_grad():
                dummy = torch.zeros(1, sd, device=device)
                dims = [b_sac.actor.get_embedding(dummy).shape[-1],
                        b_ipae.get_embedding(dummy).shape[-1],
                        b_fm.get_embedding(dummy).shape[-1],
                        b_pinn.get_embedding(dummy).shape[-1]]
            emb = SharedEmbeddingSpace(dims, shared_dim=shared_dim, n_agents=4).to(device)
            safe_load(emb, base_ckpt.get("embedding"), "baseline shared embedding")
            return emb

        b_embedding = build_baseline_embedding()

        def compute_baseline_z_shared(state_np):
            s_t = torch.FloatTensor(state_np).unsqueeze(0).to(device)
            with torch.no_grad():
                z1 = b_sac.actor.get_embedding(s_t)
                z2 = b_ipae.get_embedding(s_t)
                z3 = b_fm.get_embedding(s_t)
                z4 = b_pinn.get_embedding(s_t)
                zs, _ = b_embedding([z1, z2, z3, z4])
            return zs.squeeze(0)

        # No-CEMRL classic arbiter
        b_arbiter = Arbiter(state_dim=sd, n_agents=4, hidden_dim=128, device=device)
        if "arbiter" in base_ckpt:
            safe_load(b_arbiter.gating, base_ckpt["arbiter"], "baseline arbiter gating")

        # Evaluate the trained baseline on its own regime
        env = make_env_for_regime(cfg, regime)
        print(f"  Evaluating trained no-CEMRL R{regime} baseline...")
        returns = []
        for ep in range(args.max_episodes):
            r = run_episode_return_nocemrl(env, b_sac, b_ipae, b_fm, b_pinn, b_arbiter,
                                           b_embedding, compute_baseline_z_shared,
                                           safe_low, safe_high, device)
            returns.append(r)
            if (ep + 1) % 5 == 0:
                print(f"    ep {ep+1}: return={r:.1f} (rolling-{args.window} mean={np.mean(returns[-args.window:]):.1f})")
        baseline_returns[regime] = returns
        print(f"  R{regime} baseline: mean={np.mean(returns):.1f}, "
              f"rolling-{args.window} final={np.mean(returns[-args.window:]):.1f}")

    # ---- Phase 2: CEMRL adaptation in each regime ----
    print("\n" + "=" * 70)
    print("PHASE 2: CEMRL adaptation speed in each regime")
    print("=" * 70)
    results = {}
    for regime in range(4):
        if regime not in baseline_returns:
            continue
        env = make_env_for_regime(cfg, regime)
        baseline_ret = baseline_returns[regime]
        baseline_target = args.target_fraction * np.mean(baseline_ret[-args.window:])
        print(f"\n  CEMRL in R{regime}: target = {args.target_fraction:.0%} of baseline "
              f"({baseline_target:.1f})")
        returns = []
        for ep in range(args.max_episodes):
            r = run_episode_return_cemrl(env, sac, ipae, fm, pinn, cemrl_arbiter,
                                         embedding, compute_z_shared, safe_low, safe_high, device,
                                         z_task_dim, omega_min, omega_max, T_max,
                                         warning_caution, safety_margin_frac)
            returns.append(r)
            if (ep + 1) % 5 == 0:
                print(f"    ep {ep+1}: return={r:.1f} (rolling-{args.window} mean={np.mean(returns[-args.window:]):.1f})")

        speed = episodes_to_reach_target(returns, baseline_target, args.window)
        rm = rolling_mean(returns, args.window)
        print(f"  CEMRL R{regime}: adaptation speed = {speed} episodes "
              f"(rolling-{args.window} mean reached {baseline_target:.1f})")

        results[f"R{regime}"] = {
            "baseline": {
                "mean_return": float(np.mean(baseline_ret)),
                "rolling_final": float(np.mean(baseline_ret[-args.window:])),
                "target": float(baseline_target),
            },
            "cemrl": {
                "returns": returns,
                "mean_return": float(np.mean(returns)),
                "rolling_final": float(np.mean(returns[-args.window:])),
                "adaptation_speed_episodes": speed,
                "reached_target": bool(speed < args.max_episodes),
            },
            "ratio_cemrl_baseline": float(np.mean(returns) / np.mean(baseline_ret)),
        }

    summary = {
        "checkpoint": args.ckpt,
        "baseline_dir": args.baseline_dir,
        "max_episodes": args.max_episodes,
        "target_fraction": args.target_fraction,
        "window": args.window,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
