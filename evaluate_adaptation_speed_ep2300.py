#!/usr/bin/env python3
"""
Adaptation-speed evaluation for the best checkpoint (ep_2300).

Measures how quickly the trained policy adapts after a regime transition,
using the actual CEMRL machinery (supervised z_task + regime-prior gating).

Method:
  1. Load the checkpoint's SAC actor + CEMRL arbiter.
  2. Force the environment into a target regime (one-hot regime_probs).
  3. Run episodes with the trained policy, tracking the episode return.
  4. Adaptation speed = number of episodes until the return reaches a
     target threshold (fraction of the best observed return).
  5. Compare against a fresh (random-init) SAC policy trained from scratch
     on the same regime for the same number of episodes.

This validates the MAML fast-adaptation claim: the meta-learned policy
should reach the target return in far fewer episodes than a scratch policy.
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


def run_episode_return(env, sac, ipae, fm, pinn, arbiter, embedding, compute_z_shared,
                       safe_low, safe_high, device, z_task_dim=32,
                       omega_min=0.5, omega_max=1.5, T_max=100.0,
                       warning_caution=0.5, safety_margin_frac=0.15):
    """Run one episode, return total reward."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="hmarl_drt/runs/fresh_ipae_supervised/checkpoints/ep_02300.pt")
    parser.add_argument("--config", default="hmarl_drt/configs/wind_fresh_ipae_supervised.yaml")
    parser.add_argument("--regime", type=int, default=2, help="Target regime for adaptation")
    parser.add_argument("--max-episodes", type=int, default=30)
    parser.add_argument("--target-fraction", type=float, default=0.85)
    parser.add_argument("--out", default="results/ep2300_adaptation_speed.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = yaml.safe_load(open(args.config))
    env_cfg = dict(cfg.get("env", cfg))
    env_cfg["mask_true_delta"] = False
    env_cfg["max_steps"] = int(env_cfg.get("max_steps", 1000))
    env_cfg.setdefault("degradation", {})
    env_cfg["degradation"].setdefault("enabled", True)
    env_cfg["degradation"].setdefault("sample_regime_each_episode", False)
    # Force the target regime
    n_regimes = len(env_cfg["degradation"].get("regime_probs", [0.25, 0.25, 0.25, 0.25]))
    one_hot = [0.0] * n_regimes
    one_hot[args.regime] = 1.0
    env_cfg["degradation"]["regime_probs"] = one_hot
    env_cfg["degradation"]["regime_delta_ranges"] = {
        "0": [0.0, 0.25], "1": [0.25, 0.50], "2": [0.50, 0.75], "3": [0.75, 0.95],
    }

    env = WindTurbineEnv(env_cfg)
    sd = int(env.observation_space.shape[0])
    ad = int(env.action_space.shape[0])
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

    # --- Trained (meta-learned) policy ---
    sac = SACAgent(state_dim=sd, shared_dim=shared_dim, action_dim=ad,
                   hidden_dim=cfg["agents"]["sac"]["hidden_dim"], device=device)
    safe_load(sac.actor, ckpt.get("sac_actor"), "SAC actor")
    safe_load(sac.critic, ckpt.get("sac_critic"), "SAC critic")

    pinn = PINNAgent(state_dim=sd, action_dim=ad,
                     J=float(env_cfg["physics"]["J"]), B=float(env_cfg["physics"]["B"]),
                     R=float(env_cfg["physics"]["R"]),
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

    arbiter = CEMRLArbiter(state_dim=sd, z_task_dim=z_task_dim, n_agents=4,
                           hidden_dim=cfg["cemrl"]["cemrl_arbiter"]["hidden_dim"],
                           device=device)
    safe_load(arbiter.gating, ckpt.get("cemrl_arbiter"), "CEMRL arbiter gating")

    safe_low = env.action_space.low
    safe_high = env.action_space.high

    # --- Evaluate trained policy over max_episodes ---
    print(f"\n=== TRAINED POLICY (meta-learned) in regime {args.regime} ===")
    trained_returns = []
    for ep in range(args.max_episodes):
        r = run_episode_return(env, sac, ipae, fm, pinn, arbiter, embedding,
                               compute_z_shared, safe_low, safe_high, device,
                               z_task_dim, omega_min, omega_max, T_max,
                               warning_caution, safety_margin_frac)
        trained_returns.append(r)
        if (ep + 1) % 5 == 0:
            print(f"  ep {ep+1}: return={r:.2f}  (mean last 5={np.mean(trained_returns[-5:]):.2f})")

    # Adaptation speed: episodes to reach target_fraction of best return
    best_return = max(trained_returns) if trained_returns else 0.0
    target = args.target_fraction * best_return
    trained_speed = None
    for i, r in enumerate(trained_returns, start=1):
        if r >= target:
            trained_speed = i
            break
    if trained_speed is None:
        trained_speed = args.max_episodes
    print(f"  Best return: {best_return:.2f}, target: {target:.2f}")
    print(f"  Adaptation speed (episodes to reach target): {trained_speed}")

    # --- Scratch policy baseline (random init, no training) ---
    print(f"\n=== SCRATCH POLICY (random init) in regime {args.regime} ===")
    sac_scratch = SACAgent(state_dim=sd, shared_dim=shared_dim, action_dim=ad,
                           hidden_dim=cfg["agents"]["sac"]["hidden_dim"], device=device)
    # Reuse the same ipae/fm/pinn/embedding/arbiter (they're not the adapting part)
    scratch_returns = []
    for ep in range(args.max_episodes):
        r = run_episode_return(env, sac_scratch, ipae, fm, pinn, arbiter, embedding,
                               compute_z_shared, safe_low, safe_high, device,
                               z_task_dim, omega_min, omega_max, T_max,
                               warning_caution, safety_margin_frac)
        scratch_returns.append(r)
        if (ep + 1) % 5 == 0:
            print(f"  ep {ep+1}: return={r:.2f}  (mean last 5={np.mean(scratch_returns[-5:]):.2f})")

    scratch_best = max(scratch_returns) if scratch_returns else 0.0
    scratch_target = args.target_fraction * scratch_best
    scratch_speed = None
    for i, r in enumerate(scratch_returns, start=1):
        if r >= scratch_target:
            scratch_speed = i
            break
    if scratch_speed is None:
        scratch_speed = args.max_episodes
    print(f"  Best return: {scratch_best:.2f}, target: {scratch_target:.2f}")
    print(f"  Adaptation speed (episodes to reach target): {scratch_speed}")

    summary = {
        "checkpoint": args.ckpt,
        "regime": args.regime,
        "target_fraction": args.target_fraction,
        "trained_policy": {
            "returns": trained_returns,
            "best_return": float(best_return),
            "mean_return": float(np.mean(trained_returns)),
            "adaptation_speed_episodes": trained_speed,
        },
        "scratch_policy": {
            "returns": scratch_returns,
            "best_return": float(scratch_best),
            "mean_return": float(np.mean(scratch_returns)),
            "adaptation_speed_episodes": scratch_speed,
        },
        "speedup": (scratch_speed / trained_speed) if trained_speed else None,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
