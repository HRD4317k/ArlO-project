#!/usr/bin/env python3
"""
Deep-dive validation of the best checkpoint (ep_2300) using the CORRECT
architecture from the fresh_ipae_supervised run (iPAE + CEMRL arbiter +
learned PINN policy).

Tests:
  1. Degradation-combination: all agent action modes vs zero baseline
     - combined (full 4-agent blend)
     - 3-agent combos (arbiter re-normalized over subset)
     - 2-agent combos
     - individual agents (sac, ipae, fm, pinn)
  2. Energy production per mode (how each agent's actions affect energy)
  3. R1-start and R2-start degradation tests

For each mode, runs N_EVAL episodes from fresh seeds and reports mean+std of:
  - delta slope / change (degradation rate)
  - steps to critical
  - mean reward
  - mean energy / energy fraction
  - improvement/worsening rate
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from itertools import combinations

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
from hmarl_drt.agents.laya_agent import LayaAgent
from hmarl_drt.core.embedding import SharedEmbeddingSpace
from hmarl_drt.core.cemrl_arbiter import CEMRLArbiter


def safe_load(module, state_dict, label: str):
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


def get_arbiter_weights(arbiter, state, z_task, delta, uncertainty, omega_min,
                        omega_max, T_max, warning_caution, safety_margin_frac):
    """Compute the CEMRL arbiter's gating weights (learned + regime prior + caution)."""
    fm_severity = None  # caller handles severity; here we just get raw weights
    weights = arbiter.get_weights(state, z_task, delta, uncertainty)
    regime_bias = arbiter._regime_prior_bias(delta)
    weights = 0.7 * weights + 0.3 * regime_bias
    return weights


def run_episode(env, sac, ipae, fm, pinn, arbiter, embedding, compute_z_shared,
                safe_low, safe_high, device, action_mode="combined", init_delta=None,
                z_task_dim=32, omega_min=0.5, omega_max=1.5, T_max=100.0,
                warning_caution=0.5, safety_margin_frac=0.15, subset=None, ep_seed=None):
    """Run a single episode with a given action mode."""
    if init_delta is not None:
        state, _ = env.reset(seed=ep_seed, options={"delta_range": (init_delta, init_delta)})
    else:
        state, _ = env.reset()
    if hasattr(ipae, "reset_window"):
        ipae.reset_window()
    done = False
    steps = 0

    deltas, regimes, rewards, weights_list, energies = [], [], [], [], []

    while not done:
        delta_curr = float(state[5])
        regime_curr = regime_from_delta(delta_curr)
        deltas.append(delta_curr)
        regimes.append(regime_curr)
        energies.append(float(state[4]))

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
        agent_names = ["nn", "gp", "fm", "physics"]

        if action_mode == "zero":
            action = np.zeros_like(a_sac)
            weights_list.append([0.25, 0.25, 0.25, 0.25])
        elif action_mode == "fixed_maxpitch":
            action = np.array([1.0, -1.0, -1.0], dtype=np.float32)
            weights_list.append([0.0, 0.0, 0.0, 0.0])
        elif action_mode == "combined":
            action, weights = arbiter.select_action(
                state, proposals, {"low": safe_low, "high": safe_high},
                z_task=z_task, delta=delta_curr, uncertainty=uncertainty,
                emergency=is_anomaly, caution=caution, return_weights=True,
            )
            weights_list.append(weights.tolist())
        elif action_mode.startswith("subset_"):
            # e.g. subset_nn_gp or subset_nn_gp_fm
            members = action_mode.split("_")[1:]
            # Get full arbiter weights, then re-normalize over the subset
            weights = get_arbiter_weights(arbiter, state, z_task, delta_curr,
                                          uncertainty, omega_min, omega_max, T_max,
                                          warning_caution, safety_margin_frac)
            idx = [agent_names.index(m) for m in members]
            sub_w = weights[idx]
            sub_w = sub_w / sub_w.sum()
            full_w = np.zeros(4)
            full_w[idx] = sub_w
            # Apply caution to the subset blend
            if caution > 0:
                physics_only = np.zeros(4)
                physics_only[3] = 1.0
                full_w = (1 - caution) * full_w + caution * physics_only
            weights_list.append(full_w.tolist())
            action = sum(full_w[i] * np.array(proposals[agent_names[i]]) for i in range(4))
            action = np.clip(action, safe_low, safe_high).astype(np.float32)
        elif action_mode == "sac":
            action = a_sac
            weights_list.append([1.0, 0.0, 0.0, 0.0])
        elif action_mode == "ipae":
            action = a_ipae
            weights_list.append([0.0, 1.0, 0.0, 0.0])
        elif action_mode == "fm":
            action = a_fm
            weights_list.append([0.0, 0.0, 1.0, 0.0])
        elif action_mode == "pinn":
            action = a_pinn
            weights_list.append([0.0, 0.0, 0.0, 1.0])
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
        "energies": np.array(energies),
        "episode_length": steps,
        "terminated_early": bool(terminated),
    }


def compute_metrics(traj, label):
    deltas = traj["deltas"]
    regimes = traj["regimes"]
    rewards = traj["rewards"]
    energies = traj["energies"]
    n = len(deltas)

    delta_slope = np.polyfit(np.arange(n), deltas, 1)[0] if n > 1 else 0.0
    delta_change = deltas[-1] - deltas[0]
    regime_fractions = {f"R{r}": float((regimes == r).mean()) for r in range(4)}
    start_regime, end_regime = regimes[0], regimes[-1]
    improved = int(end_regime < start_regime)
    worsened = int(end_regime > start_regime)
    # Explicit: did the episode reach a WORSE (higher) regime than it started?
    reached_next_regime = int(end_regime > start_regime)
    # Steps until the first regime transition (if any)
    first_transition_step = None
    if n > 1:
        diffs = np.diff(regimes)
        nz = np.where(diffs != 0)[0]
        if len(nz) > 0:
            first_transition_step = int(nz[0] + 1)

    # Count regime transitions: number of times the regime changes between
    # consecutive steps. Also count upward (worsening) and downward (improving)
    # transitions separately.
    n_transitions = 0
    n_up_transitions = 0
    n_down_transitions = 0
    if n > 1:
        diffs = np.diff(regimes)
        n_transitions = int((diffs != 0).sum())
        n_up_transitions = int((diffs > 0).sum())
        n_down_transitions = int((diffs < 0).sum())

    critical_steps = np.where(deltas >= 0.9)[0]
    steps_to_critical = int(critical_steps[0]) if len(critical_steps) > 0 else None
    mean_reward = float(rewards.mean()) if len(rewards) > 0 else 0.0
    mean_energy = float(energies.mean()) if len(energies) > 0 else 0.0
    max_energy = float(energies.max()) if len(energies) > 0 else 0.0
    energy_fraction = float((energies > 0.1).mean()) if len(energies) > 0 else 0.0

    return {
        "label": label,
        "episode_length": int(n),
        "start_regime": int(start_regime),
        "end_regime": int(end_regime),
        "start_delta": float(deltas[0]),
        "end_delta": float(deltas[-1]),
        "delta_change": float(delta_change),
        "n_transitions": n_transitions,
        "n_up_transitions": n_up_transitions,
        "n_down_transitions": n_down_transitions,
        "reached_next_regime": reached_next_regime,
        "first_transition_step": first_transition_step,
        "delta_slope": float(delta_slope),
        "mean_reward": mean_reward,
        "mean_energy": mean_energy,
        "max_energy": max_energy,
        "energy_fraction": energy_fraction,
        "regime_fractions": regime_fractions,
        "improved": improved,
        "worsened": worsened,
        "steps_to_critical": steps_to_critical,
        "terminated_early": traj["terminated_early"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="hmarl_drt/runs/fresh_ipae_supervised/checkpoints/ep_02300.pt")
    parser.add_argument("--config", default="hmarl_drt/configs/wind_fresh_ipae_supervised.yaml")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=90000)
    parser.add_argument("--deg-acceleration", type=float, default=None,
                        help="Override degradation rate multiplier (default: use config)")
    parser.add_argument("--temp-deg-model", default=None, choices=["quadratic", "arrhenius"])
    parser.add_argument("--out", default="results/ep2300_deepdive_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = yaml.safe_load(open(args.config))
    env_cfg = make_env_config(cfg)
    if args.deg_acceleration is not None:
        env_cfg.setdefault("physics", {})
        env_cfg["physics"]["deg_acceleration"] = args.deg_acceleration
        print(f"Degradation acceleration override: {args.deg_acceleration}")
    if args.temp_deg_model is not None:
        env_cfg.setdefault("physics", {})
        env_cfg["physics"]["temp_deg_model"] = args.temp_deg_model
        print(f"Temperature degradation model: {args.temp_deg_model}")
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
    print(f"Loaded checkpoint: {args.ckpt}")

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

    fm_cfg = cfg.get("agents", {}).get("fm", {})
    if fm_cfg.get("backend", "legacy") == "laya":
        fm = LayaAgent(
            state_dim=sd,
            device=device,
            endpoint=fm_cfg.get("endpoint"),
            model_name=fm_cfg.get("model_name", "english"),
            timeout_s=fm_cfg.get("api_timeout_s", 10.0),
            omega_min=omega_min,
            omega_max=omega_max,
            T_max=T_max,
            delta_crit=float(constr.get("delta_crit", 0.9)),
        )
        if fm_cfg.get("health_check", True):
            fm.health_check()
    else:
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

    # Action modes: combined, zero, individuals, all 3-agent and 2-agent subsets
    agent_names = ["nn", "gp", "fm", "physics"]
    action_modes = ["combined", "zero", "fixed_maxpitch", "sac", "ipae", "fm", "pinn"]
    for r in range(3, 1, -1):
        for combo in combinations(agent_names, r):
            action_modes.append("subset_" + "_".join(combo))

    delta_ranges = {"R1_start": (0.25, 0.45), "R2_start": (0.50, 0.70)}
    all_results = {}

    for range_label, (delta_lo, delta_hi) in delta_ranges.items():
        print(f"\n{'='*70}")
        print(f"Starting regime: {range_label}  (Δ ∈ [{delta_lo}, {delta_hi}])")
        print(f"{'='*70}")
        range_results = defaultdict(list)
        for ep in range(args.n_episodes):
            init_delta = np.random.uniform(delta_lo, delta_hi)
            for mode in action_modes:
                traj = run_episode(
                    env, sac, ipae, fm, pinn, arbiter, embedding,
                    compute_z_shared, safe_low, safe_high, device,
                    action_mode=mode, init_delta=init_delta,
                    z_task_dim=z_task_dim, omega_min=omega_min, omega_max=omega_max,
                    T_max=T_max, warning_caution=warning_caution,
                    safety_margin_frac=safety_margin_frac,
                    ep_seed=10_000 + 1000 * len(all_results) + ep,
                )
                range_results[mode].append(compute_metrics(traj, mode))

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
                "avg_mean_reward": float(np.mean([e["mean_reward"] for e in entries])),
                "avg_mean_energy": float(np.mean([e["mean_energy"] for e in entries])),
                "avg_max_energy": float(np.mean([e["max_energy"] for e in entries])),
                "avg_energy_fraction": float(np.mean([e["energy_fraction"] for e in entries])),
                "improvement_rate": float(np.mean([e["improved"] for e in entries])),
                "worsening_rate": float(np.mean([e["worsened"] for e in entries])),
                # Regime transition statistics
                "avg_n_transitions": float(np.mean([e["n_transitions"] for e in entries])),
                "avg_n_up_transitions": float(np.mean([e["n_up_transitions"] for e in entries])),
                "avg_n_down_transitions": float(np.mean([e["n_down_transitions"] for e in entries])),
                "total_n_transitions": int(sum([e["n_transitions"] for e in entries])),
                "reached_next_regime_rate": float(np.mean([e["reached_next_regime"] for e in entries])),
                "n_reached_next_regime": int(sum([e["reached_next_regime"] for e in entries])),
                "avg_first_transition_step": float(np.mean([e["first_transition_step"] for e in entries if e["first_transition_step"] is not None])) if any(e["first_transition_step"] is not None for e in entries) else None,
                "transition_rate_per_1000_steps": float(
                    np.mean([e["n_transitions"] / max(e["episode_length"], 1) * 1000 for e in entries])
                ),
                "std_delta_change": float(np.std([e["delta_change"] for e in entries])),
                "std_mean_reward": float(np.std([e["mean_reward"] for e in entries])),
                "std_mean_energy": float(np.std([e["mean_energy"] for e in entries])),
                "avg_regime_fractions": {},
                "steps_to_critical": [],
            }
            for r in range(4):
                vals = [e["regime_fractions"][f"R{r}"] for e in entries]
                agg["avg_regime_fractions"][f"R{r}"] = float(np.mean(vals))
            stc = [e["steps_to_critical"] for e in entries if e["steps_to_critical"] is not None]
            agg["steps_to_critical"] = {
                "count": len(stc),
                "mean_steps": float(np.mean(stc)) if stc else None,
            }
            aggregated[mode] = agg
            print(f"  {mode:>18s}: Δ_change={agg['avg_delta_change']:+.6f}  "
                  f"reward={agg['avg_mean_reward']:.4f}  "
                  f"energy={agg['avg_mean_energy']:.4f}  "
                  f"transitions={agg['avg_n_transitions']:.1f}  "
                  f"improve={agg['improvement_rate']:.2f}")
        all_results[range_label] = aggregated

    summary = {
        "checkpoint": args.ckpt,
        "num_episodes": args.n_episodes,
        "results": all_results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
