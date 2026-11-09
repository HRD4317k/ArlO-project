#!/usr/bin/env python3
"""
Validate all checkpoints from the fresh_ipae_supervised run.

For each checkpoint (ep_00100 ... ep_05000):
  1. Loads the trained components (SAC, iPAE, FM, PINN+policy, CEMRL arbiter,
     shared embedding) using the SAME architecture as the training run.
  2. Runs R1-start (delta in [0.25, 0.45]) and R2-start (delta in [0.50, 0.70])
     degradation tests with the blended action (the full learned system).
  3. NEW: tracks energy production (P_out / rated) to see how the checkpoint's
     learned parameters affect energy generation alongside degradation control.
  4. Computes summary metrics per checkpoint:
     - delta slope / change / acceleration (degradation control)
     - steps to critical (delta >= 0.9)
     - mean reward
     - mean energy production (r_energy proxy = P_out)
     - energy-degradation tradeoff
     - gating weights (regime specialisation)

Results saved to results/checkpoint_validation_results.json
"""

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


def build_supervised_z_task(regime_id, delta, n_regimes=4, z_dim=32, device="cpu"):
    """Deterministic supervised task embedding (matches training)."""
    z = torch.zeros(z_dim, device=device)
    if regime_id is not None and 0 <= regime_id < min(n_regimes, z_dim):
        z[regime_id] = 1.0
    delta_slot = min(n_regimes, z_dim - 1)
    z[delta_slot] = float(delta)
    return z


def safety_urgency(state, omega_min, omega_max, T_max, margin_frac=0.15):
    """Continuous proximity-to-limit signal (matches training)."""
    omega = float(state[0]) if len(state) > 0 else 1.0
    T_bearing = float(state[3]) if len(state) > 3 else 50.0
    center = 0.5 * (omega_min + omega_max)
    half = 0.5 * (omega_max - omega_min)
    om_prox = abs(omega - center) / half if half > 0 else 0.0
    om_urgency = np.clip((om_prox - (1.0 - margin_frac)) / margin_frac, 0.0, 1.0)
    T_urgency = np.clip((T_bearing - (1.0 - margin_frac) * T_max) / (margin_frac * T_max), 0.0, 1.0)
    return float(max(om_urgency, T_urgency))


def run_episode(env, sac, ipae, fm, pinn, arbiter, embedding, compute_z_shared,
                safe_low, safe_high, device, init_delta=None, z_task_dim=32,
                omega_min=0.5, omega_max=1.5, T_max=100.0, warning_caution=0.5,
                safety_margin_frac=0.15):
    """Run a single episode with the full learned system (blended action)."""
    if init_delta is not None:
        state, _ = env.reset(options={"delta_range": (init_delta, init_delta)})
    else:
        state, _ = env.reset()
    if hasattr(ipae, "reset_window"):
        ipae.reset_window()
    done = False
    steps = 0

    deltas = []
    regimes = []
    rewards = []
    weights_list = []
    energies = []  # P_out values
    omegas = []
    temps = []

    while not done:
        delta_curr = float(state[5])
        regime_curr = regime_from_delta(delta_curr)
        deltas.append(delta_curr)
        regimes.append(regime_curr)
        energies.append(float(state[4]))  # P_out
        omegas.append(float(state[0]))
        temps.append(float(state[3]))

        # Compute actions (same as training loop)
        z_shared = compute_z_shared(state)
        a_sac = sac.act(state, z_shared)
        a_ipae = ipae.act(state)
        a_fm = fm.act(state)
        a_pinn = pinn.policy_act(state)  # learned PINN policy

        # Safety urgency (matches training)
        uncertainty = ipae.get_uncertainty_np(state)
        fm_severity, _ = fm.assess(state)
        is_anomaly = fm_severity == "critical"
        urgency = safety_urgency(state, omega_min, omega_max, T_max, safety_margin_frac)
        caution = max(warning_caution if fm_severity == "warning" else 0.0, urgency)

        # Supervised z_task
        z_task = build_supervised_z_task(regime_curr, delta_curr, z_dim=z_task_dim, device=device)

        action_proposals = {"nn": a_sac, "gp": a_ipae, "fm": a_fm, "physics": a_pinn}
        action, weights = arbiter.select_action(
            state, action_proposals,
            {"low": safe_low, "high": safe_high},
            z_task=z_task,
            delta=delta_curr,
            uncertainty=uncertainty,
            emergency=is_anomaly,
            caution=caution,
            return_weights=True,
        )
        weights_list.append(weights.tolist())

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
        "omegas": np.array(omegas),
        "temps": np.array(temps),
        "episode_length": steps,
        "terminated_early": bool(terminated),
    }


def compute_metrics(traj, label):
    """Summarize a trajectory including energy production."""
    deltas = traj["deltas"]
    regimes = traj["regimes"]
    rewards = traj["rewards"]
    energies = traj["energies"]
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

    # Energy production metrics
    mean_energy = float(energies.mean()) if len(energies) > 0 else 0.0
    max_energy = float(energies.max()) if len(energies) > 0 else 0.0
    energy_fraction = float((energies > 0.1).mean()) if len(energies) > 0 else 0.0

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg_path = os.environ.get("HMARL_VALIDATION_CONFIG",
                              "hmarl_drt/configs/wind_fresh_ipae_supervised.yaml")
    ckpt_dir = os.environ.get("HMARL_VALIDATION_CKPT_DIR",
                              "hmarl_drt/runs/fresh_ipae_supervised/checkpoints")
    N_EVAL = int(os.environ.get("HMARL_VALIDATION_EPISODES", 10))
    delta_ranges = {
        "R1_start": (0.25, 0.45),
        "R2_start": (0.50, 0.70),
    }

    cfg = yaml.safe_load(open(cfg_path))
    env_cfg = make_env_config(cfg)
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

    # Collect checkpoints
    ckpt_files = sorted(
        [p for p in Path(ckpt_dir).glob("ep_*.pt")],
        key=lambda p: int(p.stem.split("_")[1]),
    )
    # Optional filter: only validate specific episodes (comma-separated), e.g.
    # HMARL_VALIDATION_EPISODES_FILTER="100,500,1000"
    filter_str = os.environ.get("HMARL_VALIDATION_EPISODES_FILTER", "")
    if filter_str:
        wanted = {int(x) for x in filter_str.split(",") if x.strip()}
        ckpt_files = [p for p in ckpt_files if int(p.stem.split("_")[1]) in wanted]
    print(f"Found {len(ckpt_files)} checkpoints in {ckpt_dir}")

    all_results = {}

    for ckpt_path in ckpt_files:
        ep_num = int(ckpt_path.stem.split("_")[1])
        print(f"\n{'='*70}")
        print(f"Validating checkpoint ep_{ep_num:05d}")
        print(f"{'='*70}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Build agents with the SAME architecture as training
        sac = SACAgent(state_dim=sd, shared_dim=shared_dim, action_dim=ad,
                       hidden_dim=cfg["agents"]["sac"]["hidden_dim"], device=device)
        if "sac_actor" in ckpt:
            safe_load(sac.actor, ckpt["sac_actor"], "SAC actor")
        if "sac_critic" in ckpt:
            safe_load(sac.critic, ckpt["sac_critic"], "SAC critic")

        pinn = PINNAgent(state_dim=sd, action_dim=ad,
                         J=float(env_cfg["physics"]["J"]),
                         B=float(env_cfg["physics"]["B"]),
                         R=float(env_cfg["physics"]["R"]),
                         omega_min=omega_min, omega_max=omega_max, T_max=T_max,
                         hidden_dim=cfg["agents"]["pinn"]["hidden_dim"], device=device)
        if "pinn" in ckpt:
            safe_load(pinn.pinn, ckpt["pinn"], "PINN")
        if "pinn_policy" in ckpt:
            safe_load(pinn.policy, ckpt["pinn_policy"], "PINN policy")

        ipae = IPAEAgent(state_dim=sd, action_dim=ad,
                         latent_dim=ipae_cfg.get("latent_dim", 8),
                         window_size=ipae_cfg.get("window_size", 10),
                         encoder_hidden=ipae_cfg.get("encoder_hidden", 64),
                         lstm_hidden=ipae_cfg.get("lstm_hidden", 64),
                         lstm_layers=ipae_cfg.get("lstm_layers", 1),
                         rul_hidden=ipae_cfg.get("rul_hidden", [64, 32, 16]),
                         mc_dropout=ipae_cfg.get("mc_dropout", 0.1),
                         device=device)
        if "gp_network" in ckpt:
            safe_load(ipae.network, ckpt["gp_network"], "iPAE network")

        fm = FMAgent(state_dim=sd, use_api=False, device=device,
                     omega_min=omega_min, omega_max=omega_max,
                     T_max=T_max, delta_crit=float(constr.get("delta_crit", 0.9)))

        # Shared embedding
        def build_shared_embedding():
            with torch.no_grad():
                dummy = torch.zeros(1, sd, device=device)
                dims = [
                    sac.actor.get_embedding(dummy).shape[-1],
                    ipae.get_embedding(dummy).shape[-1],
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
                z2 = ipae.get_embedding(s_t)
                z3 = fm.get_embedding(s_t)
                z4 = pinn.get_embedding(s_t)
                zs, _ = embedding([z1, z2, z3, z4])
            return zs.squeeze(0)

        # CEMRL arbiter (matches training)
        arbiter = CEMRLArbiter(state_dim=sd, z_task_dim=z_task_dim, n_agents=4,
                               hidden_dim=cfg["cemrl"]["cemrl_arbiter"]["hidden_dim"],
                               device=device)
        if "cemrl_arbiter" in ckpt:
            safe_load(arbiter.gating, ckpt["cemrl_arbiter"], "CEMRL arbiter gating")

        safe_low = env.action_space.low
        safe_high = env.action_space.high

        # Run validation for each delta range
        ckpt_results = {}
        for range_label, (delta_lo, delta_hi) in delta_ranges.items():
            print(f"\n  Starting regime: {range_label}  (Δ ∈ [{delta_lo}, {delta_hi}])")
            entries = []
            for ep in range(N_EVAL):
                init_delta = np.random.uniform(delta_lo, delta_hi)
                traj = run_episode(
                    env, sac, ipae, fm, pinn, arbiter, embedding,
                    compute_z_shared, safe_low, safe_high, device,
                    init_delta=init_delta, z_task_dim=z_task_dim,
                    omega_min=omega_min, omega_max=omega_max, T_max=T_max,
                    warning_caution=warning_caution, safety_margin_frac=safety_margin_frac,
                )
                entries.append(compute_metrics(traj, "blended"))

            agg = {
                "num_episodes": len(entries),
                "avg_episode_length": float(np.mean([e["episode_length"] for e in entries])),
                "avg_delta_change": float(np.mean([e["delta_change"] for e in entries])),
                "avg_delta_slope": float(np.mean([e["delta_slope"] for e in entries])),
                "avg_delta_accel": float(np.mean([e["delta_accel"] for e in entries])),
                "avg_mean_reward": float(np.mean([e["mean_reward"] for e in entries])),
                # Energy production
                "avg_mean_energy": float(np.mean([e["mean_energy"] for e in entries])),
                "avg_max_energy": float(np.mean([e["max_energy"] for e in entries])),
                "avg_energy_fraction": float(np.mean([e["energy_fraction"] for e in entries])),
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
            ckpt_results[range_label] = agg
            print(f"    Δ_change={agg['avg_delta_change']:+.6f}  "
                  f"Δ_slope={agg['avg_delta_slope']:+.2e}  "
                  f"reward={agg['avg_mean_reward']:.4f}  "
                  f"energy={agg['avg_mean_energy']:.4f}  "
                  f"improve={agg['improvement_rate']:.2f}")

        all_results[f"ep_{ep_num:05d}"] = ckpt_results

    summary = {
        "checkpoint_dir": ckpt_dir,
        "num_episodes_per_range": N_EVAL,
        "delta_ranges": delta_ranges,
        "results": all_results,
    }
    print("\n\n=== CHECKPOINT VALIDATION SUMMARY ===")
    print(json.dumps(summary, indent=2))
    out_path = Path("results/checkpoint_validation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
