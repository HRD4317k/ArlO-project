#!/usr/bin/env python3
"""
Validation script for assessing the *combined* action produced by the arbiter
network (gating weights) on environment transitions and the resulting RUL /
degradation index.

The script loads a checkpoint (default: the 2500‑episode checkpoint located at
``runs/archive_pre_fresh/checkpoints/ep_02500.pt``), instantiates all agents and
the ``Arbiter`` gating module, and runs a number of evaluation episodes.  At each
environment step the following occurs:

1. Each agent proposes an action ``a_i``.
2. The arbiter computes a weight vector ``w`` (softmax over the four agents).
3. The *combined* action ``a_comb = sum_i w_i * a_i`` is applied to the
   environment.
4. The current degradation index (``delta`` – the 6th state component) and the
   remaining steps (as a proxy for RUL) are recorded together with the gating
   weights.

After the roll‑out the script reports:

* Average gating weight per agent.
* Correlation between the dominant gating weight and the change in ``delta``
  (i.e., whether the arbiter tends to push the system towards a different
  regime).
* Average episode length (proxy for RUL) per dominant agent.

The script is deliberately lightweight and prints a concise JSON summary that
can be inspected or saved to ``results/combined_action_validation.json``.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import mean_squared_error
from hmarl_drt.core.embedding import SharedEmbeddingSpace

# Ensure the repository root is on ``sys.path`` so imports resolve.
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hmarl_drt"))

from hmarl_drt.envs.wind_turbine_env import WindTurbineEnv
from hmarl_drt.agents.sac_agent import SACAgent
from hmarl_drt.agents.pinn_agent import PINNAgent
from hmarl_drt.agents.gp_agent import GPAgent
from hmarl_drt.agents.fm_agent import FMAgent
from hmarl_drt.core.arbiter import Arbiter


def safe_load(module, state_dict, label: str):
    """Load compatible tensors from a checkpoint into *module*.

    In practice checkpoints may contain extra keys (e.g. optimizer state) or
    tensors with mismatched shapes.  This helper loads only the matching ones
    and reports how many were skipped.
    """
    if not state_dict:
        return
    current = module.state_dict()
    compatible = {k: v for k, v in state_dict.items() if k in current and tuple(current[k].shape) == tuple(v.shape)}
    skipped = set(state_dict.keys()) - set(compatible.keys())
    module.load_state_dict(compatible, strict=False)
    print(f"  Loaded {label}: {len(compatible)} tensors" + (f", skipped {len(skipped)}" if skipped else ""))


def make_env_config(cfg: dict) -> dict:
    """Create a deterministic environment configuration for validation.

    The validation uses the *true* degradation index (no masking) and samples a
    new regime each episode to expose the arbiter to regime transitions.
    """
    env_cfg = dict(cfg.get("env", cfg))
    env_cfg["mask_true_delta"] = False
    env_cfg["max_steps"] = int(env_cfg.get("max_steps", 1000))
    env_cfg.setdefault("degradation", {})
    env_cfg["degradation"].setdefault("enabled", True)
    env_cfg["degradation"].setdefault("sample_regime_each_episode", True)
    env_cfg["degradation"].setdefault("regime_delta_ranges", {
        "0": [0.0, 0.25],
        "1": [0.25, 0.50],
        "2": [0.50, 0.75],
        "3": [0.75, 0.95],
    })
    env_cfg["degradation"].setdefault("regime_probs", [0.25, 0.25, 0.25, 0.25])
    return env_cfg


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---------------------------------------------------------------------
    # Configuration & checkpoint handling
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # Environment & shared dimensions
    # ---------------------------------------------------------------------
    env = WindTurbineEnv(env_cfg)
    sd = int(env.observation_space.shape[0])
    ad = int(env.action_space.shape[0])
    shared_dim = cfg.get("embedding", {}).get("shared_dim", 64)

    # ---------------------------------------------------------------------
    # Instantiate agents and load checkpoint weights where available
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # Shared embedding space (required for SAC actions)
    # ---------------------------------------------------------------------
    def build_shared_embedding():
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
        return embedding

    embedding = build_shared_embedding()

    def compute_z_shared(state_np):
        """Compute shared latent embedding for a raw state array."""
        s_t = torch.FloatTensor(state_np).unsqueeze(0).to(device)
        with torch.no_grad():
            z1 = sac.actor.get_embedding(s_t)
            z2 = gp.get_embedding(s_t)
            z3 = fm.get_embedding(s_t)
            z4 = pinn.get_embedding(s_t)
            z_shared, _ = embedding([z1, z2, z3, z4])
        return z_shared.squeeze(0)

    # ---------------------------------------------------------------------
    # Arbiter (gating network)
    # ---------------------------------------------------------------------
    # Initialise the arbiter gating network. The constructor expects ``state_dim``
    # as the first argument and ``n_agents`` for the number of expert agents.
    arbiter = Arbiter(state_dim=sd, n_agents=4, hidden_dim=128, device=device)
    # Load arbiter gating network weights if present in the checkpoint.
    if "arbiter" in ckpt:
        # The checkpoint stores the state dict of the internal GatingNetwork.
        safe_load(arbiter.gating, ckpt["arbiter"], "Arbiter gating")

    # ---------------------------------------------------------------------
    # Evaluation loop
    # ---------------------------------------------------------------------
    N_EVAL = int(os.environ.get("HMARL_VALIDATION_EPISODES", 30))
    results = {
        "gating_weights": [],  # list of [w_fm, w_sac, w_pinn, w_gp] per step
        "delta_history": [],   # degradation index per step
        "regime_history": [],  # regime index per step
        "reward_history": [],  # reward per step
        "rul_estimate": [],    # GP RUL estimate per step
        "physics_violations": [],  # bool per step
        "episode_lengths": [],
    }

    safe_low = env.action_space.low
    safe_high = env.action_space.high
    for ep in range(N_EVAL):
        state, _ = env.reset(seed=ep)
        done = False
        steps = 0
        while not done:
            # 1. Individual agent actions (numpy arrays)
            a_fm = fm.act(state)
            # Compute shared embedding for SAC action
            z_shared = compute_z_shared(state)
            a_sac = sac.act(state, z_shared)
            a_pinn = pinn.act(state)
            a_gp = gp.act(state)
            # Record auxiliary metrics before stepping
            delta_curr = float(state[5])
            regime_curr = 0 if delta_curr < 0.25 else (1 if delta_curr < 0.5 else (2 if delta_curr < 0.75 else 3))
            rul_curr = gp.estimate_rul(state)
            results["delta_history"].append(delta_curr)
            results["regime_history"].append(regime_curr)
            results["rul_estimate"].append(rul_curr)
            # 2. Arbiter selects combined action and returns weights
            action_proposals = {"fm": a_fm, "sac": a_sac, "pinn": a_pinn, "gp": a_gp}
            combined, weights = arbiter.select_action(
                state,
                action_proposals,
                {"low": safe_low, "high": safe_high},
                delta=float(state[5]),
                return_weights=True,
            )
            # 3. Step environment with the combined action
            next_state, reward, terminated, truncated, info = env.step(combined)
            # Record metrics
            results["gating_weights"].append(weights.tolist())
            results["reward_history"].append(float(reward))
            results["physics_violations"].append(bool(info.get("physics_violated", False)))
            state = next_state
            done = terminated or truncated
            steps += 1
        results["episode_lengths"].append(steps)

    # ---------------------------------------------------------------------
    # Post‑processing & summary
    # ---------------------------------------------------------------------
    gating_arr = np.array(results["gating_weights"])  # (total_steps, 4)
    avg_weights = gating_arr.mean(axis=0).tolist()
    dominant_idx = gating_arr.argmax(axis=1)

    # Compute entropy per step
    entropy = -np.sum(gating_arr * np.log(gating_arr + 1e-12), axis=1)

    # Prepare arrays for other metrics
    delta_arr = np.array(results["delta_history"]).astype(float)
    regime_arr = np.array(results["regime_history"]).astype(int)
    reward_arr = np.array(results["reward_history"]).astype(float)
    rul_arr = np.array(results["rul_estimate"]).astype(float)
    physics_violation_arr = np.array(results["physics_violations"]).astype(int)

    # Next-step values (shifted by one, last step repeats last value)
    next_regime = np.roll(regime_arr, -1)
    next_delta = np.roll(delta_arr, -1)
    next_reward = np.roll(reward_arr, -1)
    next_rul = np.roll(rul_arr, -1)

    # Transition flag (regime change)
    transition_flag = (next_regime != regime_arr).astype(int)

    # Compute per-agent aggregated statistics
    agent_names = ["SAC", "GP", "FM", "PINN"]  # order matches gating indices
    per_agent = {}
    for idx, name in enumerate(agent_names):
        mask = dominant_idx == idx
        if mask.sum() == 0:
            continue
        prob_transition = transition_flag[mask].mean()
        mean_rul_change = (next_rul - rul_arr)[mask].mean()
        mean_delta_change = (next_delta - delta_arr)[mask].mean()
        mean_reward_next = next_reward[mask].mean()
        mean_entropy = entropy[mask].mean()
        per_agent[name] = {
            "transition_prob": float(prob_transition),
            "mean_next_rul_change": float(mean_rul_change),
            "mean_next_delta_change": float(mean_delta_change),
            "mean_next_reward": float(mean_reward_next),
            "mean_entropy": float(mean_entropy),
        }

    # Correlations between each weight column and outcomes
    corr_trans = []
    corr_rul = []
    corr_delta = []
    corr_reward = []
    for i in range(gating_arr.shape[1]):
        w = gating_arr[:, i]
        if len(w) > 1:
            corr_trans.append(float(np.corrcoef(w, transition_flag)[0, 1]))
            corr_rul.append(float(np.corrcoef(w, (next_rul - rul_arr))[0, 1]))
            corr_delta.append(float(np.corrcoef(w, (next_delta - delta_arr))[0, 1]))
            corr_reward.append(float(np.corrcoef(w, next_reward)[0, 1]))
        else:
            corr_trans.append(float('nan'))
            corr_rul.append(float('nan'))
            corr_delta.append(float('nan'))
            corr_reward.append(float('nan'))

    summary = {
        "average_gating_weights": {"SAC": avg_weights[0], "GP": avg_weights[1], "FM": avg_weights[2], "PINN": avg_weights[3]},
        "average_entropy": float(entropy.mean()),
        "per_agent_stats": per_agent,
        "correlations": {
            "weight_vs_transition": {name: corr_trans[i] for i, name in enumerate(agent_names)},
            "weight_vs_rul_change": {name: corr_rul[i] for i, name in enumerate(agent_names)},
            "weight_vs_delta_change": {name: corr_delta[i] for i, name in enumerate(agent_names)},
            "weight_vs_reward_next": {name: corr_reward[i] for i, name in enumerate(agent_names)},
        },
        "average_episode_length": float(np.mean(results["episode_lengths"])),
        "num_episodes": N_EVAL,
    }

    print("\n=== Combined‑Action Validation Summary ===")
    print(json.dumps(summary, indent=2))

    out_path = Path("results/combined_action_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()