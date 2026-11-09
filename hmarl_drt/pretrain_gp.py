"""
Phase 1b: GP Agent Pre-Training
================================
Trains the GP agent on regime-specific degradation trajectories.

Loss: L_GP = -log p(RUL | state, regime) (negative log likelihood)
Target: RUL MAE < 0.15, uncertainty calibration (coverage > 90%)
Episodes: 200 (50 per regime)

Usage:
    python pretrain_gp.py --config configs/phase1b_gp_pretrain.yaml
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from envs.wind_turbine_env import WindTurbineEnv
from agents.gp_agent import GPAgent
from agents.pinn_agent import PINNAgent


def pretrain_gp(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = cfg.get("env", {})
    env = WindTurbineEnv(env_cfg)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    gp_cfg = cfg.get("agents", {}).get("gp", {})
    agent = GPAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=float(gp_cfg.get("lr", 0.01)),
        device=device,
    )

    n_episodes = cfg.get("training", {}).get("n_episodes", 200)
    seed = cfg.get("seed", 42)
    log_dir = cfg.get("logging", {}).get("log_dir", "runs/phase1b_gp/")
    os.makedirs(log_dir, exist_ok=True)
    save_freq = cfg.get("training", {}).get("save_freq", 50)

    print(f"\n{'='*60}")
    print(f"  Phase 1b: GP Agent Pre-Training")
    print(f"  Episodes: {n_episodes} | Device: {device}")
    print(f"{'='*60}\n")

    # BUG FIX: Load pre-trained PINN from Phase 1a instead of creating
    # a new random one every episode. The old code created PINNAgent inside
    # the episode loop with random weights, producing inconsistent actions.
    pinn_ckpt_path = cfg.get("training", {}).get("pinn_checkpoint", "runs/phase1a_pinn/pinn_final.pt")
    pinn_agent = PINNAgent(
        state_dim=state_dim, action_dim=action_dim,
        J=float(env_cfg.get("physics", {}).get("J", 1e6)),
        B=float(env_cfg.get("physics", {}).get("B", 100.0)),
        R=float(env_cfg.get("physics", {}).get("R", 40.0)),
        omega_min=float(env_cfg.get("constraints", {}).get("omega_min", 0.5)),
        omega_max=float(env_cfg.get("constraints", {}).get("omega_max", 1.5)),
        T_max=float(env_cfg.get("constraints", {}).get("T_max", 100.0)),
        device=device,
    )
    if os.path.exists(pinn_ckpt_path):
        try:
            pinn_ckpt = torch.load(pinn_ckpt_path, map_location=device, weights_only=False)
            if "pinn_state_dict" in pinn_ckpt:
                pinn_agent.pinn.load_state_dict(pinn_ckpt["pinn_state_dict"], strict=False)
                print(f"  [Phase1b] Loaded pre-trained PINN from {pinn_ckpt_path}")
        except Exception as exc:
            print(f"  [Phase1b] Could not load PINN checkpoint: {exc}")
    else:
        print(f"  [Phase1b] PINN checkpoint not found at {pinn_ckpt_path}, using untrained PINN")

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_rul_errors = []
        steps = 0
        done = False

        while not done and steps < 1000:
            # Use pre-trained PINN physics-based action for meaningful training signal
            action = pinn_agent.act(state)
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # True RUL proxy: based on delta (higher delta = lower RUL)
            delta = float(next_state[5]) if len(next_state) > 5 else 0.0
            true_rul = max(0.0, 500.0 * (1.0 - delta))

            # Update GP with transition
            s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            a_t = torch.FloatTensor(action).unsqueeze(0).to(device)
            rul_t = torch.FloatTensor([[true_rul]]).to(device)

            # Train action-prediction ensemble AND RUL head
            agent.update(s_t, a_t, rul_t)

            # Also do a dedicated RUL update with true RUL target
            agent.update_rul(s_t, torch.FloatTensor([true_rul]).to(device))

            # Estimate RUL and compute error
            rul_est = agent.estimate_rul(state)
            ep_rul_errors.append(abs(rul_est - true_rul))

            state = next_state
            steps += 1

        if (ep + 1) % 10 == 0:
            mean_mae = np.mean(ep_rul_errors) if ep_rul_errors else 0.0
            print(f"  Ep {ep+1:3d}/{n_episodes} | RUL MAE: {mean_mae:.4f} | Steps: {steps}")

        if (ep + 1) % save_freq == 0:
            ckpt_path = os.path.join(log_dir, f"gp_ep{ep+1:04d}.pt")
            torch.save({"gp_network_state_dict": agent.network.state_dict()}, ckpt_path)
            print(f"  [Saved] {ckpt_path}")

    final_path = os.path.join(log_dir, "gp_final.pt")
    torch.save({"gp_network_state_dict": agent.network.state_dict()}, final_path)
    print(f"\n  [Final] Saved to {final_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1b_gp_pretrain.yaml")
    args = parser.parse_args()
    pretrain_gp(args.config)
