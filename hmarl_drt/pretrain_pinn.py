"""
Phase 1a: PINN Pre-Training
=============================
Trains the PINN agent on physics simulation trajectories across all regimes.

Loss: L_PINN = L_MSE(predicted_state, true_state) + lambda_physics * L_ODE_residual
Target: ODE residual < 0.01, action magnitude 0.3-0.5
Episodes: 200 (50 per regime)

Usage:
    python pretrain_pinn.py --config configs/phase1a_pinn_pretrain.yaml
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from envs.wind_turbine_env import WindTurbineEnv
from agents.pinn_agent import PINNAgent


def pretrain_pinn(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = cfg.get("env", {})
    env = WindTurbineEnv(env_cfg)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    pinn_cfg = cfg.get("agents", {}).get("pinn", {})
    constr = env_cfg.get("constraints", {})
    physics = env_cfg.get("physics", {})
    agent = PINNAgent(
        state_dim=state_dim, action_dim=action_dim,
        J=float(physics.get("J", 1e6)), B=float(physics.get("B", 100.0)),
        R=float(physics.get("R", 40.0)),
        omega_min=float(constr.get("omega_min", 0.5)),
        omega_max=float(constr.get("omega_max", 1.5)),
        T_max=float(constr.get("T_max", 100.0)),
        hidden_dim=int(pinn_cfg.get("hidden_dim", 128)),
        lr=float(pinn_cfg.get("lr", 1e-3)),
        device=device,
    )

    n_episodes = cfg.get("training", {}).get("n_episodes", 200)
    seed = cfg.get("seed", 42)
    log_dir = cfg.get("logging", {}).get("log_dir", "runs/phase1a_pinn/")
    os.makedirs(log_dir, exist_ok=True)
    save_freq = cfg.get("training", {}).get("save_freq", 50)

    print(f"\n{'='*60}")
    print(f"  Phase 1a: PINN Pre-Training")
    print(f"  Episodes: {n_episodes} | Device: {device}")
    print(f"{'='*60}\n")

    ode_losses = []
    data_losses = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_ode_loss = 0.0
        ep_data_loss = 0.0
        steps = 0
        done = False

        while not done and steps < 1000:
            # Use random actions for exploration during pre-training
            # PINN's own actions are too conservative and cause early termination
            action = env.action_space.sample()

            s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            a_t = torch.FloatTensor(action).unsqueeze(0).to(device)

            # Compute ODE residual loss
            ode_loss = agent.ode_residual(s_t, a_t).item()

            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update PINN with transition data
            s_batch = torch.FloatTensor(state).unsqueeze(0).to(device)
            a_batch = torch.FloatTensor(action).unsqueeze(0).to(device)
            ns_batch = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            data_loss = agent.update(s_batch, a_batch, ns_batch)

            ep_ode_loss += ode_loss
            ep_data_loss += data_loss
            state = next_state
            steps += 1

        ode_losses.append(ep_ode_loss / max(steps, 1))
        data_losses.append(ep_data_loss / max(steps, 1))

        if (ep + 1) % 10 == 0:
            avg_ode = np.mean(ode_losses[-10:])
            avg_data = np.mean(data_losses[-10:])
            print(f"  Ep {ep+1:3d}/{n_episodes} | ODE loss: {avg_ode:.6f} | Data loss: {avg_data:.6f} | Steps: {steps}")

        if (ep + 1) % save_freq == 0:
            ckpt_path = os.path.join(log_dir, f"pinn_ep{ep+1:04d}.pt")
            torch.save({"pinn_state_dict": agent.pinn.state_dict()}, ckpt_path)
            print(f"  [Saved] {ckpt_path}")

    # Save final model
    final_path = os.path.join(log_dir, "pinn_final.pt")
    torch.save({"pinn_state_dict": agent.pinn.state_dict()}, final_path)
    print(f"\n  [Final] Saved to {final_path}")
    print(f"  Final ODE loss: {np.mean(ode_losses[-20:]):.6f}")
    print(f"  Final Data loss: {np.mean(data_losses[-20:]):.6f}")
    print(f"  Mean action magnitude: {np.mean([abs(agent.act(env.reset(seed=seed+e)[0])).mean() for e in range(10)]):.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1a_pinn_pretrain.yaml")
    args = parser.parse_args()
    pretrain_pinn(args.config)
