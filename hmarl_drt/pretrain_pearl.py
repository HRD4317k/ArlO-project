"""
Phase 1c: PEARL Context Encoder Pre-Training (UNSUPERVISED)
============================================================
Trains the PEARL context encoder using UNSUPERVISED dynamics prediction.

Loss: L_PEARL = L_dynamics + 0.05 * L_entropy + 1e-3 * KL
  - L_dynamics: Predict next_state from (z_task, state, action) — NO delta labels
  - L_entropy: Cluster entropy minimization + diversity penalty
  - KL: Variational regularization

Target: Dynamics loss < 0.05, cluster separation > 0.3
Episodes: 500 (context sequences of length H=20)

NOTE: This is UNSUPERVISED — no true delta or regime labels used during training.
      The encoder learns to cluster by observing dynamics differences across regimes.

Usage:
    python pretrain_pearl.py --config configs/phase1c_pearl_pretrain.yaml
"""

import os
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))

from envs.wind_turbine_env import WindTurbineEnv
from core.context_encoder import PEARLContextEncoder


def pretrain_pearl(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = cfg.get("env", {})
    env = WindTurbineEnv(env_cfg)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    pearl_cfg = cfg.get("cemrl", {}).get("pearl", {})
    context_encoder = PEARLContextEncoder(
        state_dim=state_dim,
        action_dim=action_dim,
        z_dim=pearl_cfg.get("embedding_dim", 32),
        hidden_dim=pearl_cfg.get("hidden_dim", 128),
        context_len=pearl_cfg.get("context_window", 20),
        device=device,
    )
    optimizer = torch.optim.Adam(
        context_encoder.parameters(),
        lr=float(pearl_cfg.get("lr", 3e-4)),
    )

    n_episodes = cfg.get("training", {}).get("n_episodes", 500)
    seed = cfg.get("seed", 42)
    log_dir = cfg.get("logging", {}).get("log_dir", "runs/phase1c_pearl/")
    os.makedirs(log_dir, exist_ok=True)
    save_freq = cfg.get("training", {}).get("save_freq", 100)

    print(f"\n{'='*60}")
    print(f"  Phase 1c: PEARL Context Encoder Pre-Training (UNSUPERVISED)")
    print(f"  Episodes: {n_episodes} | Device: {device}")
    print(f"  Loss: dynamics_prediction + cluster_entropy + KL")
    print(f"{'='*60}\n")

    context_buffer = deque(maxlen=pearl_cfg.get("context_window", 20))
    dynamics_losses = []
    entropy_losses = []
    kl_losses = []

    # Use PINN physics-based actions for meaningful state transitions
    from agents.pinn_agent import PINNAgent
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

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        context_buffer.clear()
        context_encoder.reset_context()
        ep_dynamics_loss = 0.0
        ep_entropy_loss = 0.0
        ep_kl_loss = 0.0
        steps = 0
        done = False

        while not done and steps < 1000:
            # Use PINN physics-based actions for meaningful dynamics
            action = pinn_agent.act(state)
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Add transition to context (MASKED state — delta channel = 0)
            # This is critical: PEARL must NOT see true delta during training
            masked_state = np.array(state, copy=True)
            masked_state[5] = 0.0  # Mask delta channel
            masked_next_state = np.array(next_state, copy=True)
            masked_next_state[5] = 0.0

            context_encoder.add_transition(masked_state, action, 0.0, masked_next_state)

            # Every 5 steps, compute UNSUPERVISED PEARL loss
            if steps % 5 == 0 and len(context_encoder.context_buffer) >= 5:
                z_task = context_encoder.forward()

                # Prepare tensors for dynamics prediction loss
                s_t = torch.FloatTensor(masked_state).unsqueeze(0).to(device)
                a_t = torch.FloatTensor(action).unsqueeze(0).to(device)
                ns_t = torch.FloatTensor(masked_next_state).unsqueeze(0).to(device)

                # UNSUPERVISED losses — NO delta labels, NO regime labels
                dynamics_loss = context_encoder.dynamics_prediction_loss(s_t, a_t, ns_t)
                entropy_loss = context_encoder.cluster_entropy_loss()
                kl = context_encoder.kl_loss()

                # Total loss: dynamics + entropy + KL
                loss = dynamics_loss + 0.05 * entropy_loss + 1e-3 * kl

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(context_encoder.parameters(), 1.0)
                optimizer.step()

                ep_dynamics_loss += dynamics_loss.item()
                ep_entropy_loss += entropy_loss.item()
                ep_kl_loss += kl.item()

            state = next_state
            steps += 1

        dynamics_losses.append(ep_dynamics_loss / max(steps // 5, 1))
        entropy_losses.append(ep_entropy_loss / max(steps // 5, 1))
        kl_losses.append(ep_kl_loss / max(steps // 5, 1))

        if (ep + 1) % 25 == 0:
            avg_dyn = np.mean(dynamics_losses[-25:])
            avg_ent = np.mean(entropy_losses[-25:])
            avg_kl = np.mean(kl_losses[-25:])
            n_regimes = len(context_encoder.regime_centroids) if hasattr(context_encoder, 'regime_centroids') else 0
            print(f"  Ep {ep+1:4d}/{n_episodes} | Dyn: {avg_dyn:.4f} | Ent: {avg_ent:.3f} | KL: {avg_kl:.4f} | Centroids: {n_regimes}")

        if (ep + 1) % save_freq == 0:
            ckpt_path = os.path.join(log_dir, f"pearl_ep{ep+1:04d}.pt")
            torch.save({
                "pearl_context_encoder": context_encoder.state_dict(),
                "pearl_centroids": context_encoder.regime_centroids,
                "episode": ep + 1,
                "dynamics_loss": np.mean(dynamics_losses[-25:]),
                "entropy_loss": np.mean(entropy_losses[-25:]),
            }, ckpt_path)
            print(f"  [Saved] {ckpt_path}")

    final_path = os.path.join(log_dir, "pearl_final.pt")
    torch.save({
        "pearl_context_encoder": context_encoder.state_dict(),
        "pearl_centroids": context_encoder.regime_centroids,
        "episode": n_episodes,
        "final_dynamics_loss": np.mean(dynamics_losses[-50:]),
        "final_entropy_loss": np.mean(entropy_losses[-50:]),
    }, final_path)
    print(f"\n  [Final] Saved to {final_path}")
    print(f"  Final dynamics loss: {np.mean(dynamics_losses[-50:]):.4f}")
    print(f"  Final entropy loss: {np.mean(entropy_losses[-50:]):.3f}")
    print(f"  Regime centroids: {len(context_encoder.regime_centroids)}")
    print(f"\n  NOTE: PEARL trained UNSUPERVISED — no delta/regime labels used.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1c_pearl_pretrain.yaml")
    args = parser.parse_args()
    pretrain_pearl(args.config)
