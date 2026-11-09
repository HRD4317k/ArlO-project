"""
Phase 1d: SAC Agent Pre-Training (Regime-Specific)
===================================================
Trains SAC with regime-specific heads (trunk shared).

Data:    Regime-specific replay buffers (balanced)
Loss:    L_SAC = L_actor + L_critic + alpha * H(pi)
Target:  Stable Q-values, no divergence
Episodes: 100 per regime (400 total)

Usage:
    python pretrain_sac.py --config configs/phase1d_sac_pretrain.yaml
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))

from envs.wind_turbine_env import WindTurbineEnv
from agents.sac_agent import SACAgent
from core.embedding import SharedEmbeddingSpace
from utils.replay_buffer import ReplayBuffer


def pretrain_sac(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = cfg.get("env", {})
    env = WindTurbineEnv(env_cfg)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    shared_dim = cfg.get("embedding", {}).get("shared_dim", 64)

    sac_cfg = cfg.get("agents", {}).get("sac", {})
    agent = SACAgent(
        state_dim=state_dim,
        shared_dim=shared_dim,
        action_dim=action_dim,
        hidden_dim=sac_cfg.get("hidden_dim", 256),
        lr=float(sac_cfg.get("lr_actor", 3e-4)),
        gamma=float(sac_cfg.get("gamma", 0.99)),
        tau=float(sac_cfg.get("tau", 0.005)),
        alpha=float(sac_cfg.get("alpha", 0.2)),
        auto_entropy=sac_cfg.get("auto_entropy", True),
        device=device,
    )

    # Shared embedding space
    embedding = SharedEmbeddingSpace(
        agent_dims=[sac_cfg.get("hidden_dim", 256) // 2, 32, 128, 64],
        shared_dim=shared_dim,
        n_agents=4,
        lambda_orth=cfg.get("embedding", {}).get("lambda_orth", 0.01),
    ).to(device)
    embed_opt = torch.optim.Adam(embedding.parameters(), lr=1e-3)

    n_episodes = cfg.get("training", {}).get("n_episodes", 400)
    batch_size = cfg.get("training", {}).get("batch_size", 256)
    update_freq = cfg.get("training", {}).get("update_freq", 4)
    warmup_steps = cfg.get("training", {}).get("warmup_steps", 500)
    seed = cfg.get("seed", 42)
    log_dir = cfg.get("logging", {}).get("log_dir", "runs/phase1d_sac/")
    os.makedirs(log_dir, exist_ok=True)
    save_freq = cfg.get("training", {}).get("save_freq", 100)

    buffer = ReplayBuffer(100000)

    print(f"\n{'='*60}")
    print(f"  Phase 1d: SAC Agent Pre-Training")
    print(f"  Episodes: {n_episodes} (100 per regime) | Device: {device}")
    print(f"{'='*60}\n")

    total_steps = 0
    regime_ep_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        steps = 0
        done = False

        # Determine regime from delta
        delta = float(state[5]) if len(state) > 5 else 0.0
        regime = 0 if delta < 0.25 else (1 if delta < 0.5 else (2 if delta < 0.75 else 3))
        regime_ep_counts[regime] += 1

        while not done and steps < 1000:
            s_t = torch.FloatTensor(state).unsqueeze(0).to(device)

            # Get shared embedding
            with torch.no_grad():
                z1 = agent.actor.get_embedding(s_t)  # [1, 128]
                # Placeholder embeddings matching agent_dims from config: [128, 32, 128, 64]
                z2 = torch.zeros(1, 32, device=device)   # GP encoder dim
                z3 = torch.zeros(1, 128, device=device)  # FM encoder dim
                z4 = torch.zeros(1, 64, device=device)   # PINN encoder dim
                z_shared, _ = embedding([z1, z2, z3, z4])

            # SAC action
            action = agent.act(state, z_shared.squeeze(0), deterministic=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, done)
            ep_reward += reward
            state = next_state
            steps += 1
            total_steps += 1

            # SAC update
            if len(buffer) > batch_size and total_steps % update_freq == 0 and total_steps > warmup_steps:
                batch = buffer.sample(batch_size)
                s_batch = torch.FloatTensor(batch["states"]).to(device)
                a_batch = torch.FloatTensor(batch["actions"]).to(device)
                s_next_batch = torch.FloatTensor(batch["next_states"]).to(device)

                with torch.no_grad():
                    z1_b = agent.actor.get_embedding(s_batch)
                    z2_b = torch.zeros(batch_size, 32, device=device)
                    z3_b = torch.zeros(batch_size, 128, device=device)
                    z4_b = torch.zeros(batch_size, 64, device=device)
                    z_shared_b, _ = embedding([z1_b, z2_b, z3_b, z4_b])
                    z1_n = agent.actor.get_embedding(s_next_batch)
                    z2_n = torch.zeros(batch_size, 32, device=device)
                    z3_n = torch.zeros(batch_size, 128, device=device)
                    z4_n = torch.zeros(batch_size, 64, device=device)
                    z_shared_next, _ = embedding([z1_n, z2_n, z3_n, z4_n])

                sac_losses = agent.update(batch, z_shared_b, z_shared_next)

                # Update embedding alignment
                align_loss = embedding.alignment_loss([z1_b.detach(), z2_b.detach(), z3_b.detach(), z4_b.detach()])
                embed_opt.zero_grad()
                align_loss.backward()
                embed_opt.step()

        if (ep + 1) % 10 == 0:
            print(f"  Ep {ep+1:4d}/{n_episodes} | Reward: {ep_reward:.1f} | Steps: {steps} | Regime: {regime}")

        if (ep + 1) % save_freq == 0:
            ckpt_path = os.path.join(log_dir, f"sac_ep{ep+1:04d}.pt")
            torch.save({
                "sac_actor_state_dict": agent.actor.state_dict(),
                "sac_critic_state_dict": agent.critic.state_dict(),
                "embedding_state_dict": embedding.state_dict(),
            }, ckpt_path)
            print(f"  [Saved] {ckpt_path}")

    final_path = os.path.join(log_dir, "sac_final.pt")
    torch.save({
        "sac_actor_state_dict": agent.actor.state_dict(),
        "sac_critic_state_dict": agent.critic.state_dict(),
        "embedding_state_dict": embedding.state_dict(),
    }, final_path)
    print(f"\n  [Final] Saved to {final_path}")
    print(f"  Regime distribution: {regime_ep_counts}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1d_sac_pretrain.yaml")
    args = parser.parse_args()
    pretrain_sac(args.config)
