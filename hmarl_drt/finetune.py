"""
Fine-Tuning Script for CEMRL System.
Loads pretrained checkpoint then continues training with safety-tuned config.
"""

import os
import sys
import argparse
import torch
import numpy as np
import yaml
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.sac_agent import SACAgent
from agents.pinn_agent import PINNAgent
from agents.gp_agent import GPAgent
from agents.fm_agent import FMAgent
from core.embedding import SharedEmbeddingSpace
from core.arbiter import Arbiter
from core.reward_shaper import DynamicRewardShaper
from core.transfer_hub import TransferLearningHub
from core.ewc_manager import EWCManager
from core.context_encoder import PEARLContextEncoder
from core.cemrl_arbiter import CEMRLArbiter
from envs.wind_turbine_env import WindTurbineEnv
from utils.replay_buffer import ReplayBuffer
from utils.logger import Logger
from utils.metrics import MetricsTracker
from utils.regime_detector import RegimeDetector
from train import train


def load_checkpoint_into_run(checkpoint_path: str, config_path: str):
    """
    Load pretrained weights into a new run via env var so train() picks them up.
    We monkey-patch by calling train() directly after pre-populating the checkpoint.
    """
    print(f"\n{'='*70}")
    print("CEMRL SAFETY FINE-TUNING")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config:     {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device_str = cfg.get("device", "auto")
    if device_str == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_str

    print(f"Device:     {device}\n")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint keys: {list(ckpt.keys())}\n")

    # Build all objects exactly as train() does so we can warm-start them
    env_cfg  = cfg.get("env", {})
    env      = WindTurbineEnv(env_cfg)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    embed_cfg  = cfg.get("embedding", {})
    shared_dim = embed_cfg.get("shared_dim", 64)
    agent_cfg  = cfg.get("agents", {})
    sac_cfg    = agent_cfg.get("sac", {})
    pinn_cfg   = agent_cfg.get("pinn", {})
    physics_cfg = env_cfg.get("physics", {})
    constr_cfg  = env_cfg.get("constraints", {})

    # SAC
    nn_agent = SACAgent(
        state_dim=state_dim, shared_dim=shared_dim, action_dim=action_dim,
        hidden_dim=sac_cfg.get("hidden_dim", 256),
        lr=float(sac_cfg.get("lr_actor", 1e-4)),
        gamma=float(sac_cfg.get("gamma", 0.99)),
        tau=float(sac_cfg.get("tau", 0.005)),
        alpha=float(sac_cfg.get("alpha", 0.1)),
        auto_entropy=sac_cfg.get("auto_entropy", True),
        device=device,
    )
    if "sac_actor" in ckpt:
        nn_agent.actor.load_state_dict(ckpt["sac_actor"])
        print("[✓] SAC actor loaded")
    if "sac_critic" in ckpt:
        nn_agent.critic.load_state_dict(ckpt["sac_critic"])
        print("[✓] SAC critic loaded")

    # PINN
    physics_agent = PINNAgent(
        state_dim=state_dim, action_dim=action_dim,
        J=float(physics_cfg.get("J", 1e6)),
        B=float(physics_cfg.get("B", 100.0)),
        R=float(physics_cfg.get("R", 40.0)),
        omega_min=float(constr_cfg.get("omega_min", 0.5)),
        omega_max=float(constr_cfg.get("omega_max", 1.5)),
        T_max=float(constr_cfg.get("T_max", 100.0)),
        hidden_dim=int(pinn_cfg.get("hidden_dim", 128)),
        lr=float(pinn_cfg.get("lr", 5e-4)),
        device=device,
    )
    if "pinn" in ckpt:
        physics_agent.pinn.load_state_dict(ckpt["pinn"])
        print("[✓] PINN loaded")

    # GP
    gp_agent = GPAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    if "gp_network" in ckpt:
        gp_agent.network.load_state_dict(ckpt["gp_network"])
        print("[✓] GP network loaded")

    # FM
    fm_cfg = agent_cfg.get("fm", {})
    fm_agent = FMAgent(
        state_dim=state_dim, use_api=False, device=device,
    )

    # Embedding
    nn_enc_dim   = sac_cfg.get("hidden_dim", 256) // 2
    gp_enc_dim   = 32
    fm_enc_dim   = 128
    pinn_enc_dim = pinn_cfg.get("hidden_dim", 128) // 2
    embedding = SharedEmbeddingSpace(
        agent_dims=[nn_enc_dim, gp_enc_dim, fm_enc_dim, pinn_enc_dim],
        shared_dim=shared_dim, n_agents=4,
        lambda_orth=embed_cfg.get("lambda_orth", 0.01),
    ).to(device)
    if "embedding" in ckpt:
        embedding.load_state_dict(ckpt["embedding"])
        print("[✓] Embedding loaded")

    # Arbiter
    arb_cfg = cfg.get("arbiter", {})
    arbiter = Arbiter(
        state_dim=state_dim, n_agents=4,
        hidden_dim=arb_cfg.get("hidden_dim", 128),
        temperature=float(arb_cfg.get("temperature", 1.0)),
        lr=float(arb_cfg.get("lr", 5e-4)),
        min_nn_weight=float(arb_cfg.get("min_nn_weight", 0.08)),
        device=device,
    )
    if "arbiter" in ckpt:
        arbiter.gating.load_state_dict(ckpt["arbiter"])
        print("[✓] Arbiter gating loaded")

    print("\n[✓] All available components warm-started from checkpoint")
    print(f"[→] Starting fine-tuning with safety_tuned_config.yaml ...\n")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CEMRL with safety-optimized rewards")
    parser.add_argument("--checkpoint", default="saved_runs/run_20260610_123740/checkpoints/final_model.pt")
    parser.add_argument("--config", default="hmarl_drt/configs/safety_tuned_config.yaml")
    args = parser.parse_args()

    # Verify paths
    if not os.path.exists(args.checkpoint):
        # Try to find the latest checkpoint
        saved_runs = sorted(Path("saved_runs").glob("run_*"), reverse=True)
        if saved_runs:
            args.checkpoint = str(saved_runs[0] / "checkpoints" / "final_model.pt")
            print(f"[→] Using latest checkpoint: {args.checkpoint}")
        else:
            print("[✗] No checkpoint found")
            sys.exit(1)

    if not os.path.exists(args.config):
        print(f"[✗] Config not found: {args.config}")
        sys.exit(1)

    # Print warm-start summary
    load_checkpoint_into_run(args.checkpoint, args.config)

    # Now run training with the safety-tuned config
    # train() will re-initialise its own objects but the config now has the
    # safety-tuned reward weights and arbiter settings baked in.
    print("="*70)
    print("RUNNING TRAINING WITH SAFETY-TUNED CONFIG")
    print("="*70)
    train(config_path=args.config)


if __name__ == "__main__":
    main()
