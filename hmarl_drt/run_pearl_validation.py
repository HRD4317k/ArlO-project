#!/usr/bin/env python3
"""
Standalone PEARL validation script.
Loads the latest checkpoint and runs fixed validation (PEARL receives masked delta).
Does NOT interfere with the running training process.
"""

import os
import sys
import torch
import numpy as np
import yaml

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from agents.sac_agent import SACAgent
from agents.pinn_agent import PINNAgent
from agents.gp_agent import GPAgent
from agents.fm_agent import FMAgent
from core.embedding import SharedEmbeddingSpace
from core.arbiter import Arbiter
from core.context_encoder import PEARLContextEncoder
from envs.wind_turbine_env import WindTurbineEnv


def get_device(cfg_device: str = "auto") -> str:
    if cfg_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return cfg_device


def run_fixed_validation_episode(
    env, nn_agent, gp_agent, physics_agent, fm_agent,
    embedding, arbiter, context_encoder, device,
    regime_id, delta_range, deterministic=True,
):
    """
    FIXED validation: PEARL receives state with delta masked (state[5]=0.0),
    just like in the training loop. All other components use true delta.
    """
    state, _ = env.reset(options={"delta_range": delta_range})
    true_state = state.copy()  # Keep original for ground truth

    if context_encoder is not None:
        context_encoder.reset_context()

    episode_reward = 0.0
    physics_violations = 0
    regime_correct = 0
    total_steps = 0
    delta_errors = []
    rul_errors = []
    z_task_embeddings = []
    regime_predictions = []
    routing_delta = 0.0

    done = False
    while not done and total_steps < 1000:
        s_t = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            z1 = nn_agent.actor.get_embedding(s_t)
            z2 = gp_agent.get_embedding(s_t)
            z3 = fm_agent.get_embedding(s_t)
            z4 = physics_agent.get_embedding(s_t)
            z_shared, alpha = embedding([z1, z2, z3, z4])

        if deterministic:
            with torch.no_grad():
                mean, _ = nn_agent.actor.forward(s_t, z_shared)
                action = torch.tanh(mean).cpu().numpy().flatten()
        else:
            action = nn_agent.act(state, z_shared.cpu().numpy().flatten())

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_reward += reward
        if info.get("physics_violated", False):
            physics_violations += 1

        # Ground truth from env (unmasked)
        true_delta = float(next_state[5]) if len(next_state) > 5 else 0.0
        true_regime = 0 if true_delta < 0.25 else (1 if true_delta < 0.5 else (2 if true_delta < 0.75 else 3))

        # ── FIXED: PEARL gets masked delta ──
        if context_encoder is not None:
            state_pearl = state.copy()
            next_state_pearl = next_state.copy()
            if len(state_pearl) > 5:
                state_pearl[5] = 0.0
            if len(next_state_pearl) > 5:
                next_state_pearl[5] = 0.0

            context_encoder.add_transition(state_pearl, action, reward, next_state_pearl)

            if len(context_encoder.context_buffer) >= 5:
                z_task = context_encoder.forward()
                z_task_embeddings.append(z_task.detach().cpu().numpy().flatten())

                pred_regime, conf = context_encoder.classify_regime(z_task)
                regime_predictions.append(pred_regime)

                if pred_regime == true_regime:
                    regime_correct += 1

                delta_score = context_encoder.infer_degradation_score(z_task)
                routing_delta = delta_score if isinstance(delta_score, float) else delta_score.item()
                delta_errors.append(abs(routing_delta - true_delta))

        # RUL estimation (uses routing_delta from PEARL)
        rul_est = gp_agent.estimate_rul(state, delta_override=routing_delta if context_encoder else 0.0)
        true_rul = max(0.0, (1.0 - true_delta) * 1000.0)
        rul_errors.append(abs(rul_est - true_rul))

        state = next_state
        total_steps += 1

    return {
        "reward": episode_reward,
        "steps": total_steps,
        "physics_violations": physics_violations,
        "regime_accuracy": regime_correct / max(total_steps, 1),
        "delta_mae": np.mean(delta_errors) if delta_errors else 1.0,
        "rul_mae": np.mean(rul_errors) if rul_errors else 500.0,
        "z_task_embeddings": np.array(z_task_embeddings) if z_task_embeddings else None,
        "regime_predictions": regime_predictions,
    }


def compute_pearl_quality_metrics(context_encoder, z_task_embeddings, true_regimes):
    metrics = {}
    if hasattr(context_encoder, 'regime_centroids') and len(context_encoder.regime_centroids) > 1:
        centroids = list(context_encoder.regime_centroids.values())
        min_sep = float('inf')
        max_norm = 0.0
        for i, c1 in enumerate(centroids):
            c1_np = c1.cpu().numpy() if torch.is_tensor(c1) else c1
            max_norm = max(max_norm, np.linalg.norm(c1_np))
            for j, c2 in enumerate(centroids):
                if i < j:
                    c2_np = c2.cpu().numpy() if torch.is_tensor(c2) else c2
                    dist = np.linalg.norm(c1_np - c2_np)
                    min_sep = min(min_sep, dist)
        metrics["centroid_separation"] = min_sep / max(max_norm, 1e-6)
        metrics["n_centroids"] = len(centroids)
    else:
        metrics["centroid_separation"] = 0.0
        metrics["n_centroids"] = 0

    if z_task_embeddings is not None and len(z_task_embeddings) > 10 and true_regimes is not None:
        try:
            from sklearn.metrics import silhouette_score
            valid_mask = true_regimes < len(context_encoder.regime_centroids) if hasattr(context_encoder, 'regime_centroids') else np.ones(len(true_regimes), dtype=bool)
            if valid_mask.sum() > 10:
                metrics["silhouette"] = silhouette_score(z_task_embeddings[valid_mask], true_regimes[valid_mask])
            else:
                metrics["silhouette"] = 0.0
        except ImportError:
            metrics["silhouette"] = 0.0
        except Exception:
            metrics["silhouette"] = 0.0
    else:
        metrics["silhouette"] = 0.0
    return metrics


def main():
    config_path = "hmarl_drt/configs/wind_config.yaml"
    checkpoint_dir = "hmarl_drt/runs/checkpoints"

    # Find latest checkpoint
    ckpt_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")])
    if not ckpt_files:
        print("No checkpoints found!")
        return

    # Prefer latest.pt, then highest numbered episode
    if "latest.pt" in ckpt_files:
        ckpt_path = os.path.join(checkpoint_dir, "latest.pt")
    else:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_files[-1])

    print(f"Loading checkpoint: {ckpt_path}")
    print(f"Loading config: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg.get("device", "auto"))
    print(f"Device: {device}")

    # ── Recreate all components ──
    env_cfg = cfg.get("env", {})
    agent_cfg = cfg.get("agents", {})
    embed_cfg = cfg.get("embedding", {})
    shared_dim = embed_cfg.get("shared_dim", 64)
    state_dim = 8
    action_dim = 3

    # SAC
    sac_cfg = agent_cfg.get("sac", {})
    nn_agent = SACAgent(
        state_dim=state_dim, shared_dim=shared_dim, action_dim=action_dim,
        hidden_dim=sac_cfg.get("hidden_dim", 256),
        lr=float(sac_cfg.get("lr_actor", 3e-4)),
        gamma=float(sac_cfg.get("gamma", 0.99)),
        tau=float(sac_cfg.get("tau", 0.005)),
        alpha=float(sac_cfg.get("alpha", 0.2)),
        auto_entropy=sac_cfg.get("auto_entropy", True),
        device=device,
    )

    # PINN
    pinn_cfg = agent_cfg.get("pinn", {})
    physics_cfg = env_cfg.get("physics", {})
    constr_cfg = env_cfg.get("constraints", {})
    physics_agent = PINNAgent(
        state_dim=state_dim, action_dim=action_dim,
        J=float(physics_cfg.get("J", 1e6)),
        B=float(physics_cfg.get("B", 100.0)),
        R=float(physics_cfg.get("R", 40.0)),
        omega_min=float(constr_cfg.get("omega_min", 0.5)),
        omega_max=float(constr_cfg.get("omega_max", 1.5)),
        T_max=float(constr_cfg.get("T_max", 100.0)),
        hidden_dim=int(pinn_cfg.get("hidden_dim", 128)),
        lr=float(pinn_cfg.get("lr", 1e-3)),
        device=device,
    )

    # GP
    gp_agent = GPAgent(
        state_dim=state_dim, action_dim=action_dim,
        lr=float(agent_cfg.get("gp", {}).get("lr", 0.01)),
        device=device,
    )

    # FM
    fm_cfg = agent_cfg.get("fm", {})
    fm_agent = FMAgent(
        state_dim=state_dim,
        model_name=fm_cfg.get("model_name", "gpt-4o-mini"),
        temperature=float(fm_cfg.get("temperature", 0.3)),
        confidence_threshold=float(fm_cfg.get("confidence_threshold", 0.7)),
        use_api=False,  # No API calls during validation
        api_key="",
        api_timeout_s=float(fm_cfg.get("api_timeout_s", 8.0)),
        api_interval=int(fm_cfg.get("api_interval", 50)),
        device=device,
    )

    # Embedding
    def _infer_dim(agent, fallback=shared_dim):
        try:
            dummy_state = torch.zeros(1, state_dim, device=device)
            if hasattr(agent, 'actor') and hasattr(agent.actor, 'get_embedding'):
                emb = agent.actor.get_embedding(dummy_state)
            elif hasattr(agent, 'get_embedding'):
                emb = agent.get_embedding(dummy_state)
            else:
                return fallback
            return int(emb.shape[-1])
        except Exception:
            return fallback

    nn_enc_dim = _infer_dim(nn_agent)
    gp_enc_dim = _infer_dim(gp_agent)
    fm_enc_dim = _infer_dim(fm_agent)
    pinn_enc_dim = _infer_dim(physics_agent)

    embedding = SharedEmbeddingSpace(
        agent_dims=[nn_enc_dim, gp_enc_dim, fm_enc_dim, pinn_enc_dim],
        shared_dim=shared_dim, n_agents=4,
        lambda_orth=embed_cfg.get("lambda_orth", 0.01),
    ).to(device)

    # Arbiter
    arb_cfg = cfg.get("arbiter", {})
    arbiter = Arbiter(
        state_dim=state_dim, n_agents=4,
        hidden_dim=arb_cfg.get("hidden_dim", 128),
        temperature=float(arb_cfg.get("temperature", 1.0)),
        lr=float(arb_cfg.get("lr", 1e-3)),
        min_nn_weight=float(arb_cfg.get("min_nn_weight", 0.10)),
        device=device,
    )

    # PEARL Context Encoder
    cemrl_cfg = cfg.get("cemrl", {})
    pearl_cfg = cemrl_cfg.get("pearl", {})
    context_encoder = PEARLContextEncoder(
        state_dim=state_dim, action_dim=action_dim,
        z_dim=pearl_cfg.get("embedding_dim", 32),
        hidden_dim=pearl_cfg.get("hidden_dim", 128),
        context_len=pearl_cfg.get("context_window", 20),
        device=device,
    )

    # ── Load checkpoint ──
    ckpt = torch.load(ckpt_path, map_location=device)

    load_map = {
        "sac_actor": (nn_agent.actor, "sac_actor"),
        "sac_critic": (nn_agent.critic, "sac_critic"),
        "gp_network": (gp_agent.network, "gp_network"),
        "pinn": (physics_agent.pinn, "pinn"),
        "embedding": (embedding, "embedding"),
        "arbiter": (arbiter.gating, "arbiter"),
    }
    for name, (obj, key) in load_map.items():
        if key in ckpt:
            try:
                obj.load_state_dict(ckpt[key], strict=False)
                print(f"  Loaded: {name}")
            except Exception as e:
                print(f"  Skipped {name}: {e}")

    # Load PEARL
    if "pearl_context_encoder" in ckpt:
        context_encoder.load_state_dict(ckpt["pearl_context_encoder"], strict=False)
        print("  Loaded: pearl_context_encoder")
    elif "pearl_encoder" in ckpt:
        context_encoder.load_state_dict(ckpt["pearl_encoder"], strict=False)
        print("  Loaded: pearl_encoder (old format)")

    if "pearl_centroids" in ckpt:
        context_encoder.regime_centroids = ckpt["pearl_centroids"]
        print(f"  Loaded: pearl_centroids ({len(context_encoder.regime_centroids)} centroids)")

    # ── Run fixed validation ──
    regime_delta_ranges = {
        0: (0.0, 0.25),
        1: (0.25, 0.5),
        2: (0.5, 0.75),
        3: (0.75, 0.95),
    }
    n_episodes_per_regime = 5  # Fewer episodes — delta is frozen so variance is low

    all_results = {}
    all_embeddings = []
    all_true_regimes = []

    print(f"\n{'='*70}")
    print(f"  FIXED PEARL VALIDATION (delta masked for PEARL)")
    print(f"  {n_episodes_per_regime} episodes × 4 regimes = {n_episodes_per_regime * 4} total")
    print(f"{'='*70}\n")

    for regime_id, delta_range in regime_delta_ranges.items():
        regime_rewards = []
        regime_violations = []
        regime_accuracies = []
        regime_delta_maes = []
        regime_rul_maes = []

        for ep in range(n_episodes_per_regime):
            # FIX: freeze delta so true_regime stays constant for the full episode
            frozen_env_cfg = {**env_cfg, "physics": {**env_cfg.get("physics", {}), "deg_acceleration": 0.0}}
            env = WindTurbineEnv(frozen_env_cfg)
            result = run_fixed_validation_episode(
                env=env, nn_agent=nn_agent, gp_agent=gp_agent,
                physics_agent=physics_agent, fm_agent=fm_agent,
                embedding=embedding, arbiter=arbiter,
                context_encoder=context_encoder, device=device,
                regime_id=regime_id, delta_range=delta_range,
                deterministic=True,
            )
            regime_rewards.append(result["reward"])
            regime_violations.append(result["physics_violations"])
            regime_accuracies.append(result["regime_accuracy"])
            regime_delta_maes.append(result["delta_mae"])
            regime_rul_maes.append(result["rul_mae"])

            if result["z_task_embeddings"] is not None:
                all_embeddings.append(result["z_task_embeddings"])
                n_steps = len(result["z_task_embeddings"])
                all_true_regimes.extend([regime_id] * n_steps)

        # Collect raw prediction distribution to expose collapse
        all_preds_this_regime = []
        for res_preds in [result["regime_predictions"]]:
            all_preds_this_regime.extend(res_preds)
        pred_dist = {r: 0 for r in range(4)}
        for p in all_preds_this_regime:
            if p is not None:
                pred_dist[int(p)] = pred_dist.get(int(p), 0) + 1
        total_preds = max(sum(pred_dist.values()), 1)

        all_results[f"regime_{regime_id}"] = {
            "reward_mean": np.mean(regime_rewards),
            "reward_std": np.std(regime_rewards),
            "violations_mean": np.mean(regime_violations),
            "regime_accuracy": np.mean(regime_accuracies),
            "delta_mae": np.mean(regime_delta_maes),
            "rul_mae": np.mean(regime_rul_maes),
            "pred_dist": pred_dist,
        }

        print(f"  Regime {regime_id} (δ∈[{delta_range[0]:.2f},{delta_range[1]:.2f}], delta FROZEN):")
        print(f"    R={all_results[f'regime_{regime_id}']['reward_mean']:7.1f} ± {all_results[f'regime_{regime_id}']['reward_std']:.1f}")
        print(f"    PEARL Accuracy: {all_results[f'regime_{regime_id}']['regime_accuracy']*100:.1f}%  (correct={regime_id})")
        print(f"    Prediction distribution: " + " | ".join(f"R{r}={pred_dist[r]*100//total_preds}%" for r in range(4)))
        print(f"    PEARL Delta MAE: {all_results[f'regime_{regime_id}']['delta_mae']:.4f}")
        print()

    # Aggregate
    all_rewards = [r["reward_mean"] for r in all_results.values()]
    all_accuracies = [r["regime_accuracy"] for r in all_results.values()]
    all_delta_maes = [r["delta_mae"] for r in all_results.values()]

    z_task_all = np.concatenate(all_embeddings) if all_embeddings else None
    true_regimes_all = np.array(all_true_regimes) if all_true_regimes else None

    pearl_metrics = compute_pearl_quality_metrics(context_encoder, z_task_all, true_regimes_all)

    print(f"{'='*70}")
    print(f"  AGGREGATED RESULTS")
    print(f"{'='*70}")
    print(f"  Mean Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"  Mean PEARL Regime Accuracy: {np.mean(all_accuracies)*100:.1f}%")
    print(f"  Mean PEARL Delta MAE: {np.mean(all_delta_maes):.4f}")
    print(f"  Silhouette Score: {pearl_metrics.get('silhouette', 'N/A'):.3f}")
    print(f"  Centroid Separation: {pearl_metrics.get('centroid_separation', 'N/A'):.3f}")
    print(f"  Number of Centroids: {pearl_metrics.get('n_centroids', 0)}")
    print(f"{'='*70}")

    # Compare with old (leaky) validation
    print(f"\n{'='*70}")
    print(f"  COMPARISON: Leaky vs Fixed Validation")
    print(f"{'='*70}")
    print(f"  The training loop's validation (every 50 eps) uses LEAKY add_transition")
    print(f"  (PEARL sees true delta). This script uses FIXED add_transition")
    print(f"  (PEARL sees state[5]=0.0).")
    print(f"")
    print(f"  If fixed accuracy is HIGHER than leaky accuracy, PEARL learned")
    print(f"  genuine physics-based regime separation despite the leak bug.")
    print(f"  If fixed accuracy is LOWER, the leak was masking PEARL's failure.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()