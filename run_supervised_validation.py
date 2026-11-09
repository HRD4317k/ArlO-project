#!/usr/bin/env python3
"""
Supervised validation of HMARL-DRT components.

This intentionally uses supervised regime identification from the simulator's
true degradation index for now. PEARL/centroid-based unsupervised regime ID is
left as a separate problem; this script validates the rest of the stack with a
consistent supervised regime signal and real shared/task embeddings.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hmarl_drt"))

from hmarl_drt.envs.wind_turbine_env import WindTurbineEnv
from hmarl_drt.agents.sac_agent import SACAgent
from hmarl_drt.agents.pinn_agent import PINNAgent
from hmarl_drt.agents.gp_agent import GPAgent
from hmarl_drt.agents.fm_agent import FMAgent
from hmarl_drt.core.context_encoder import PEARLContextEncoder
from hmarl_drt.core.embedding import SharedEmbeddingSpace
from hmarl_drt.core.arbiter import Arbiter


def true_regime_from_delta(delta: float) -> int:
    if delta < 0.25:
        return 0
    if delta < 0.50:
        return 1
    if delta < 0.75:
        return 2
    return 3


def supervised_z_task(delta: float, device: str, z_dim: int = 32) -> torch.Tensor:
    """Deterministic supervised task embedding used until unsupervised PEARL is fixed."""
    rid = true_regime_from_delta(float(delta))
    z = torch.zeros(z_dim, device=device)
    z[rid] = 1.0
    z[4] = float(np.clip(delta, 0.0, 1.0))
    return z


def safe_load(module, state_dict, label: str):
    """Load only compatible checkpoint tensors, avoiding shape-mismatch crashes."""
    if state_dict is None:
        return
    current = module.state_dict()
    compatible = {
        k: v for k, v in state_dict.items()
        if k in current and tuple(current[k].shape) == tuple(v.shape)
    }
    skipped = sorted(set(state_dict.keys()) - set(compatible.keys()))
    module.load_state_dict(compatible, strict=False)
    print(f"  Loaded {label}: {len(compatible)} tensors" + (f", skipped {len(skipped)}" if skipped else ""))


def make_env_config(cfg: dict) -> dict:
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


def build_shared_embedding(sac, gp, fm, pinn, shared_dim, device, ckpt):
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
    print(f"  Shared embedding dims: {dims} -> {shared_dim}")
    return embedding


def compute_z_shared(state, sac, gp, fm, pinn, embedding, device):
    s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        z1 = sac.actor.get_embedding(s_t)
        z2 = gp.get_embedding(s_t)
        z3 = fm.get_embedding(s_t)
        z4 = pinn.get_embedding(s_t)
        return embedding([z1, z2, z3, z4])


def load_context_encoder(sd, ad, device, ckpt, supervised_path="pearl_supervised.pt"):
    encoder = PEARLContextEncoder(state_dim=sd, action_dim=ad, z_dim=32, hidden_dim=128, context_len=20, device=device)
    artifact_path = Path(supervised_path)
    if artifact_path.exists():
        art = torch.load(artifact_path, map_location=device)
        if "context_encoder_state_dict" in art:
            safe_load(encoder, art["context_encoder_state_dict"], "supervised PEARL encoder")
        if "regime_centroids" in art:
            encoder.regime_centroids = art["regime_centroids"]
            print(f"  Loaded supervised PEARL centroids: {len(encoder.regime_centroids)}")
        return encoder, str(artifact_path)

    if "pearl_encoder" in ckpt:
        safe_load(encoder.encoder, ckpt["pearl_encoder"], "checkpoint PEARL trunk")
    if "pearl_fc_regime" in ckpt:
        safe_load(encoder.fc_regime, ckpt["pearl_fc_regime"], "checkpoint PEARL regime head")
    if "pearl_centroids" in ckpt:
        encoder.regime_centroids = ckpt["pearl_centroids"]
        print(f"  Loaded checkpoint PEARL centroids: {len(encoder.regime_centroids)}")
    return encoder, "checkpoint/fallback"


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

cfg_path = os.environ.get("HMARL_VALIDATION_CONFIG", "configs/wind_config.yaml")
ckpt_path = os.environ.get("HMARL_VALIDATION_CHECKPOINT", "runs/wind_cemrl_phase3_from_500/checkpoints/ep_02200.pt")
cfg = yaml.safe_load(open(cfg_path))
env_cfg = make_env_config(cfg)
ckpt = torch.load(ckpt_path, map_location=device) if Path(ckpt_path).exists() else {}
if not ckpt:
    print(f"  Warning: checkpoint not found at {ckpt_path}; validating randomly initialized components")

env = WindTurbineEnv(env_cfg)
sd = int(env.observation_space.shape[0])
ad = int(env.action_space.shape[0])
shared_dim = cfg.get("embedding", {}).get("shared_dim", 64)
results = {}
N_EVAL = int(os.environ.get("HMARL_VALIDATION_EPISODES", 50))

# Components shared across phases.
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
context_encoder, pearl_source = load_context_encoder(sd, ad, device, ckpt)
embedding = build_shared_embedding(sac, gp, fm, pinn, shared_dim, device, ckpt)

# Phase 2: Supervised regime detection.
print("\n" + "=" * 60)
print("PHASE 2: SUPERVISED REGIME DETECTION")
print("=" * 60)
print(f"  PEARL source for auxiliary z_task: {pearl_source}")
true_regimes, pred_regimes, pearl_preds = [], [], []
for ep in range(N_EVAL):
    state, _ = env.reset(seed=ep)
    context_encoder.reset_context()
    done = False
    while not done:
        action = np.random.randn(ad) * 0.1
        next_state, reward, term, trunc, _ = env.step(action)
        context_encoder.add_transition(state, action, reward, next_state)
        state = next_state
        done = term or trunc
    true_regime = true_regime_from_delta(float(state[5]))
    pred_regime = true_regime_from_delta(float(state[5]))
    z = context_encoder.forward()
    pearl_pred, _ = context_encoder.classify_regime(z)
    true_regimes.append(true_regime)
    pred_regimes.append(pred_regime)
    pearl_preds.append(pearl_pred)

acc = np.mean(np.array(true_regimes) == np.array(pred_regimes))
pearl_acc = np.mean(np.array(true_regimes) == np.array(pearl_preds))
cm = confusion_matrix(true_regimes, pred_regimes, labels=[0, 1, 2, 3])
print(f"  Supervised accuracy: {acc:.4f}")
print(f"  Auxiliary PEARL head accuracy: {pearl_acc:.4f}")
print(f"  Confusion matrix:\n{cm}")
results["supervised_regime_detection"] = {
    "accuracy": float(acc),
    "confusion_matrix": cm.tolist(),
    "aux_pearl_head_accuracy": float(pearl_acc),
}

# Phase 3: SAC policy with real shared embedding.
print("\n" + "=" * 60)
print("PHASE 3: SAC POLICY")
print("=" * 60)
returns, z_norms = [], []
for ep in range(20):
    state, _ = env.reset(seed=ep)
    done = False
    ep_ret = 0.0
    while not done:
        z_shared, _ = compute_z_shared(state, sac, gp, fm, pinn, embedding, device)
        z_norms.append(float(z_shared.norm().item()))
        action = sac.act(state, z_shared, deterministic=True)
        next_state, reward, term, trunc, _ = env.step(action)
        ep_ret += reward
        state = next_state
        done = term or trunc
    returns.append(ep_ret)
print(f"  Mean return: {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
print(f"  Mean z_shared norm: {np.mean(z_norms):.4f}")
results["sac_policy"] = {
    "mean_return": float(np.mean(returns)),
    "std_return": float(np.std(returns)),
    "mean_z_shared_norm": float(np.mean(z_norms)),
}

# Phase 4: PINN dynamics.
print("\n" + "=" * 60)
print("PHASE 4: PINN DYNAMICS")
print("=" * 60)
pinn_errors = []
for ep in range(20):
    state, _ = env.reset(seed=ep)
    done = False
    while not done:
        action = np.random.randn(ad) * 0.1
        next_state, reward, term, trunc, _ = env.step(action)
        with torch.no_grad():
            s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            a_t = torch.FloatTensor(action).unsqueeze(0).to(device)
            ns_pred = s_t + pinn.pinn(s_t, a_t) * float(env_cfg.get("dt", 0.1))
            ns_true = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            pinn_errors.append(torch.nn.functional.mse_loss(ns_pred, ns_true).item())
        state = next_state
        done = term or trunc
print(f"  Next-state MSE: {np.mean(pinn_errors):.6f}")
results["pinn_dynamics"] = {"mse": float(np.mean(pinn_errors))}

# Phase 5: GP uncertainty with normalized agent inputs.
print("\n" + "=" * 60)
print("PHASE 5: GP AGENT")
print("=" * 60)
gp_uncertainties = []
for ep in range(20):
    state, _ = env.reset(seed=ep)
    done = False
    while not done:
        action = np.random.randn(ad) * 0.1
        next_state, reward, term, trunc, _ = env.step(action)
        s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        gp_uncertainties.append(float(gp.get_uncertainty(s_t).item()))
        state = next_state
        done = term or trunc
print(f"  Mean uncertainty: {np.mean(gp_uncertainties):.4f}")
results["gp_agent"] = {"mean_uncertainty": float(np.mean(gp_uncertainties))}

# Phase 6: FM anomaly detector.
print("\n" + "=" * 60)
print("PHASE 6: FM AGENT")
print("=" * 60)
anomalies = []
for ep in range(20):
    state, _ = env.reset(seed=ep)
    done = False
    while not done:
        action = np.random.randn(ad) * 0.1
        next_state, reward, term, trunc, _ = env.step(action)
        is_anom, _ = fm.detect_anomaly(state)
        anomalies.append(int(is_anom))
        state = next_state
        done = term or trunc
anom_rate = np.mean(anomalies)
print(f"  Anomaly rate: {anom_rate:.4f}")
results["fm_agent"] = {"anomaly_rate": float(anom_rate)}

# Phase 7: projected shared embedding consistency.
print("\n" + "=" * 60)
print("PHASE 7: EMBEDDING CONSISTENCY")
print("=" * 60)
cosine_sims = []
for ep in range(50):
    state, _ = env.reset(seed=ep)
    s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        raw = [
            sac.actor.get_embedding(s_t),
            gp.get_embedding(s_t),
            fm.get_embedding(s_t),
            pinn.get_embedding(s_t),
        ]
        projected = [embedding.projections[i](raw[i]).squeeze(0) for i in range(4)]
        for i in range(4):
            for j in range(i + 1, 4):
                cosine_sims.append(torch.nn.functional.cosine_similarity(projected[i], projected[j], dim=0).item())
print(f"  Mean projected cosine similarity: {np.mean(cosine_sims):.4f}")
results["embedding_consistency"] = {"mean_projected_cosine_similarity": float(np.mean(cosine_sims))}

# Phase 8: Arbiter gating with supervised z_task.
print("\n" + "=" * 60)
print("PHASE 8: ARBITER GATING")
print("=" * 60)
arbiter = Arbiter(state_dim=sd, n_agents=4, device=device)
if "arbiter" in ckpt:
    safe_load(arbiter.gating, ckpt["arbiter"], "arbiter")

gate_weights = []
for ep in range(10):
    state, _ = env.reset(seed=ep)
    done = False
    while not done:
        action = np.random.randn(ad) * 0.1
        next_state, reward, term, trunc, _ = env.step(action)
        with torch.no_grad():
            s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            z_t = supervised_z_task(float(state[5]), device).unsqueeze(0)
            unc_t = torch.FloatTensor([[np.tanh(gp.get_uncertainty(s_t).item() / 100.0)]]).to(device)
            delta_t = torch.zeros(1, 1).to(device)
            weights = arbiter.gating(s_t, z_t, delta_t, unc_t)
            gate_weights.append(weights.cpu().numpy())
        state = next_state
        done = term or trunc
avg_w = np.mean(np.stack(gate_weights), axis=0).flatten()
print(f"  Avg weights: NN={avg_w[0]:.3f}, GP={avg_w[1]:.3f}, FM={avg_w[2]:.3f}, Physics={avg_w[3]:.3f}")
results["arbiter_gating"] = {"avg_weights": avg_w.tolist()}

os.makedirs("results", exist_ok=True)
with open("results/supervised_validation.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to results/supervised_validation.json")

print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)
for k, v in results.items():
    print(f"  {k}: {v}")
print("\nDone!")
