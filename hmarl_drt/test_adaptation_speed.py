"""
Adaptation Speed Test — threshold-based regime detection (no PEARL).

Measures how quickly the system adapts to regime transitions using only
the threshold-based RegimeDetector (hysteresis + dwell) for regime ID,
and the standard Arbiter (true delta for routing).

Metrics per transition:
  - Detection latency: steps for RegimeDetector to confirm new regime
  - Weight shift: how arbiter gating rebalances after transition
  - Reward recovery: pre vs post-transition reward
  - Violation delta: change in physics violations

Usage:
    python3 hmarl_drt/test_adaptation_speed.py
"""

import os, sys, time, json
import numpy as np
import torch
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "hmarl_drt"))

from envs.wind_turbine_env import WindTurbineEnv
from agents.sac_agent import SACAgent
from agents.pinn_agent import PINNAgent
from agents.gp_agent import GPAgent
from agents.fm_agent import FMAgent
from core.embedding import SharedEmbeddingSpace
from core.arbiter import Arbiter
from core.constraint_projector import HardConstraintProjector
from utils.regime_detector import RegimeDetector

CKPT_PATH = "hmarl_drt/runs/checkpoints/latest.pt"
CFG_PATH  = "hmarl_drt/configs/wind_config.yaml"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
OUT_PATH  = "hmarl_drt/runs/adaptation_speed_results.json"

REGIMES = {
    0: (0.00, 0.25, "healthy"),
    1: (0.25, 0.50, "degrading"),
    2: (0.50, 0.75, "critical"),
    3: (0.75, 0.95, "emergency"),
}
TRANSITIONS = [(0, 1), (1, 2), (2, 3), (0, 3)]


def load_components():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg.get("environment", cfg)
    env = WindTurbineEnv(env_cfg)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    sac_cfg = cfg.get("agents", {}).get("sac", {})
    shared_dim = 64
    nn_agent = SACAgent(state_dim=state_dim, shared_dim=shared_dim,
                        action_dim=action_dim,
                        hidden_dim=sac_cfg.get("hidden_dim", 256))
    gp_agent = GPAgent(state_dim=state_dim, action_dim=action_dim)
    physics_agent = PINNAgent(state_dim=state_dim, action_dim=action_dim)
    fm_agent = FMAgent(state_dim=state_dim, action_dim=action_dim)
    embedding = SharedEmbeddingSpace(agent_dim=128, shared_dim=64)
    arbiter = Arbiter(state_dim=state_dim, n_agents=4)
    projector = HardConstraintProjector()
    detector = RegimeDetector(n_hysteresis=10, dwell_steps=200)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    load_map = {
        "sac_actor": (nn_agent.actor, "sac_actor"),
        "sac_critic": (nn_agent.critic, "sac_critic"),
        "gp_network": (gp_agent.network, "gp_network"),
        "pinn": (physics_agent.pinn, "pinn"),
        "embedding": (embedding, "embedding"),
        "arbiter": (arbiter.gating, "arbiter"),
    }
    for name, (obj, key) in load_map.items():
        if key in ckpt and hasattr(obj, "load_state_dict"):
            obj.load_state_dict(ckpt[key], strict=False)

    return env, nn_agent, gp_agent, physics_agent, fm_agent, embedding, arbiter, projector, detector


def run_test(env, nn_agent, gp_agent, physics_agent, fm_agent,
             embedding, arbiter, projector, detector):
    results = {}

    for from_r, to_r in TRANSITIONS:
        from_name = REGIMES[from_r][2]
        to_name = REGIMES[to_r][2]
        print(f"\n{'='*60}")
        print(f"  Transition: {from_name} (R{from_r}) → {to_name} (R{to_r})")
        print(f"{'='*60}")

        state, _ = env.reset(options={"delta_range": (REGIMES[from_r][0], REGIMES[from_r][1])})
        # Fresh detector per transition
        detector.__init__(n_hysteresis=10, dwell_steps=200)

        step_log = []
        done = False
        t = 0
        detection_step = None

        while not done and t < 1000:
            # ── Force regime transition at step 200 ──
            if t == 200:
                target_delta = np.random.uniform(REGIMES[to_r][0], REGIMES[to_r][1])
                env.state[5] = target_delta

            true_delta = float(env.state[5])

            # RegimeDetector update (uses true delta)
            regime_name, regime_id, transition_confirmed = detector.detect(delta_t=true_delta)
            if transition_confirmed and detection_step is None:
                detection_step = t

            s_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                z1 = nn_agent.actor.get_embedding(s_t)
                z2 = gp_agent.get_embedding(s_t)
                z3 = fm_agent.get_embedding(s_t)
                z4 = physics_agent.get_embedding(s_t)
                z_shared, alpha = embedding([z1, z2, z3, z4])

            proposals = {
                "nn": nn_agent.act(state, z_shared),
                "gp": gp_agent.act(state),
                "fm": fm_agent.act(state),
                "physics": physics_agent.act(state),
            }

            # Standard Arbiter with true delta (no PEARL)
            uncertainty = gp_agent.get_uncertainty_np(state)
            uncertainty_norm = float(np.tanh(uncertainty / 100.0))
            is_anomaly, _ = fm_agent.detect_anomaly(state, routing_delta=true_delta)

            action, weights = arbiter.select_action(
                state, proposals,
                {"low": np.full(3, -1.0), "high": np.full(3, 1.0)},
                delta=true_delta,
                uncertainty=uncertainty_norm,
                emergency=is_anomaly,
                return_weights=True,
            )
            action = projector.project(action, regime_id, state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            step_log.append({
                "step": t,
                "true_delta": true_delta,
                "true_regime": int(true_delta / 0.25) if true_delta < 0.95 else 3,
                "detected_regime": regime_id,
                "detected_name": regime_name,
                "reward": reward,
                "weights": weights.tolist() if isinstance(weights, np.ndarray) else list(weights),
                "physics_violated": info.get("physics_violated", False),
            })

            state = next_state
            t += 1

        # ── Compute metrics ──
        pre = [s for s in step_log if s["step"] < 200]
        post = [s for s in step_log if s["step"] >= 200]

        detection_latency = (detection_step - 200) if detection_step else None

        # Weight shift: L2 norm between avg weights 50 steps before vs 100 steps after
        pre_w = np.mean([s["weights"] for s in pre[-50:]], axis=0) if pre else np.ones(4)/4
        post_w = np.mean([s["weights"] for s in post[:100]], axis=0) if post else np.ones(4)/4
        weight_shift = float(np.linalg.norm(post_w - pre_w))

        # Reward
        pre_r = np.mean([s["reward"] for s in pre[-50:]]) if pre else 0.0
        post_r_early = np.mean([s["reward"] for s in post[:50]]) if post else 0.0
        post_r_late = np.mean([s["reward"] for s in post[-100:]]) if len(post) >= 100 else 0.0

        # Violations
        pre_v = sum(1 for s in pre[-50:] if s["physics_violated"]) / 50.0 if pre else 0.0
        post_v = sum(1 for s in post[:100] if s["physics_violated"]) / 100.0 if post else 0.0

        # Regime detection accuracy after transition
        correct = sum(1 for s in post if s["detected_regime"] == s["true_regime"])
        acc = correct / max(len(post), 1)

        result = {
            "from_regime": from_r, "to_regime": to_r,
            "from_name": from_name, "to_name": to_name,
            "detection_latency_steps": detection_latency,
            "detected": detection_step is not None,
            "weight_shift_magnitude": round(weight_shift, 4),
            "pre_reward": round(pre_r, 2),
            "post_reward_early": round(post_r_early, 2),
            "post_reward_late": round(post_r_late, 2),
            "reward_delta": round(post_r_late - pre_r, 2),
            "pre_violation_rate": round(pre_v, 3),
            "post_violation_rate": round(post_v, 3),
            "regime_detection_accuracy": round(acc, 3),
            "n_steps_after": len(post),
        }
        results[f"R{from_r}→R{to_r}"] = result

        det_str = f"{detection_latency}s" if detection_step else "FAIL"
        print(f"    Detection:        {det_str}")
        print(f"    Weight shift:     {weight_shift:.4f}")
        print(f"    Reward:           {pre_r:.1f} → early={post_r_early:.1f} → late={post_r_late:.1f}")
        print(f"    Violations:       {pre_v:.1%} → {post_v:.1%}")
        print(f"    Regime accuracy:  {acc:.1%}")

    return results


def main():
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CKPT_PATH}")
    print("Loading components (no PEARL)...")
    env, nn_agent, gp_agent, physics_agent, fm_agent, embedding, arbiter, projector, detector = load_components()
    print("Running adaptation speed tests...")
    t0 = time.time()
    results = run_test(env, nn_agent, gp_agent, physics_agent, fm_agent,
                       embedding, arbiter, projector, detector)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  ADAPTATION SPEED SUMMARY (threshold-based, no PEARL)")
    print(f"{'='*60}")
    print(f"  {'Transition':<12} {'Detect':>8} {'WtShift':>8} {'RewardΔ':>8} {'ViolΔ':>8} {'Acc':>8}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for key, r in results.items():
        det = f"{r['detection_latency_steps']}s" if r['detected'] else "FAIL"
        vd = f"{r['post_violation_rate']-r['pre_violation_rate']:+.1%}"
        print(f"  {key:<12} {det:>8} {r['weight_shift_magnitude']:>8.4f} {r['reward_delta']:>+8.1f} {vd:>8} {r['regime_detection_accuracy']:>7.1%}")

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Results → {OUT_PATH}")

    with open(OUT_PATH, "w") as f:
        json.dump({"elapsed_seconds": elapsed, "results": results}, f, indent=2)

    env.close()


if __name__ == "__main__":
    main()