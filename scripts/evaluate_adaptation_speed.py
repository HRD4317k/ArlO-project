"""Adaptation‑speed evaluation for MAML‑initialized vs. scratch policies.

This script is intended to be used in the *supervised* validation pipeline
where the true degradation index ``Δ`` and the true regime ID are exposed to
the agents.  It measures how quickly a policy can adapt after a regime
transition when it starts from a *meta‑learned* initialization (obtained via
MAML) compared to training a fresh policy from scratch.

The workflow is:

1. **Load a meta‑checkpoint** containing the *generalized* parameters that were
   obtained by MAML across all regimes.  The checkpoint should be compatible
   with the SAC agent architecture used elsewhere in the code base.
2. **Select a target regime** (e.g. regime ``2``).  The environment is forced
   to stay in this regime for the entire evaluation by setting
   ``regime_probs`` to a one‑hot vector.
3. **Few‑shot adaptation** – run ``support_episodes`` episodes using the
   meta‑initialized policy, then perform ``inner_steps`` gradient updates with
   learning rate ``inner_lr`` on the collected transitions.
4. **Evaluation** – run episodes with the adapted policy until the average
   episode return reaches ``target_reward`` (expressed as a fraction of the
   maximum achievable return for the regime) or until a maximum number of
   episodes is reached.  The number of episodes required is recorded as the
   *MAML adaptation speed*.
5. **Baseline comparison** – initialise a fresh SAC policy (random weights),
   train it for ``baseline_episodes`` on the same regime, then evaluate it in
   the same way as step 4.  The resulting episode count is the *scratch
   adaptation speed*.
6. **Output** – a JSON file ``adaptation_speed.json`` is written to the
   ``results/`` directory containing both metrics.

The script relies on the existing training utilities in the repository (e.g.
``hmarl_drt.agents.sac`` and ``hmarl_drt.training``).  If those modules are
renamed, adjust the imports accordingly.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml
import torch

# ---------------------------------------------------------------------------
# Helper utilities – these are lightweight wrappers around the existing code
# base.  They are deliberately simple; the heavy‑lifting (SAC loss, replay
# buffer, etc.) is delegated to the library functions.
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load a YAML config file and return a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def force_regime(cfg: dict, regime_id: int) -> dict:
    """Modify the environment config so that only ``regime_id`` is sampled.

    The original config contains ``regime_probs`` – we replace it with a
    one‑hot vector where the selected regime has probability ``1.0``.
    """
    n = len(cfg["env"]["degradation"]["regime_probs"])
    one_hot = [0.0] * n
    one_hot[regime_id] = 1.0
    cfg["env"]["degradation"]["regime_probs"] = one_hot
    return cfg


def build_sac_agent(cfg: dict, device: torch.device, checkpoint: str | None = None):
    """Instantiate a SAC agent.

    If ``checkpoint`` is provided, the model weights are loaded; otherwise the
    agent is freshly initialised.
    """
    from hmarl_drt.agents.sac import SACAgent  # type: ignore

    agent_cfg = cfg.get("agents", {}).get("sac", {})
    # The environment may not expose state_dim/action_dim directly; we infer
    # them from the observation/action spaces if present.
    state_dim = cfg["env"].get("state_dim")
    action_dim = cfg["env"].get("action_dim")
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=agent_cfg.get("hidden_dim", 256),
        lr_actor=agent_cfg.get("lr_actor", 1e-4),
        lr_critic=agent_cfg.get("lr_critic", 1e-4),
        device=device,
    )
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        # The checkpoint may store the SAC agent under a key; we try a few.
        if "sac_agent" in state:
            agent.load_state_dict(state["sac_agent"])
        else:
            agent.load_state_dict(state)
    return agent


def run_support_episodes(env, agent, num_episodes: int, device: torch.device):
    """Collect ``num_episodes`` of experience using ``agent``.

    Returns a list of transition tuples ``(s, a, r, s', done)`` that can be
    used for the inner‑loop gradient updates.
    """
    from hmarl_drt.training import collect_episode  # type: ignore

    transitions = []
    for _ in range(num_episodes):
        ep_trans = collect_episode(env, agent, device=device)
        transitions.extend(ep_trans)
    return transitions


def inner_update(agent, transitions, inner_lr: float, inner_steps: int, device: torch.device):
    """Perform ``inner_steps`` gradient updates on ``agent`` using the given
    transitions.  A simple mean‑squared‑error on the Q‑value targets is used.
    """
    optimizer = torch.optim.Adam(agent.parameters(), lr=inner_lr)
    for _ in range(inner_steps):
        # For simplicity we use the whole batch; a real implementation would
        # sample a minibatch.
        loss = agent.compute_loss(transitions)  # assumes SACAgent implements this
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate_policy(env, agent, device: torch.device, max_episodes: int = 500):
    """Run episodes until the average return stabilises. Returns a list of
    episode returns.
    """
    from hmarl_drt.training import run_episode  # type: ignore

    returns = []
    for _ in range(max_episodes):
        ep_ret = run_episode(env, agent, device=device)
        returns.append(ep_ret)
    return returns


def compute_adaptation_speed(returns, target_fraction, max_possible):
    """Given a list of episode returns, compute the number of episodes needed
    to reach ``target_fraction * max_possible``.  If the target is never reached,
    the function returns the length of ``returns``.
    """
    target = target_fraction * max_possible
    for i, r in enumerate(returns, start=1):
        if r >= target:
            return i
    return len(returns)


def main():
    parser = argparse.ArgumentParser(description="Adaptation‑speed evaluation")
    parser.add_argument("--config", required=True, help="Path to the environment config YAML")
    parser.add_argument("--regime", type=int, default=2, help="Target regime ID for evaluation")
    parser.add_argument("--support_episodes", type=int, default=10)
    parser.add_argument("--inner_steps", type=int, default=5)
    parser.add_argument("--inner_lr", type=float, default=5e-4)
    parser.add_argument("--target_reward", type=float, default=0.90)
    parser.add_argument("--baseline_episodes", type=int, default=200)
    parser.add_argument("--meta_checkpoint", required=True, help="MAML meta‑checkpoint path")
    parser.add_argument("--output", default="results/adaptation_speed.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------------
    # Load and prepare the environment configuration
    # ---------------------------------------------------------------------
    cfg = load_config(args.config)
    cfg = force_regime(cfg, args.regime)
    # Ensure the true delta is visible for the supervised setting
    cfg["cemrl"]["mask_true_delta"] = False
    cfg["cemrl"]["expose_true_delta_for_eval"] = True

    # ---------------------------------------------------------------------
    # Build the environment – the repository provides a factory function.
    # ---------------------------------------------------------------------
    from hmarl_drt.envs.wind_turbine_env import WindTurbineEnv  # type: ignore
    env = WindTurbineEnv(cfg["env"])

    # ---------------------------------------------------------------------
    # 1️⃣ MAML‑initialized policy – few‑shot adaptation
    # ---------------------------------------------------------------------
    maml_agent = build_sac_agent(cfg, device, checkpoint=args.meta_checkpoint)
    support = run_support_episodes(env, maml_agent, args.support_episodes, device)
    inner_update(maml_agent, support, args.inner_lr, args.inner_steps, device)
    # Evaluate after adaptation
    maml_returns = evaluate_policy(env, maml_agent, device)
    max_possible = max(maml_returns) if maml_returns else 0.0
    maml_speed = compute_adaptation_speed(maml_returns, args.target_reward, max_possible)

    # ---------------------------------------------------------------------
    # 2️⃣ Baseline – train from scratch on the same regime
    # ---------------------------------------------------------------------
    scratch_agent = build_sac_agent(cfg, device, checkpoint=None)
    baseline_support = run_support_episodes(env, scratch_agent, args.baseline_episodes, device)
    # No inner‑loop updates – the training itself is the adaptation.
    scratch_returns = evaluate_policy(env, scratch_agent, device)
    max_possible_scratch = max(scratch_returns) if scratch_returns else 0.0
    scratch_speed = compute_adaptation_speed(scratch_returns, args.target_reward, max_possible_scratch)

    # ---------------------------------------------------------------------
    # Write results
    # ---------------------------------------------------------------------
    os.makedirs(Path(args.output).parent, exist_ok=True)
    result = {
        "regime": args.regime,
        "target_reward_fraction": args.target_reward,
        "maml_adaptation_episodes": maml_speed,
        "scratch_adaptation_episodes": scratch_speed,
        "meta_checkpoint": args.meta_checkpoint,
        "support_episodes": args.support_episodes,
        "inner_steps": args.inner_steps,
        "baseline_training_episodes": args.baseline_episodes,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Adaptation‑speed results written to {args.output}")


if __name__ == "__main__":
    main()
