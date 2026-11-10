"""
Training, evaluation, few-shot adaptation and FOMAML meta-training.

Every method gets the same budget, so the comparison measures adaptation and
not compute:
  * pre-training: the same number of environment episodes and the same number
    of gradient computations (UTD = updates_per_episode / episode length);
  * adaptation: K episodes on the target turbine, `updates_per_episode` SAC
    updates after each one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .env import REGIMES, RewardWeights, Task, WindTurbineEnv
from .sac import SAC, ReplayBuffer


@dataclass
class Budget:
    steps_per_episode: int = 240       # 240 x 30 s = 2 h of operation per episode
    pretrain_iters: int = 60           # each iter = one episode on each of 3 tasks
    updates_per_episode: int = 120     # UTD 0.5
    batch: int = 128
    hidden: int = 128
    adapt_K: tuple = (0, 1, 5, 10, 20)
    n_eval: int = 8                    # fixed test episodes (test-split wind)
    warmup_random_episodes: int = 1    # SAC-from-scratch starts with random actions
    inner_lr: float = 3e-4
    meta_lr: float = 3e-4


TRAIN_REGIMES = (0, 1, 3)   # R2 is held out entirely for the unseen-regime test
HELD_OUT = 2


# ---------------------------------------------------------------------------
def eval_task(regime: int) -> Task:
    """The fixed 'target turbine' for each regime: randomised parameters, fixed seed."""
    return Task.sample(regime, np.random.default_rng(777 + regime))


def rollout(env, policy, buffer=None, random_actions=False, rng=None, reset_kw=None):
    o, _ = env.reset(**(reset_kw or {}))
    R, info, n = 0.0, {}, 0
    powers, unsafe = [], 0
    while True:
        a = rng.uniform(-1, 1, 2).astype(np.float32) if random_actions else policy(o)
        o2, r, term, trunc, info = env.step(a)
        if buffer is not None:
            buffer.add(o, a, r, o2, float(term))
        R += r
        n += 1
        powers.append(info["power"])
        unsafe += info["unsafe"]
        o = o2
        if term or trunc:
            break
    return {
        "return": R,
        "d_delta": env.delta - env.delta_start,
        "slope_per_h": (env.delta - env.delta_start) / (n * env.dt / 3600.0),
        "failed": float(term),
        "power": float(np.mean(powers)),
        "unsafe_frac": unsafe / n,
        "steps": n,
    }


def evaluate(policy, regime: int, budget: Budget, reward: RewardWeights | None = None) -> dict:
    """Deterministic evaluation on the fixed target turbine with unseen (Oct-Dec) wind."""
    task = eval_task(regime)
    env = WindTurbineEnv(task, split="test", steps_per_episode=budget.steps_per_episode, reward=reward)
    _, lo, hi = REGIMES[regime]
    res = []
    for i in range(budget.n_eval):
        rng = np.random.default_rng(10_000 + i)
        seg = env.bank.test[rng.integers(len(env.bank.test))]
        d0 = lo + (hi - lo) * (i + 0.5) / budget.n_eval
        env.rng = np.random.default_rng(20_000 + i)
        res.append(rollout(env, policy, reset_kw={"options": {"wind_segment": seg, "delta0": d0}}))
    return {k: float(np.mean([r[k] for r in res])) for k in res[0]}


def greedy(agent: SAC):
    return lambda o: agent.act(o, deterministic=True)


def fixed_controller(o):
    """Rated-power reference: zero offsets = the turbine's own optimal-torque + PI-pitch control."""
    return np.zeros(2, dtype=np.float32)


# ---------------------------------------------------------------------------
def new_agent(budget: Budget, seed: int) -> SAC:
    torch.set_num_threads(1)
    return SAC(hidden=budget.hidden, batch=budget.batch, lr=budget.inner_lr, seed=seed)


def pretrain_sac(regimes, budget: Budget, seed: int, reward=None, log=None) -> SAC:
    """Plain SAC on a list of regimes (one regime = single-regime SAC; several =
    joint multi-task SAC, baseline (c)). Same episodes and gradient computations
    as FOMAML: per iteration one episode on each of 3 tasks."""
    rng = np.random.default_rng(seed)
    agent = new_agent(budget, seed)
    buf = ReplayBuffer(8, 2, 200_000)
    env = WindTurbineEnv(steps_per_episode=budget.steps_per_episode, reward=reward,
                         seed=seed)
    tasks_per_iter = 3
    for it in range(budget.pretrain_iters):
        for j in range(tasks_per_iter):
            r = regimes[(it * tasks_per_iter + j) % len(regimes)]
            env.set_task(Task.sample(r, rng))
            rand = it * tasks_per_iter + j < budget.warmup_random_episodes
            out = rollout(env, lambda o: agent.act(o), buf, random_actions=rand, rng=rng)
            for _ in range(budget.updates_per_episode):
                agent.update(buf.sample(budget.batch, rng))
            if log is not None:
                log.append({"iter": it, "regime": r, **out})
    return agent


def meta_train_fomaml(regimes, budget: Budget, seed: int, reward=None, tasks_per_regime=4,
                      log=None, algo: str = "reptile") -> SAC:
    """First-order meta-learning for off-policy SAC (Finn et al. 2017; Nichol et al. 2018).

    Each task tau_i keeps its own replay buffer (keeps the procedure off-policy).
    Per meta-iteration, for one task from each training regime:
      1. clone the meta-parameters theta -> theta'_i and collect one episode with theta'_i,
      2. inner loop: SAC updates on theta'_i using that task's buffer,
      3. outer update of theta (no second derivatives), either
         algo="reptile" (default): theta <- theta + eps * mean_i(theta'_i - theta),
                                   eps annealed 1.0 -> 0.1 (the first-order update of [3]);
         algo="fomaml":  query gradient of the SAC loss at theta'_i on a fresh batch,
                         averaged over tasks and applied to theta with Adam.
    Both use one episode + updates_per_episode gradient computations per task, the
    same budget as the SAC baselines.

    Why Reptile is the default: with this budget the "fomaml" form gets only one
    outer Adam step per meta-iteration (60 in total, versus ~21,600 SAC updates
    for the joint baseline), so its initialisation stays close to random. The
    Reptile form moves theta by the whole inner-loop progress every iteration.
    """
    rng = np.random.default_rng(seed)
    meta = new_agent(budget, seed)
    params = meta.meta_params()
    meta_opt = torch.optim.Adam(params, lr=budget.meta_lr)
    pool = {r: [Task.sample(r, rng) for _ in range(tasks_per_regime)] for r in regimes}
    bufs = {(r, i): ReplayBuffer(8, 2, 50_000) for r in regimes for i in range(tasks_per_regime)}
    env = WindTurbineEnv(steps_per_episode=budget.steps_per_episode, reward=reward, seed=seed)
    n_inner = budget.updates_per_episode - (1 if algo == "fomaml" else 0)
    N = budget.pretrain_iters
    for it in range(N):
        acc = [torch.zeros_like(p) for p in params]
        new_targets = []
        for r in regimes:
            i = int(rng.integers(tasks_per_regime))
            env.set_task(pool[r][i])
            buf = bufs[(r, i)]
            learner = meta.clone()
            rand = it == 0 and budget.warmup_random_episodes > 0
            out = rollout(env, lambda o: learner.act(o), buf, random_actions=rand, rng=rng)
            for _ in range(n_inner):
                learner.update(buf.sample(budget.batch, rng))
            if algo == "fomaml":
                g = learner.grads(buf.sample(budget.batch, rng))
            else:  # Reptile direction: theta - theta'
                g = [p.detach() - q.detach() for p, q in zip(params, learner.meta_params())]
            for a, gi in zip(acc, g):
                a.add_(gi / len(regimes))
            new_targets.append(learner.critic_t.state_dict())
            if log is not None:
                log.append({"iter": it, "regime": r, **out})
        with torch.no_grad():
            if algo == "fomaml":
                meta_opt.zero_grad()
                for p, a in zip(params, acc):
                    p.grad = a
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                meta_opt.step()
            else:
                eps = 1.0 - 0.9 * it / max(N - 1, 1)
                for p, a in zip(params, acc):
                    p.sub_(eps * a)
            # meta target critic: average of the adapted learners' target critics
            sd = meta.critic_t.state_dict()
            for k in sd:
                sd[k].copy_(torch.stack([t[k] for t in new_targets]).mean(0))
    return meta


def adapt(init: SAC | None, regime: int, budget: Budget, seed: int, reward=None,
          from_scratch=False) -> list:
    """Few-shot adaptation on the target turbine: K episodes (train-split wind),
    updates_per_episode SAC updates after each, evaluated at K in budget.adapt_K."""
    rng = np.random.default_rng(seed + 99)
    agent = new_agent(budget, seed) if init is None else init.clone()
    buf = ReplayBuffer(8, 2, 50_000)
    env = WindTurbineEnv(eval_task(regime), split="train",
                         steps_per_episode=budget.steps_per_episode, reward=reward, seed=seed + 5)
    curve = []
    Kmax = max(budget.adapt_K)
    for k in range(Kmax + 1):
        if k in budget.adapt_K:
            curve.append({"K": k, **evaluate(greedy(agent), regime, budget, reward)})
        if k == Kmax:
            break
        rand = from_scratch and k < budget.warmup_random_episodes
        rollout(env, lambda o: agent.act(o), buf, random_actions=rand, rng=rng)
        for _ in range(budget.updates_per_episode):
            agent.update(buf.sample(budget.batch, rng))
    return curve
