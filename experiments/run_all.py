"""
Runs the four planned experiments of the proposal and writes results/*.json.

  Exp 1  Single-regime baseline : SAC trained and tested on each regime + fixed controller
  Exp 2  Cross-regime gap       : each single-regime SAC tested zero-shot on every regime
  Exp 3  Meta-train + few-shot  : FOMAML+SAC over R0, R1, R3 (R2 held out), adapt K episodes
  Exp 4  Main comparison        : (a) scratch, (b) R0-pretrained + fine-tune, (c) joint
                                  multi-regime SAC, (d) first-order MAML+SAC (Reptile
                                  update), (e) FOMAML+SAC (query-gradient update),
                                  same budget, 5 seeds,
                                  plus a beta2 ablation.

Usage
  python experiments/run_all.py                 # full run (5 seeds)
  python experiments/run_all.py --quick         # 2-minute smoke test
  python experiments/run_all.py --workers 4     # use more CPU cores
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from multiprocessing import Pool

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from windrl.core import (HELD_OUT, TRAIN_REGIMES, Budget, adapt, evaluate,  # noqa: E402
                         fixed_controller, greedy, meta_train_fomaml, new_agent, pretrain_sac)
from windrl.env import RewardWeights  # noqa: E402

RES = os.path.join(ROOT, "results")
MODELS = os.path.join(RES, "models")


def pretrain_job(args):
    seed, kind, budget, b2 = args
    torch.set_num_threads(1)
    rw = RewardWeights(b2=b2)
    tag = f"seed{seed}_{kind}" + ("" if b2 == RewardWeights().b2 else f"_b2_{int(b2)}")
    cache = os.path.join(RES, "jobs", tag + ".json")
    if os.path.exists(cache) and os.path.exists(os.path.join(MODELS, tag + ".pt")):
        print(f"  cached {tag}", flush=True)
        return json.load(open(cache))
    log, t0 = [], time.time()
    if kind.startswith("single"):
        agent = pretrain_sac([int(kind[-1])], budget, seed, rw, log)
    elif kind == "joint":
        agent = pretrain_sac(list(TRAIN_REGIMES), budget, seed, rw, log)
    elif kind == "fomaml_q":
        agent = meta_train_fomaml(list(TRAIN_REGIMES), budget, seed, rw, log=log, algo="fomaml")
    else:
        agent = meta_train_fomaml(list(TRAIN_REGIMES), budget, seed, rw, log=log, algo="reptile")
    torch.save(agent.state_dict(), os.path.join(MODELS, tag + ".pt"))
    out = {"seed": seed, "kind": kind, "b2": b2, "train_log": log, "minutes": (time.time() - t0) / 60}
    if kind.startswith("single"):
        out["eval"] = {r: evaluate(greedy(agent), r, budget, rw) for r in range(4)}
    json.dump(out, open(cache, "w"))
    print(f"  done {tag} in {out['minutes']:.1f} min", flush=True)
    return out


def adapt_job(args):
    seed, target, method, budget, b2 = args
    torch.set_num_threads(1)
    rw = RewardWeights(b2=b2)
    suffix = "" if b2 == RewardWeights().b2 else f"_b2_{int(b2)}"
    src = {"finetune_R0": "single0", "joint": "joint", "fomaml": "fomaml",
           "fomaml_q": "fomaml_q"}.get(method)
    init = None
    if src:
        init = new_agent(budget, seed)
        init.load_state_dict(torch.load(os.path.join(MODELS, f"seed{seed}_{src}{suffix}.pt")))
    curve = adapt(init, target, budget, seed, rw, from_scratch=init is None)
    print(f"  adapted seed{seed} {method} -> R{target}{suffix}", flush=True)
    return {"seed": seed, "target": target, "method": method, "b2": b2, "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=60, help="pre-training iterations (x3 episodes)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ablation-seeds", type=int, default=3)
    ap.add_argument("--skip-ablation", action="store_true")
    a = ap.parse_args()

    budget = Budget(pretrain_iters=a.iters)
    seeds = list(range(a.seeds))
    abl_seeds = list(range(a.ablation_seeds))
    if a.quick:
        budget = replace(budget, pretrain_iters=3, n_eval=2, adapt_K=(0, 1, 2), updates_per_episode=20)
        seeds, abl_seeds = [0], [0]
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(os.path.join(RES, "jobs"), exist_ok=True)
    b2_main = RewardWeights().b2
    b2_abl = [0.5 * b2_main, 2.0 * b2_main]
    t0 = time.time()

    with Pool(a.workers) as pool:
        print("Phase A: pre-training (Exp 1-3)", flush=True)
        kinds = ["single0", "single1", "single2", "single3", "joint", "fomaml", "fomaml_q"]
        jobs = [(s, k, budget, b2_main) for s in seeds for k in kinds]
        if not a.skip_ablation:
            jobs += [(s, k, budget, b) for s in abl_seeds for b in b2_abl for k in ("fomaml",)]
        pre = pool.map(pretrain_job, jobs, chunksize=1)
        fixed = {r: evaluate(fixed_controller, r, budget) for r in range(4)}

        print("Phase B: few-shot adaptation (Exp 3-4)", flush=True)
        methods = ["scratch", "finetune_R0", "joint", "fomaml", "fomaml_q"]
        jobs = [(s, t, m, budget, b2_main) for s in seeds for t in (HELD_OUT, 3) for m in methods]
        if not a.skip_ablation:
            jobs += [(s, HELD_OUT, m, budget, b) for s in abl_seeds for b in b2_abl
                     for m in ("scratch", "fomaml")]
        ad = pool.map(adapt_job, jobs, chunksize=1)

    out = {
        "budget": asdict(budget), "seeds": seeds, "train_regimes": TRAIN_REGIMES,
        "held_out": HELD_OUT, "reward": asdict(RewardWeights()), "fixed_controller": fixed,
        "pretrain": pre, "adaptation": ad, "minutes": (time.time() - t0) / 60,
    }
    name = "results_quick.json" if a.quick else "results.json"
    with open(os.path.join(RES, name), "w") as f:
        json.dump(out, f)
    print(f"Saved results/{name} in {out['minutes']:.1f} min")


if __name__ == "__main__":
    main()
