#!/usr/bin/env python3
"""
Regime-wise analysis of PINN ODE residuals and SAC policy entropy.

Part A (training logs): recovers the per-episode ODE residual from the logged
    r_physics = -tanh(|residual|) (episode mean) and groups by regime_id and
    training stage.
Part B (rollouts): loads checkpoints, runs the combined CEMRL policy starting in
    each regime under the TRAINING env (quadratic temp model, deg_acceleration=1)
    and records per step:
      - ode_res   : PINN training ODE term  MSE(pinn dω pred, physics dω/dt)
      - data_res  : MSE(pinn Δstate pred, actual env Δstate)   (all dims)
      - data_res_w: same, ω dim only
      - sac_H_mc  : SAC entropy, Monte-Carlo  -E[log π(a|s)]  (tanh-squashed)
      - sac_H_gauss: pre-tanh Gaussian entropy  Σ(0.5 log(2πe) + log σ)
      - sac_std   : mean σ of pre-tanh Gaussian
      - sac_sat   : fraction of |mean action| > 0.95 after tanh (saturation)
    and aggregates by the regime of the current δ.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hmarl_drt"))

from hmarl_drt.envs.wind_turbine_env import WindTurbineEnv
from hmarl_drt.agents.sac_agent import SACAgent
from hmarl_drt.agents.pinn_agent import PINNAgent
from hmarl_drt.agents.ipae_agent import IPAEAgent
from hmarl_drt.agents.fm_agent import FMAgent
from hmarl_drt.core.embedding import SharedEmbeddingSpace
from hmarl_drt.core.cemrl_arbiter import CEMRLArbiter
from validate_ep2300_deepdive import (
    safe_load, make_env_config, regime_from_delta, build_supervised_z_task, safety_urgency,
)

RUN = Path("hmarl_drt/runs/fresh_ipae_supervised")
REGIME_MID = {0: (0.05, 0.20), 1: (0.28, 0.45), 2: (0.53, 0.70), 3: (0.78, 0.90)}


def part_a_logs():
    df = pd.read_csv(RUN / "training_log.csv")
    df = df[df["cemrl/regime_id"].notna()].copy()
    df["regime"] = df["cemrl/regime_id"].astype(int)
    # episode-mean of -tanh(|res|) -> tanh-space proxy of residual
    df["res_proxy"] = np.arctanh(np.clip(-df["r_physics"], 0, 1 - 1e-7))
    bins = [0, 500, 1000, 2300, 3500, 5001]
    labels = ["0-500", "500-1000", "1000-2300", "2300-3500", "3500-5000"]
    df["stage"] = pd.cut(df["episode"], bins=bins, labels=labels, right=False)
    out = {}
    print("\n=== Part A: training-log ODE residual proxy (atanh(-r_physics)) by regime ===")
    tab = df.groupby(["stage", "regime"], observed=True)["res_proxy"].agg(["mean", "std", "count"])
    print(tab.round(4).to_string())
    overall = df.groupby("regime")[["res_proxy", "r_physics"]].agg(["mean", "std", "count"])
    print("\nOverall by regime:")
    print(overall.round(4).to_string())
    out["by_stage_regime"] = {f"{s}|R{r}": row.to_dict() for (s, r), row in tab.iterrows()}
    out["overall"] = {f"R{r}": {"res_mean": float(df[df.regime == r].res_proxy.mean()),
                                "res_std": float(df[df.regime == r].res_proxy.std()),
                                "n": int((df.regime == r).sum())} for r in sorted(df.regime.unique())}
    return out


def load_system(ckpt_path, cfg, env, device):
    sd = int(env.observation_space.shape[0]); ad = int(env.action_space.shape[0])
    phys = cfg["env"]["physics"]
    shared_dim = cfg.get("embedding", {}).get("shared_dim", 64)
    ipae_cfg = cfg.get("ipae", {})
    constr = cfg.get("env", {}).get("constraints", {})
    om_min, om_max = float(constr.get("omega_min", 0.5)), float(constr.get("omega_max", 1.5))
    T_max = float(constr.get("T_max", 100.0))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    sac = SACAgent(state_dim=sd, shared_dim=shared_dim, action_dim=ad,
                   hidden_dim=cfg["agents"]["sac"]["hidden_dim"], device=device)
    safe_load(sac.actor, ckpt.get("sac_actor"), "SAC actor")
    pinn = PINNAgent(state_dim=sd, action_dim=ad, J=float(phys["J"]), B=float(phys["B"]),
                     R=float(phys["R"]), omega_min=om_min, omega_max=om_max, T_max=T_max,
                     hidden_dim=cfg["agents"]["pinn"]["hidden_dim"], device=device)
    safe_load(pinn.pinn, ckpt.get("pinn"), "PINN")
    safe_load(pinn.policy, ckpt.get("pinn_policy"), "PINN policy")
    ipae = IPAEAgent(state_dim=sd, action_dim=ad,
                     latent_dim=ipae_cfg.get("latent_dim", 8),
                     window_size=ipae_cfg.get("window_size", 10),
                     encoder_hidden=ipae_cfg.get("encoder_hidden", 64),
                     lstm_hidden=ipae_cfg.get("lstm_hidden", 64),
                     lstm_layers=ipae_cfg.get("lstm_layers", 1),
                     rul_hidden=ipae_cfg.get("rul_hidden", [64, 32, 16]),
                     mc_dropout=ipae_cfg.get("mc_dropout", 0.1), device=device)
    safe_load(ipae.network, ckpt.get("gp_network"), "iPAE network")
    fm = FMAgent(state_dim=sd, use_api=False, device=device, omega_min=om_min,
                 omega_max=om_max, T_max=T_max, delta_crit=float(constr.get("delta_crit", 0.9)))
    with torch.no_grad():
        d = torch.zeros(1, sd, device=device)
        dims = [sac.actor.get_embedding(d).shape[-1], ipae.get_embedding(d).shape[-1],
                fm.get_embedding(d).shape[-1], pinn.get_embedding(d).shape[-1]]
    emb = SharedEmbeddingSpace(dims, shared_dim=shared_dim, n_agents=4).to(device)
    safe_load(emb, ckpt.get("embedding"), "shared embedding")
    z_task_dim = cfg.get("cemrl", {}).get("z_task_dim", 32)
    arb = CEMRLArbiter(state_dim=sd, z_task_dim=z_task_dim, n_agents=4,
                       hidden_dim=cfg["cemrl"]["cemrl_arbiter"]["hidden_dim"], device=device)
    safe_load(arb.gating, ckpt.get("cemrl_arbiter"), "CEMRL arbiter gating")
    return dict(sac=sac, pinn=pinn, ipae=ipae, fm=fm, emb=emb, arb=arb, z_task_dim=z_task_dim,
                om_min=om_min, om_max=om_max, T_max=T_max,
                warning_caution=float(cfg.get("arbiter", {}).get("warning_caution", 0.5)),
                margin=float(cfg.get("arbiter", {}).get("safety_margin_frac", 0.15)))


@torch.no_grad()
def rollout(env, S, device, init_delta, seed, n_mc=64):
    sac, pinn, ipae, fm, emb, arb = S["sac"], S["pinn"], S["ipae"], S["fm"], S["emb"], S["arb"]
    state, _ = env.reset(seed=seed, options={"delta_range": (init_delta, init_delta)})
    if hasattr(ipae, "reset_window"):
        ipae.reset_window()
    rec = []
    done = False
    lo, hi = env.action_space.low, env.action_space.high
    while not done:
        s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        zs, _ = emb([sac.actor.get_embedding(s_t), ipae.get_embedding(s_t),
                     fm.get_embedding(s_t), pinn.get_embedding(s_t)])
        # SAC entropy
        mean, log_std = sac.actor(s_t, zs)
        s_rep = s_t.expand(n_mc, -1); z_rep = zs.expand(n_mc, -1)
        _, logp = sac.actor.sample(s_rep, z_rep)
        H_mc = float(-logp.mean())
        H_gauss = float((0.5 * np.log(2 * np.pi * np.e) + log_std).sum())
        sac_std = float(log_std.exp().mean())
        sac_sat = float((torch.tanh(mean).abs() > 0.95).float().mean())

        a_sac = sac.act(state, zs.squeeze(0))
        a_ipae = ipae.act(state); a_fm = fm.act(state); a_pinn = pinn.policy_act(state)
        delta = float(state[5]); reg = regime_from_delta(delta)
        unc = ipae.get_uncertainty_np(state)
        sev, _ = fm.assess(state)
        urg = safety_urgency(state, S["om_min"], S["om_max"], S["T_max"], S["margin"])
        caution = max(S["warning_caution"] if sev == "warning" else 0.0, urg)
        z_task = build_supervised_z_task(reg, delta, z_dim=S["z_task_dim"], device=device)
        action, w = arb.select_action(
            state, {"nn": a_sac, "gp": a_ipae, "fm": a_fm, "physics": a_pinn},
            {"low": lo, "high": hi}, z_task=z_task, delta=delta, uncertainty=unc,
            emergency=(sev == "critical"), caution=caution, return_weights=True)
        action = np.asarray(action, dtype=np.float32)

        nxt, _, term, trunc, _ = env.step(action)
        a_t = torch.FloatTensor(action).unsqueeze(0).to(device)
        ode = float(pinn.ode_residual(s_t, a_t))
        pred = pinn.pinn(s_t, a_t)
        actual = torch.FloatTensor(nxt).unsqueeze(0).to(device) - s_t
        data = float(F.mse_loss(pred, actual))
        data_w = float((pred[0, 0] - actual[0, 0]) ** 2)
        rec.append(dict(regime=reg, ode_res=ode, data_res=data, data_res_w=data_w,
                        sac_H_mc=H_mc, sac_H_gauss=H_gauss, sac_std=sac_std, sac_sat=sac_sat,
                        w_pinn=float(w[3]), w_sac=float(w[0])))
        state = nxt
        done = term or trunc
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="hmarl_drt/configs/wind_fresh_ipae_supervised.yaml")
    p.add_argument("--ckpts", nargs="+", default=["ep_00100", "ep_01000", "ep_02300", "ep_05000"])
    p.add_argument("--n-episodes", type=int, default=3)
    p.add_argument("--out", default="results/pinn_sac_regimewise.json")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {"part_a_logs": part_a_logs(), "part_b_rollouts": {}}

    cfg = yaml.safe_load(open(args.config))
    env_cfg = make_env_config(cfg)
    env_cfg.setdefault("physics", {})
    env_cfg["physics"]["deg_acceleration"] = 1.0       # match training-time dynamics
    env_cfg["physics"]["temp_deg_model"] = "quadratic"
    env = WindTurbineEnv(env_cfg)
    metrics = ["ode_res", "data_res", "data_res_w", "sac_H_mc", "sac_H_gauss",
               "sac_std", "sac_sat", "w_pinn", "w_sac"]

    for ck in args.ckpts:
        print(f"\n=== Part B: {ck} ===")
        S = load_system(RUN / "checkpoints" / f"{ck}.pt", cfg, env, device)
        rows = []
        rng = np.random.default_rng(123)
        for r, (lo, hi) in REGIME_MID.items():
            for ep in range(args.n_episodes):
                rows += rollout(env, S, device, float(rng.uniform(lo, hi)), seed=7000 + 10 * r + ep)
        df = pd.DataFrame(rows)
        agg = df.groupby("regime")[metrics].agg(["mean", "std"])
        cnt = df.groupby("regime").size()
        with pd.option_context("display.width", 250, "display.float_format", "{:.4g}".format):
            print(agg.xs("mean", axis=1, level=1).assign(n_steps=cnt).to_string())
        results["part_b_rollouts"][ck] = {
            f"R{r}": {**{m: {"mean": float(agg.loc[r, (m, "mean")]),
                             "std": float(agg.loc[r, (m, "std")])} for m in metrics},
                      "n_steps": int(cnt.loc[r])}
            for r in agg.index}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2, default=str)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
