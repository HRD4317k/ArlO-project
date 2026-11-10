"""Builds every figure and the results tables from results/results.json."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from windrl.core import Budget, eval_task, fixed_controller, greedy, new_agent  # noqa: E402
from windrl.data import binned_power_curve, get_wind_bank, load_scada  # noqa: E402
from windrl.env import REGIMES, Task, TurbineParams, WindTurbineEnv  # noqa: E402

FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)

C = {"scratch": "#8a8f98", "finetune_R0": "#e08a2c", "joint": "#3a7bd5", "fomaml": "#d63a6a",
     "fomaml_q": "#9b59b6", "fixed": "#222222"}
LABEL = {"scratch": "(a) SAC from scratch", "finetune_R0": "(b) SAC pre-trained R0 + fine-tune",
         "joint": "(c) SAC joint multi-regime", "fomaml": "(d) First-order MAML + SAC, Reptile update",
         "fomaml_q": "(e) FOMAML + SAC, query-gradient update"}
RNAME = {r: REGIMES[r][0] for r in range(4)}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/" + name)


# ---------------------------------------------------------------------------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.2)
    boxes = [
        (0.2, 2.4, 2.2, 1.4, "Real SCADA data\n3.6 MW turbine, 2018\n10-min wind + power", "#fde7c7"),
        (0.2, 0.3, 2.2, 1.6, "Split by time\nJan-Sep: training wind\nOct-Dec: test wind\n+ power-curve calibration", "#fde7c7"),
        (3.1, 1.2, 2.5, 1.9, "Gymnasium simulator\nrotor + inner controller\nbearing heat + wear δ\nregimes R0-R3", "#d9ecff"),
        (6.3, 2.5, 2.1, 1.3, "SAC\n(twin critics,\nauto entropy)", "#ffd9e4"),
        (6.3, 0.4, 2.1, 1.6, "FOMAML outer loop\nper-task buffers\nmeta-train R0, R1, R3\nR2 held out", "#ffd9e4"),
        (9.0, 1.2, 1.85, 1.9, "Adapt with\nK = 0,1,5,10,20\nepisodes on the\ntarget turbine", "#dff3e3"),
    ]
    for x, y, w, h, t, col in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                                    fc=col, ec="#444", lw=1.2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9)
    arrows = [((2.4, 3.1), (3.1, 2.6)), ((2.4, 1.1), (3.1, 1.7)), ((5.6, 2.4), (6.3, 3.1)),
              ((7.35, 2.5), (7.35, 2.0)), ((8.4, 3.1), (9.0, 2.5)), ((8.4, 1.2), (9.0, 1.7)),
              ((6.3, 1.0), (5.6, 1.6))]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#333"))
    ax.text(5.55, 0.55, "actions:\nΔpitch, Δtorque", fontsize=8, ha="center", color="#333")
    ax.set_title("Degradation-aware wind turbine control: pipeline", fontsize=12, fontweight="bold")
    save(fig, "fig0_pipeline.png")


def fig_data():
    df = load_scada()
    vc, pc, cnt = binned_power_curve(df)
    p = replace(TurbineParams(), TI=0.0)
    sim = []
    for v in vc:
        env = WindTurbineEnv(Task(0, p), seed=0)
        env.reset(options={"wind_segment": np.full(40, v, np.float32), "delta0": 0.0})
        P = [env.step([0, 0])[4]["power"] for _ in range(20)]
        sim.append(np.mean(P[-5:]))
    sim = np.array(sim)
    m = (vc > 3) & (vc < 15)
    rmse = float(np.sqrt(np.mean((sim[m] - pc[m]) ** 2)))
    bank = get_wind_bank(14)
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    d = df.dropna().sample(6000, random_state=0)
    axs[0].scatter(d["wind"], d["power_kw"] / 3600, s=2, alpha=0.15, c="#999", label="SCADA 10-min records")
    axs[0].plot(vc, pc, "o-", ms=3, c="#e08a2c", label="measured median curve")
    axs[0].plot(vc, sim, "-", lw=2.2, c="#3a7bd5", label=f"simulator (RMSE {rmse:.3f} of rated)")
    axs[0].set(xlabel="wind speed [m/s]", ylabel="power / rated", title="Calibration: power curve")
    axs[0].legend(fontsize=8)
    w = df["wind"].dropna()
    axs[1].hist(w[w.index < "2018-10-01"], bins=50, density=True, alpha=0.6, label="train (Jan-Sep)", color="#3a7bd5")
    axs[1].hist(w[w.index >= "2018-10-01"], bins=50, density=True, alpha=0.6, label="test (Oct-Dec)", color="#d63a6a")
    axs[1].set(xlabel="wind speed [m/s]", ylabel="density", title="Real wind used for train / test")
    axs[1].legend(fontsize=8)
    env = WindTurbineEnv(Task(0), split="test", seed=3)
    env.reset(options={"wind_segment": bank.test[5]})
    ws = []
    for _ in range(240):
        env.step([0, 0])
        ws.append(env.v)
    t = np.arange(240) * 0.5
    axs[2].plot(t, ws, c="#3a7bd5", lw=1, label="simulator wind (30-s steps)")
    axs[2].plot(np.arange(len(bank.test[5])) * 10, bank.test[5], "o-", c="#e08a2c", label="real 10-min SCADA")
    axs[2].set(xlim=(0, 120), xlabel="time [min]", ylabel="wind [m/s]", title="One test episode's wind input")
    axs[2].legend(fontsize=8)
    save(fig, "fig1_real_data_and_calibration.png")
    return rmse


def fig_single_and_cross(res):
    pre = [p for p in res["pretrain"] if p["kind"].startswith("single") and p["b2"] == res["reward"]["b2"]]
    seeds = sorted({p["seed"] for p in pre})
    M = np.zeros((len(seeds), 4, 4))
    D = np.zeros((len(seeds), 4, 4))
    F = np.zeros((len(seeds), 4, 4))
    for p in pre:
        i, tr = seeds.index(p["seed"]), int(p["kind"][-1])
        for te in range(4):
            e = p["eval"][str(te)] if str(te) in p["eval"] else p["eval"][te]
            M[i, tr, te], D[i, tr, te], F[i, tr, te] = e["return"], e["d_delta"], e["failed"]
    fx = res["fixed_controller"]
    fixed_ret = np.array([fx[str(r)]["return"] for r in range(4)])
    fixed_dd = np.array([fx[str(r)]["d_delta"] for r in range(4)])

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(4)
    diag = np.array([M[:, r, r] for r in range(4)]).T
    diag_d = np.array([D[:, r, r] for r in range(4)]).T
    axs[0].bar(x - 0.2, diag.mean(0), 0.4, yerr=diag.std(0), color="#d63a6a", capsize=3, label="SAC (trained on same regime)")
    axs[0].bar(x + 0.2, fixed_ret, 0.4, color="#222", label="fixed rated-power controller")
    axs[0].set(xticks=x, xticklabels=[RNAME[r] for r in range(4)], ylabel="episode return", title="Exp 1: single-regime baseline")
    axs[0].legend(fontsize=8)
    axs[1].bar(x - 0.2, diag_d.mean(0), 0.4, yerr=diag_d.std(0), color="#d63a6a", capsize=3, label="SAC")
    axs[1].bar(x + 0.2, fixed_dd, 0.4, color="#222", label="fixed controller")
    axs[1].set(xticks=x, xticklabels=[RNAME[r] for r in range(4)], ylabel="δ increase per 2-h episode", title="Degradation growth")
    axs[1].legend(fontsize=8)
    save(fig, "fig2_single_regime.png")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    Mm = M.mean(0)
    im = axs[0].imshow(Mm, cmap="RdYlGn")
    for a in range(4):
        for b in range(4):
            axs[0].text(b, a, f"{Mm[a, b]:.0f}", ha="center", va="center", fontsize=10,
                        fontweight="bold" if a == b else None)
    axs[0].set(xticks=range(4), yticks=range(4), xticklabels=[f"test R{r}" for r in range(4)],
               yticklabels=[f"train R{r}" for r in range(4)], title="Exp 2: zero-shot return (mean of seeds)")
    axs[0].grid(False)
    fig.colorbar(im, ax=axs[0], fraction=0.046)
    gap = np.array([[M[:, te, te].mean() - M[:, tr, te].mean() for te in range(4)] for tr in range(4)])
    Fm = F.mean(0)
    im2 = axs[1].imshow(Fm * 100, cmap="Reds", vmin=0, vmax=max(50, Fm.max() * 100))
    for a in range(4):
        for b in range(4):
            axs[1].text(b, a, f"{Fm[a, b] * 100:.0f}%", ha="center", va="center", fontsize=10)
    axs[1].set(xticks=range(4), yticks=range(4), xticklabels=[f"test R{r}" for r in range(4)],
               yticklabels=[f"train R{r}" for r in range(4)], title="Critical-failure rate (δ ≥ 0.9)")
    axs[1].grid(False)
    fig.colorbar(im2, ax=axs[1], fraction=0.046)
    save(fig, "fig3_cross_regime.png")
    return M, D, F, gap, fixed_ret, fixed_dd


def collect(res, target, b2=None):
    b2 = b2 if b2 is not None else res["reward"]["b2"]
    out = {}
    for a in res["adaptation"]:
        if a["target"] == target and abs(a["b2"] - b2) < 1e-6:
            out.setdefault(a["method"], []).append(a["curve"])
    return out


def arr(curves, key):
    return np.array([[c[key] for c in cv] for cv in curves])


def fig_adaptation(res, oracle, fixed_ret):
    Ks = res["budget"]["adapt_K"]
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5))
    for row, target in enumerate((2, 3)):
        cur = collect(res, target)
        for m in LABEL:
            if m not in cur:
                continue
            for col, (key, ylab) in enumerate((("return", "evaluation return"),
                                               ("d_delta", "δ increase per episode"),
                                               ("failed", "critical-failure rate"))):
                A = arr(cur[m], key)
                mu, sd = A.mean(0), A.std(0)
                ax = axs[row, col]
                ax.plot(Ks, mu, "o-", color=C[m], label=LABEL[m], lw=2)
                lo = mu - sd if key == "return" else np.clip(mu - sd, 0, None)
                ax.fill_between(Ks, lo, mu + sd, color=C[m], alpha=0.15)
                ax.set(xlabel="adaptation episodes K", ylabel=ylab, xticks=Ks)
        axs[row, 0].axhline(oracle[target], ls="--", c="#2a9d4b", lw=1.2, label=f"SAC trained on R{target} (oracle)")
        axs[row, 0].axhline(fixed_ret[target], ls=":", c="#222", lw=1.2, label="fixed controller")
        tag = "UNSEEN in meta-training" if target == 2 else "seen in meta-training"
        axs[row, 0].set_title(f"Target {RNAME[target]} ({tag})", fontweight="bold", loc="left")
    axs[0, 0].legend(fontsize=7.5, loc="lower right")
    save(fig, "fig4_adaptation_curves.png")


def adaptation_speed(curves, Ks, oracle, fixed, frac=0.9):
    """Episodes needed to recover `frac` of the oracle's improvement over the fixed controller."""
    ks = []
    for cv in curves:
        hit = [c["K"] for c in cv if (c["return"] - fixed) >= frac * (oracle - fixed)]
        ks.append(min(hit) if hit else np.inf)
    return ks


def fig_ablation(res):
    b2s = sorted({a["b2"] for a in res["adaptation"]})
    if len(b2s) < 2:
        return None
    Ks = res["budget"]["adapt_K"]
    abl_seeds = sorted({a["seed"] for a in res["adaptation"] if a["b2"] != res["reward"]["b2"]})
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    cols = ["#7fb3e6", "#d63a6a", "#6b1f3a"]
    rows = []
    for b2, col in zip(b2s, cols):
        def by_seed(method):
            return {x["seed"]: x["curve"] for x in res["adaptation"]
                    if x["target"] == 2 and abs(x["b2"] - b2) < 1e-6 and x["method"] == method}
        fm, sc = by_seed("fomaml"), by_seed("scratch")
        f = [fm[k] for k in abl_seeds if k in fm and k in sc]
        s = [sc[k] for k in abl_seeds if k in fm and k in sc]
        if not f or not s:
            continue
        diff = arr(f, "return") - arr(s, "return")
        dd = arr(f, "d_delta") - arr(s, "d_delta")
        ax.plot(Ks, diff.mean(0), "o-", c=col, lw=2, label=f"β₂ = {b2:.0f}")
        ax.fill_between(Ks, diff.mean(0) - diff.std(0), diff.mean(0) + diff.std(0), color=col, alpha=0.12)
        rows.append((b2, diff.mean(0), dd.mean(0)))
    ax.axhline(0, c="#222", lw=1)
    ax.set(xlabel="adaptation episodes K", ylabel="return: meta-learned (d) − scratch", xticks=Ks,
           title=f"Ablation on β₂ (target R2, {len(abl_seeds)} seeds)")
    ax.legend(fontsize=8)
    save(fig, "fig6_beta2_ablation.png")
    return rows


def fig_training(res):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for kind, col, lab in (("joint", C["joint"], "joint multi-regime SAC"),
                           ("fomaml", C["fomaml"], "meta-learning, Reptile update"),
                           ("fomaml_q", C["fomaml_q"], "meta-learning, query-gradient update")):
        logs = [p["train_log"] for p in res["pretrain"] if p["kind"] == kind and p["b2"] == res["reward"]["b2"]]
        for ax, key in zip(axs, ("return", "d_delta")):
            A = np.array([[l[key] for l in lg] for lg in logs])
            w = 15
            sm = np.array([np.convolve(a, np.ones(w) / w, mode="valid") for a in A])
            x = np.arange(sm.shape[1]) + w
            ax.plot(x, sm.mean(0), c=col, lw=2, label=lab)
            ax.fill_between(x, sm.mean(0) - sm.std(0), sm.mean(0) + sm.std(0), color=col, alpha=0.15)
    axs[0].set(xlabel="training episode", ylabel="return (moving avg 15)", title="Pre-training on R0, R1, R3")
    axs[1].set(xlabel="training episode", ylabel="δ increase per episode", title="Degradation during pre-training")
    axs[0].legend(fontsize=8)
    save(fig, "fig7_pretraining_curves.png")


def fig_trajectory(res, budget):
    """δ, bearing temperature and power over one R2 test episode."""
    seed = 0
    pols = {"fixed controller": (fixed_controller, C["fixed"])}
    for kind, lab, col in (("single0", "SAC trained on R0 (zero-shot)", C["finetune_R0"]),
                           ("single2", "SAC trained on R2", "#2a9d4b"),
                           ("fomaml", "Meta-learned init, Reptile (K = 0)", C["fomaml"]), ("joint", "Joint multi-regime SAC (K = 0)", C["joint"])):
        path = os.path.join(RES, "models", f"seed{seed}_{kind}.pt")
        if os.path.exists(path):
            ag = new_agent(budget, seed)
            ag.load_state_dict(torch.load(path))
            pols[lab] = (greedy(ag), col)
    env = WindTurbineEnv(eval_task(2), split="test")
    seg = env.bank.test[np.random.default_rng(10_003).integers(len(env.bank.test))]
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.4))
    for lab, (pol, col) in pols.items():
        env.rng = np.random.default_rng(7)
        o, _ = env.reset(options={"wind_segment": seg, "delta0": 0.70})
        tr = {"delta": [env.delta], "T_b": [env.T_b], "power": [o[4]], "pitch": [0.0]}
        while True:
            o, r, te, tc, info = env.step(pol(o))
            for k in ("delta", "T_b", "power"):
                tr[k].append(info[k])
            tr["pitch"].append(env.beta_off)
            if te or tc:
                break
        t = np.arange(len(tr["delta"])) * 0.5
        for ax, k in zip(axs, ("delta", "T_b", "power", "pitch")):
            ax.plot(t, tr[k], c=col, lw=1.6, label=lab)
    axs[0].axhline(0.9, c="r", ls="--", lw=1)
    axs[0].set(title="degradation δ", xlabel="time [min]")
    axs[1].axhline(70, c="r", ls=":", lw=1)
    axs[1].set(title="bearing temperature [°C]", xlabel="time [min]")
    axs[2].set(title="power / rated", xlabel="time [min]")
    axs[3].set(title="pitch offset chosen by agent [deg]", xlabel="time [min]")
    axs[0].legend(fontsize=7.5)
    save(fig, "fig5_episode_trajectories_R2.png")


def fmt(a):
    a = np.asarray(a, float)
    return f"{a.mean():.3f} ± {a.std():.3f}" if abs(a.mean()) < 1 and a.std() < 1 else f"{a.mean():.1f} ± {a.std():.1f}"


def main():
    res = json.load(open(os.path.join(RES, "results.json")))
    budget = Budget(**{k: tuple(v) if isinstance(v, list) else v for k, v in res["budget"].items()})
    fig_pipeline()
    rmse = fig_data()
    M, D, F, gap, fixed_ret, fixed_dd = fig_single_and_cross(res)
    oracle = {r: M[:, r, r].mean() for r in range(4)}
    fig_adaptation(res, oracle, fixed_ret)
    fig_trajectory(res, budget)
    abl = fig_ablation(res)
    fig_training(res)

    Ks = res["budget"]["adapt_K"]
    L = ["# Results\n", f"Seeds: {len(res['seeds'])} (mean ± std). Episode = 2 h of turbine operation "
         f"(240 steps x 30 s). Evaluation: {budget.n_eval} fixed episodes with unseen Oct-Dec 2018 wind. "
         f"Simulator power-curve RMSE vs measured SCADA curve: {rmse:.3f} of rated power.\n",
         "## Exp 1: single-regime baseline\n",
         "| Regime | SAC return | Fixed return | SAC δ increase | Fixed δ increase | SAC failure rate | Fixed failure rate |",
         "|---|---|---|---|---|---|---|"]
    for r in range(4):
        L.append(f"| {RNAME[r]} | {fmt(M[:, r, r])} | {fixed_ret[r]:.1f} | {fmt(D[:, r, r])} | {fixed_dd[r]:.4f} | "
                 f"{fmt(F[:, r, r])} | {res['fixed_controller'][str(r)]['failed']:.3f} |")
    L += ["\n## Exp 2: cross-regime return (rows = trained on, columns = tested on)\n",
          "| | " + " | ".join(f"test R{r}" for r in range(4)) + " |", "|---" * 5 + "|"]
    for tr in range(4):
        L.append(f"| train R{tr} | " + " | ".join(f"{M[:, tr, te].mean():.1f}" for te in range(4)) + " |")
    L += ["\nGeneralisation gap (same-regime minus cross-regime return):\n",
          "| | " + " | ".join(f"test R{r}" for r in range(4)) + " |", "|---" * 5 + "|"]
    for tr in range(4):
        L.append(f"| train R{tr} | " + " | ".join(f"{gap[tr, te]:.1f}" for te in range(4)) + " |")
    for target in (2, 3):
        cur = collect(res, target)
        tag = "unseen during meta-training" if target == 2 else "seen during meta-training"
        L += [f"\n## Exp 3-4: adaptation to {RNAME[target]} ({tag})\n",
              "| Method | " + " | ".join(f"return K={k}" for k in Ks) + " | δ increase K=5 | failure K=0 | failure K=5 | episodes to 90 % of oracle (per seed) |",
              "|---" * (len(Ks) + 5) + "|"]
        for m in LABEL:
            if m not in cur:
                continue
            R = arr(cur[m], "return")
            Dd = arr(cur[m], "d_delta")
            Fr = arr(cur[m], "failed")
            i5 = Ks.index(5) if 5 in Ks else -1
            sp = adaptation_speed(cur[m], Ks, oracle[target], fixed_ret[target])
            sps = ", ".join("never" if np.isinf(s) else str(int(s)) for s in sp)
            L.append(f"| {LABEL[m]} | " + " | ".join(fmt(R[:, j]) for j in range(len(Ks)))
                     + f" | {fmt(Dd[:, i5])} | {fmt(Fr[:, 0])} | {fmt(Fr[:, i5])} | {sps} |")
        L.append(f"\nOracle (SAC trained on R{target}): {oracle[target]:.1f}; fixed controller: {fixed_ret[target]:.1f}.")
    if abl:
        L += ["\n## Ablation on β₂ (target R2, seeds 0-2): method (d) minus scratch\n",
              "| β₂ | " + " | ".join(f"Δreturn K={k}" for k in Ks) + " | " + " | ".join(f"Δ(δ increase) K={k}" for k in Ks) + " |",
              "|---" * (2 * len(Ks) + 1) + "|"]
        for b2, dr, dd in abl:
            L.append(f"| {b2:.0f} | " + " | ".join(f"{x:.1f}" for x in dr) + " | " + " | ".join(f"{x:+.4f}" for x in dd) + " |")
    open(os.path.join(RES, "RESULTS.md"), "w").write("\n".join(L) + "\n")
    print("wrote results/RESULTS.md")


if __name__ == "__main__":
    main()
