"""
Real wind data for the simulator.

Source: Kaggle "Wind Turbine Scada Dataset" (B. Erisen), one 3.6 MW turbine in
Turkey, 10-minute SCADA records for all of 2018 (50,530 rows). Columns used:
hub-height wind speed, active power and the manufacturer's theoretical curve.

What the data is used for
-------------------------
1. Wind input: every episode replays a real 10-minute wind-speed segment
   (linearly interpolated to the simulator step) plus small-scale turbulence.
2. Calibration: the rotor's power coefficient is scaled so the simulator's
   steady-state power curve matches the turbine's *measured* power curve.
3. Train/test split by time: Jan-Sep 2018 for training, Oct-Dec 2018 for
   evaluation, so test wind is never seen during training.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "T1.csv")

P_RATED_KW = 3600.0
CUT_IN, CUT_OUT = 3.5, 25.0


def load_scada(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = ["time", "power_kw", "wind", "power_theory_kw", "direction"]
    df["time"] = pd.to_datetime(df["time"], format="%d %m %Y %H:%M")
    df = df.sort_values("time").reset_index(drop=True)
    # Reindex to a regular 10-min grid; short gaps (<= 1 h) are interpolated.
    df = df.set_index("time").asfreq("10min")
    df["wind"] = df["wind"].interpolate(limit=6)
    df["power_kw"] = df["power_kw"].clip(lower=0.0)
    return df


def clean_for_power_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Drop downtime / curtailment rows (near-zero power above cut-in, or power far
    below the theoretical curve). Standard SCADA power-curve filtering."""
    d = df.dropna(subset=["wind", "power_kw"])
    bad = (d["wind"] > CUT_IN + 0.5) & (d["power_kw"] < 0.6 * d["power_theory_kw"] - 50)
    return d[~bad]


def binned_power_curve(df: pd.DataFrame, width: float = 0.5):
    """Median measured power per wind-speed bin (normalised by rated power)."""
    d = clean_for_power_curve(df)
    edges = np.arange(0, 25 + width, width)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(d["wind"].values, edges) - 1
    med = np.full(len(centres), np.nan)
    cnt = np.zeros(len(centres), dtype=int)
    for i in range(len(centres)):
        m = idx == i
        cnt[i] = m.sum()
        if cnt[i] >= 20:
            med[i] = np.median(d["power_kw"].values[m]) / P_RATED_KW
    ok = ~np.isnan(med)
    return centres[ok], med[ok], cnt[ok]


@dataclass
class WindBank:
    """Pools of real wind segments for train and test."""
    train: np.ndarray  # (n_segments, seg_len) 10-min wind speeds
    test: np.ndarray
    seg_len: int

    def sample(self, rng: np.random.Generator, split: str = "train") -> np.ndarray:
        pool = self.train if split == "train" else self.test
        return pool[rng.integers(len(pool))]


def build_wind_bank(df: pd.DataFrame, seg_len: int, min_mean: float = 4.0,
                    test_from: str = "2018-10-01") -> WindBank:
    """Cut the year into non-overlapping segments of `seg_len` 10-min points.
    Segments with gaps or with mean wind below `min_mean` (turbine idle) are
    dropped, because the controller has nothing to do in calm air."""
    w = df["wind"]
    out = {}
    for name, part in (("train", w[w.index < test_from]), ("test", w[w.index >= test_from])):
        v = part.values
        n = len(v) // seg_len
        segs = v[: n * seg_len].reshape(n, seg_len)
        keep = (~np.isnan(segs).any(1)) & (segs.mean(1) >= min_mean)
        out[name] = segs[keep].astype(np.float32)
    return WindBank(out["train"], out["test"], seg_len)


_CACHE: dict = {}


def get_wind_bank(seg_len: int) -> WindBank:
    key = ("bank", seg_len)
    if key not in _CACHE:
        _CACHE[key] = build_wind_bank(load_scada(), seg_len)
    return _CACHE[key]
