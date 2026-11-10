"""
Degradation-aware wind turbine environment (Gymnasium).

A simplified variable-speed turbine, sized to match the real 3.6 MW turbine in
the SCADA data, driven by real measured wind. It follows the proposal:

  J dw/dt     = T_aero - T_gen - B(1 + lam*delta) w
  d delta/dt  = k1 w^2 + k2 max(T_bear - T_th, 0)^2 + k3 |vibration|

so aggressive operation (high speed, hot bearing, large accelerations) wears
the turbine faster, and a worn turbine (high delta) runs hotter and loses more
to friction.

Control structure
-----------------
The turbine has a standard inner controller (optimal-torque law below rated,
PI pitch control above rated). The RL agent acts as a *health-aware supervisor*
that adjusts it every 30 s with two bounded continuous actions:
  a[0] -> change of the blade-pitch offset  (+/- 2 deg per step, offset in [0, 20] deg)
  a[1] -> change of the generator-torque gain (+/- 0.05 per step, gain in [0.6, 1.3])
With both offsets at their neutral value the turbine is exactly the fixed
rated-power controller, which is the non-learning reference.

Observation (8 values, normalised): rotor speed w, wind v, generator torque,
bearing temperature, power output, degradation delta  (the proposal's 6 state
variables) plus the two current offsets, which the agent needs because its
actions are increments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .data import P_RATED_KW, WindBank, get_wind_bank

# ---- Health regimes (= meta-learning tasks) -------------------------------
REGIMES = {
    0: ("R0 healthy", 0.00, 0.25),
    1: ("R1 mild", 0.25, 0.50),
    2: ("R2 moderate", 0.50, 0.75),
    3: ("R3 critical", 0.75, 0.85),  # start below 0.9 so failure is possible, not certain
}
DELTA_CRIT = 0.90


def cp_heier(lam: float, beta: float) -> float:
    """Heier power-coefficient model, max ~0.48 at lam ~8.1, beta = 0."""
    inv = 1.0 / (lam + 0.08 * beta) - 0.035 / (beta ** 3 + 1.0)
    cp = 0.5176 * (116.0 * inv - 0.4 * beta - 5.0) * math.exp(-21.0 * inv) + 0.0068 * lam
    return max(cp, 0.0)


@dataclass
class TurbineParams:
    # Rotor / drivetrain (sized to the 3.6 MW SCADA turbine)
    R: float = 55.0            # rotor radius [m]
    rho: float = 1.225
    eta: float = 0.90          # Cp scale, calibrated to the measured power curve
    lam_opt: float = 8.1
    cp_max: float = 0.48
    P_rated: float = P_RATED_KW * 1e3
    tau_mech: float = 8.0      # rotor time constant at rated [s] (sets J)
    fric_frac: float = 0.02    # friction loss at rated, as fraction of rated torque
    # Degradation coupling (task-randomised for meta-learning)
    lam_fric: float = 1.0      # friction grows as B(1 + lam*delta)
    k1: float = 0.002          # per hour, speed term
    k2: float = 0.0015         # per hour per degC^2, thermal term
    k3: float = 0.003          # per hour, vibration term
    kappa_heat: float = 1.0    # worn bearing heats up as (1 + kappa*delta)
    # Thermal
    T_amb: float = 25.0
    dT_rated: float = 25.0     # healthy bearing rise at rated [degC]
    tau_th: float = 1200.0     # thermal time constant [s]
    T_th: float = 60.0         # wear threshold
    T_safe: float = 70.0       # reward threshold
    T_max: float = 95.0        # unsafe
    # Turbulence on top of the real 10-min wind
    TI: float = 0.10
    tau_turb: float = 10.0

    @property
    def area(self):
        return math.pi * self.R ** 2

    @property
    def v_rated(self):
        return (self.P_rated / (self.eta * 0.5 * self.rho * self.area * self.cp_max)) ** (1 / 3)

    @property
    def w_rated(self):
        return self.lam_opt * self.v_rated / self.R

    @property
    def T_rated(self):
        return self.P_rated / self.w_rated

    @property
    def J(self):
        return self.tau_mech * self.T_rated / self.w_rated

    @property
    def B(self):
        return self.fric_frac * self.T_rated / self.w_rated

    @property
    def K_opt(self):  # optimal-torque gain, T = K w^2
        return self.eta * 0.5 * self.rho * math.pi * self.R ** 5 * self.cp_max / self.lam_opt ** 3


@dataclass
class RewardWeights:
    """r = b1 P/P_rated - b2 (delta_{t+1}-delta_t) - b3 max(T_b - T_safe, 0) - b4 1_unsafe
    Fixed for every method so differences come from adaptation, not reward tuning."""
    b1: float = 1.0
    b2: float = 2000.0
    b3: float = 0.05
    b4: float = 2.0
    fail: float = 50.0   # one-off penalty when delta crosses the critical threshold


@dataclass
class Task:
    regime: int
    params: TurbineParams = field(default_factory=TurbineParams)

    @staticmethod
    def sample(regime: int, rng: np.random.Generator, randomise: bool = True) -> "Task":
        p = TurbineParams()
        if randomise:
            p = replace(
                p,
                lam_fric=float(rng.uniform(0.5, 1.5)),
                k1=p.k1 * float(rng.uniform(0.75, 1.25)),
                k2=p.k2 * float(rng.uniform(0.75, 1.25)),
                k3=p.k3 * float(rng.uniform(0.75, 1.25)),
            )
        return Task(regime, p)


class WindTurbineEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, task: Task | None = None, split: str = "train",
                 steps_per_episode: int = 240, control_dt: float = 30.0, sub_dt: float = 1.0,
                 reward: RewardWeights | None = None, wind_bank: WindBank | None = None,
                 seed: int | None = None):
        super().__init__()
        self.task = task or Task(0)
        self.split = split
        self.n_steps = steps_per_episode
        self.dt = control_dt
        self.sub_dt = sub_dt
        self.n_sub = int(round(control_dt / sub_dt))
        self.rw = reward or RewardWeights()
        seg_len = int(math.ceil(steps_per_episode * control_dt / 600.0)) + 2
        self.bank = wind_bank or get_wind_bank(seg_len)
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def set_task(self, task: Task):
        self.task = task

    # ------------------------------------------------------------------
    def _wind_at(self, t: float) -> float:
        x = t / 600.0
        i = int(x)
        f = x - i
        return float(self.seg[i] * (1 - f) + self.seg[i + 1] * f)

    def _steady_bearing_temp(self, wn: float, tn: float, delta: float) -> float:
        p = self.task.params
        return p.T_amb + p.dT_rated * wn * wn * (0.5 + 0.5 * tn) * (1 + p.kappa_heat * delta)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        opts = options or {}
        p = self.task.params
        self.seg = opts.get("wind_segment", self.bank.sample(self.rng, self.split))
        _, lo, hi = REGIMES[self.task.regime]
        self.delta = float(opts.get("delta0", self.rng.uniform(lo, hi)))
        self.t = 0.0
        self.k = 0
        self.turb = 0.0
        self.beta_off = 0.0
        self.k_gain = 1.0
        self.beta_pi = 0.0
        v = self._wind_at(0.0)
        self.w = min(p.lam_opt * max(v, 3.0) / p.R, p.w_rated)
        self.T_gen = min(p.K_opt * self.w ** 2, p.T_rated)
        self.T_b = self._steady_bearing_temp(self.w / p.w_rated, self.T_gen / p.T_rated, self.delta)
        self.P = 0.0
        # short burn-in with the neutral controller so the rotor is near equilibrium
        for _ in range(60):
            self._substep(advance_time=False)
        self.T_b = self._steady_bearing_temp(self.w / p.w_rated, self.T_gen / p.T_rated, self.delta)
        self.delta_start = self.delta
        return self._obs(), {"delta": self.delta}

    def _obs(self) -> np.ndarray:
        p = self.task.params
        return np.array([
            self.w / p.w_rated,
            self.v / 25.0,
            self.T_gen / p.T_rated,
            (self.T_b - p.T_amb) / 50.0,
            self.P / p.P_rated,
            self.delta,
            self.beta_off / 20.0,
            (self.k_gain - 1.0) / 0.3,
        ], dtype=np.float32)

    def _substep(self, advance_time: bool = True):
        """One 1-second integration step of rotor, inner controller, heat and wear."""
        p = self.task.params
        h = self.sub_dt
        # wind = real 10-min mean (interpolated) x (1 + TI * OU turbulence)
        a = math.exp(-h / p.tau_turb)
        self.turb = a * self.turb + math.sqrt(1 - a * a) * float(self.rng.standard_normal())
        v = max(self._wind_at(self.t) * (1.0 + p.TI * self.turb), 0.1)
        self.v = v
        # inner controller: optimal-torque law + PI pitch above rated, plus RL offsets
        wn = self.w / p.w_rated
        self.beta_pi = min(max(self.beta_pi + h * 8.0 * (wn - 1.0), 0.0), 30.0)
        beta = min(self.beta_pi + 15.0 * max(wn - 1.0, 0.0) + self.beta_off, 30.0)
        T_cmd = self.k_gain * min(p.K_opt * self.w ** 2, p.T_rated)
        self.T_gen = T_cmd
        # aerodynamics
        lam = self.w * p.R / v
        cp = p.eta * cp_heier(lam, beta)
        P_aero = 0.5 * p.rho * p.area * cp * v ** 3
        T_aero = P_aero / max(self.w, 0.05)
        # rotor: J dw/dt = T_aero - T_gen - B(1 + lam*delta) w
        dw = (T_aero - self.T_gen - p.B * (1 + p.lam_fric * self.delta) * self.w) / p.J
        self.w = min(max(self.w + h * dw, 0.05), 1.5 * p.w_rated)
        self.P = self.T_gen * self.w
        if advance_time:
            wn = self.w / p.w_rated
            tn = self.T_gen / p.T_rated
            # bearing temperature relaxes to its operating-point steady state
            Tss = self._steady_bearing_temp(wn, tn, self.delta)
            self.T_b += h * (Tss - self.T_b) / p.tau_th
            # vibration: speed-driven, amplified by wear and by rotor acceleration
            vib = wn * wn * (1 + 2 * self.delta) * (1 + 20.0 * abs(dw) / p.w_rated)
            rate_h = (p.k1 * wn * wn + p.k2 * max(self.T_b - p.T_th, 0.0) ** 2 + p.k3 * abs(vib))
            self.delta = min(self.delta + rate_h * h / 3600.0, 1.0)
            self.t += h

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64).ravel(), -1.0, 1.0)
        self.beta_off = float(np.clip(self.beta_off + 2.0 * a[0], 0.0, 20.0))
        self.k_gain = float(np.clip(self.k_gain + 0.05 * a[1], 0.6, 1.3))
        p = self.task.params
        d0 = self.delta
        E, unsafe = 0.0, False
        for _ in range(self.n_sub):
            self._substep()
            E += self.P
            if self.w > 1.2 * p.w_rated or self.T_b > p.T_max or self.P > 1.15 * p.P_rated:
                unsafe = True
        P_mean = E / self.n_sub
        self.k += 1
        failed = self.delta >= DELTA_CRIT
        rw = self.rw
        r = (rw.b1 * P_mean / p.P_rated - rw.b2 * (self.delta - d0)
             - rw.b3 * max(self.T_b - p.T_safe, 0.0) - rw.b4 * float(unsafe or failed))
        if failed:
            r -= rw.fail
        terminated = bool(failed)
        truncated = self.k >= self.n_steps
        info = {"power": P_mean / p.P_rated, "delta": self.delta, "T_b": self.T_b,
                "unsafe": unsafe, "failed": failed, "wind": self.v}
        return self._obs(), float(r), terminated, truncated, info
