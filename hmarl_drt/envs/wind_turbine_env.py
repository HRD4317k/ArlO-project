"""
Wind Turbine Farm Environment for HMARL-DRT.

Simulates turbine dynamics via ODE system:
  dω/dt = (1/J)(T_aero(λ,β) - T_gen - Bω)
  dT/dt = (1/C_th)(P_loss - h(T - T_amb))
  dδ/dt = f_deg(ω, T, F_vib)

State: [omega, v_wind, T_gen, T_bearing, P_out, delta, grid_demand, weather_forecast]
Action: [pitch_angle_delta, torque_setpoint_delta, maintenance_flag_continuous]
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class WindTurbineEnv(gym.Env):
    """Custom Gym environment for a single wind turbine with degradation."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict = None):
        super().__init__()
        cfg = config or {}
        phys = cfg.get("physics", {})

        # Physical parameters
        self.J = float(phys.get("J", 1e6))
        self.B = float(phys.get("B", 100.0))
        self.R = float(phys.get("R", 40.0))
        self.rho = float(phys.get("rho", 1.225))
        self.C_th = float(phys.get("C_th", 5000.0))
        self.h_conv = float(phys.get("h_conv", 10.0))
        # Multiplier on the degradation rate. 1.0 = baseline; higher values
        # accelerate degradation so regime transitions occur within an episode
        # (useful for testing degradation-control and adaptation).
        self.deg_acceleration = float(phys.get("deg_acceleration", 1.0))
        # "quadratic": original 0.01*max(T-60,0)^2 cliff; "arrhenius": smooth exponential in T
        self.temp_deg_model = str(phys.get("temp_deg_model", "quadratic"))
        self.dt = float(cfg.get("dt", 0.1))
        self.max_steps = int(cfg.get("max_steps", 1000))

        # Constraints
        constr = cfg.get("constraints", {})
        self.omega_min = float(constr.get("omega_min", 0.5))
        self.omega_max = float(constr.get("omega_max", 1.5))
        self.T_max = float(constr.get("T_max", 100.0))
        self.delta_crit = float(constr.get("delta_crit", 0.9))

        # Rated power
        self.P_rated = 5e6  # 5 MW rated

        # State: [omega, v_wind, T_gen, T_bearing, P_out, delta, grid_demand, weather]
        state_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        state_high = np.array([2.0, 30.0, 1.0, 150.0, 1.5, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=state_low, high=state_high, dtype=np.float32)

        # Action: [delta_pitch, delta_torque, maintenance_signal] all continuous [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        self.state = None
        self.step_count = 0
        self._rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        omega = self._rng.uniform(0.8, 1.2)
        v_wind = self._rng.uniform(5.0, 15.0)
        T_gen = self._rng.uniform(0.3, 0.7)
        T_bearing = self._rng.uniform(30.0, 50.0)
        P_out = self._rng.uniform(0.2, 0.6)
        if options and "delta_range" in options:
            lo, hi = options["delta_range"]
            delta = self._rng.uniform(float(lo), float(hi))
        else:
            delta = self._rng.uniform(0.0, 0.2)
        grid_demand = self._rng.uniform(0.3, 0.9)
        weather = self._rng.uniform(0.3, 0.8)

        self.state = np.array(
            [omega, v_wind, T_gen, T_bearing, P_out, delta, grid_demand, weather],
            dtype=np.float32,
        )
        self.step_count = 0
        self._base_wind = v_wind
        return self.state.copy(), {"delta": float(delta)}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).flatten()
        action = np.clip(action, -1.0, 1.0)

        # Unpack as native Python floats to avoid numpy dtype issues
        omega = float(self.state[0])
        v_wind = float(self.state[1])
        T_gen = float(self.state[2])
        T_bearing = float(self.state[3])
        P_out = float(self.state[4])
        delta = float(self.state[5])
        grid_demand = float(self.state[6])
        weather = float(self.state[7])

        a0 = float(action[0])
        a1 = float(action[1])
        a2 = float(action[2])

        # Map actions to physical quantities
        delta_pitch = a0 * 15.0      # up to ±15 degrees
        delta_torque = a1 * 0.1       # torque setpoint adjustment
        maintenance_signal = (a2 + 1) / 2  # map to [0, 1]

        beta = max(0.0, min(30.0, 15.0 + delta_pitch))
        T_gen_new = max(0.0, min(1.0, T_gen + delta_torque))

        # -- Aerodynamic torque --
        lam = omega * self.R / max(v_wind, 0.1)
        # Simplified Cp approximation
        denom = lam + 0.08 * beta + 1e-6
        C_Q = max(-0.1, min(0.5,
            0.5 * (0.22 * (116.0 / denom - 0.4 * beta - 5.0)
                   * float(np.exp(-12.5 / denom)))
        ))
        T_aero = 0.5 * self.rho * np.pi * self.R**3 * v_wind**2 * C_Q

        # -- ODE: dω/dt --
        domega_dt = (T_aero - T_gen_new * 1e6 - self.B * omega) / self.J
        omega_new = max(0.0, min(2.0, omega + domega_dt * self.dt))

        # -- Power output --
        # T_gen and omega are normalized units ([0,1] and [0,2]); dividing by
        # P_rated (5e6 W) was dimensionally wrong and made P_out ~0 always.
        # Normalize by the max achievable product instead: T_gen_max * omega_max.
        P_out_new = max(0.0, min(1.5, T_gen_new * omega_new / (1.0 * 2.0)))

        # -- Thermal dynamics: dT/dt --
        P_loss = 0.02 * (omega_new**2 + T_gen_new**2) * 1e6
        dT_dt = (P_loss - self.h_conv * (T_bearing - 25.0)) / self.C_th
        T_bearing_new = max(0.0, min(150.0, T_bearing + dT_dt * self.dt))

        # -- Degradation dynamics: dδ/dt --
        vibration = 0.5 * omega_new**2 + 0.1 * float(self._rng.standard_normal())
        if self.temp_deg_model == "arrhenius":
            # Rate roughly doubles every ~14 C, no hard threshold
            temp_term = 0.5 * float(np.exp((T_bearing_new - 60.0) / 20.0))
        else:
            temp_term = 0.01 * max(T_bearing_new - 60.0, 0) ** 2
        deg_rate = 1e-5 * (omega_new**2 + temp_term + 0.1 * abs(vibration))
        deg_rate *= self.deg_acceleration
        if maintenance_signal > 0.7:
            deg_rate *= 0.1  # Maintenance slows degradation
        delta_new = max(0.0, min(1.0, delta + deg_rate * self.dt))

        # -- Wind variation --
        v_wind_new = max(0.0, min(30.0,
            self._base_wind + 3.0 * float(np.sin(0.01 * self.step_count))
            + 0.5 * float(self._rng.standard_normal())
        ))

        # -- Grid demand & weather variation --
        grid_new = max(0.0, min(1.0, grid_demand + 0.01 * float(self._rng.standard_normal())))
        weather_new = max(0.0, min(1.0, weather + 0.005 * float(self._rng.standard_normal())))

        # Update state
        self.state = np.array(
            [omega_new, v_wind_new, T_gen_new, T_bearing_new, P_out_new, delta_new, grid_new, weather_new],
            dtype=np.float32,
        )
        self.step_count += 1

        # -- Termination --
        terminated = bool(delta_new >= self.delta_crit)
        truncated = bool(self.step_count >= self.max_steps)

        info = {
            "omega": float(omega_new),
            "T_bearing": float(T_bearing_new),
            "P_out": float(P_out_new),
            "delta": float(delta_new),
            "v_wind": float(v_wind_new),
            "T_aero": float(T_aero),
            "physics_violated": bool(
                omega_new > self.omega_max or omega_new < self.omega_min or T_bearing_new > self.T_max
            ),
        }

        # Simple placeholder reward (the Dynamic Reward Shaper computes the real one)
        reward = float(P_out_new - 0.5 * max(delta_new - 0.5, 0))

        return self.state.copy(), reward, terminated, truncated, info
