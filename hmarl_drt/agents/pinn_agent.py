"""
Agent 4: Physics-Informed Neural Network (PINN) Agent.

ODE System:
  dω/dt = (1/J)(T_aero(λ,β) - T_gen - Bω)
  dT/dt = (1/C_th)(P_loss - h(T - T_amb))

L_PINN = L_data + λ₁·L_ODE + λ₂·L_BC

Provides: safe action bounds, physics policy, ODE residual for transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PINNNetwork(nn.Module):
    """Neural network that learns turbine dynamics f: (s,a) → ds/dt."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class PINNPolicy(nn.Module):
    """
    Learnable physics-informed control policy: state → action.

    Trained to (a) imitate the hand-coded safe controller and (b) avoid
    actions whose predicted next-state (via the learned dynamics model)
    violates the omega/T_bearing constraints. This makes the control path
    trainable — unlike the fixed proportional controller — so PINN's safety
    behaviour can improve with training.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(state))


class PINNAgent:
    """
    Physics agent using ODE + PINN for safe control.
    Primary agent in early training when data is sparse.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        J: float = 1e6,
        B: float = 100.0,
        R: float = 40.0,
        rho: float = 1.225,
        omega_min: float = 0.5,
        omega_max: float = 1.5,
        T_max: float = 100.0,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.J = J
        self.B = B
        self.R = R
        self.rho = rho
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.T_max = T_max
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.pinn = PINNNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.opt = torch.optim.Adam(self.pinn.parameters(), lr=lr)

        # Learnable control policy (state → action), trained to imitate the
        # safe controller and avoid constraint-violating next states.
        self.policy = PINNPolicy(state_dim, action_dim, hidden_dim).to(device)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Residual correction network (updated by NN agent data)
        self.residual_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim),
        ).to(device)
        self.res_opt = torch.optim.Adam(self.residual_net.parameters(), lr=lr)

    def aerodynamic_torque(self, omega: torch.Tensor, v_wind: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """T_aero = 0.5·ρ·π·R³·v²·C_Q(λ,β)"""
        lam = omega * self.R / (v_wind + 1e-6)
        denom = lam + 0.08 * beta + 1e-6
        C_Q = 0.5 * (0.22 * (116.0 / denom - 0.4 * beta - 5.0) * torch.exp(-12.5 / denom))
        C_Q = C_Q.clamp(-0.1, 0.5)
        return 0.5 * self.rho * np.pi * self.R**3 * v_wind**2 * C_Q

    def ode_residual(self, state: torch.Tensor, action: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """||dω/dt_pred - (1/J)(T_aero - T_gen - Bω)||²"""
        omega = state[:, 0:1]
        v_wind = state[:, 1:2]
        T_gen = state[:, 2:3]
        beta = action[:, 0:1] * 15.0 + 15.0  # map from [-1,1] to [0,30]

        T_aero = self.aerodynamic_torque(omega, v_wind, beta)
        dw_dt_physics = (T_aero - T_gen * 1e6 - self.B * omega) / self.J

        ds_dt_pred = self.pinn(state, action)
        dw_dt_pred = ds_dt_pred[:, 0:1]

        return F.mse_loss(dw_dt_pred, dw_dt_physics)

    def get_safe_bounds(self, state: np.ndarray) -> dict:
        """Return safe action bounds from physics constraints."""
        omega = state[0] if len(state) > 0 else 1.0
        T = state[3] if len(state) > 3 else 50.0

        beta_min_norm = -1.0
        beta_max_norm = 1.0

        # Tighten bounds near constraint boundaries
        if omega > 0.9 * self.omega_max:
            beta_min_norm = 0.0  # Force pitch increase to reduce speed
        if T > 0.9 * self.T_max:
            beta_min_norm = max(beta_min_norm, 0.3)

        return {
            "low": np.array([beta_min_norm, -1.0, -1.0], dtype=np.float32),
            "high": np.array([beta_max_norm, 1.0, 1.0], dtype=np.float32),
        }

    def act(self, state: np.ndarray) -> np.ndarray:
        """Conservative physics-based action."""
        omega = state[0] if len(state) > 0 else 1.0
        v_wind = state[1] if len(state) > 1 else 10.0
        T = state[3] if len(state) > 3 else 50.0

        # Simple proportional controller based on physics
        pitch = 0.0
        if omega > self.omega_max * 0.8:
            pitch = min(1.0, (omega - self.omega_max * 0.8) / (self.omega_max * 0.2))
        elif omega < self.omega_min * 1.2:
            pitch = max(-1.0, -(self.omega_min * 1.2 - omega) / (self.omega_min * 0.2))

        torque_adj = 0.0
        if T > self.T_max * 0.8:
            torque_adj = -min(1.0, (T - self.T_max * 0.8) / (self.T_max * 0.2))

        maint = -1.0  # no maintenance by default

        return np.array([pitch, torque_adj, maint], dtype=np.float32)

    def policy_act(self, state: np.ndarray) -> np.ndarray:
        """Action from the learnable policy network (tanh-squashed)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.policy(state_t)
        return action.squeeze(0).cpu().numpy()

    def update_policy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        lambda_imitate: float = 1.0,
        lambda_safe: float = 1.0,
    ) -> float:
        """
        Train the learnable policy.

        Loss = λ_imitate · ||π(s) - a_safe(s)||²   (imitate safe controller)
             + λ_safe · Σ max(0, predicted_violation)   (avoid constraint
               violations, using the learned dynamics model to predict the
               next state and penalising predicted omega/T_bearing outside
               the legal range).

        The imitation target is the hand-coded safe controller's output
        computed on the fly from the batch states (not the blended action
        that was actually executed), so the policy learns to reproduce the
        safe controller and then improve on it via the safety term.
        """
        pred_actions = self.policy(states)

        # Imitation target: hand-coded safe controller on the same states
        safe_actions = []
        for i in range(states.shape[0]):
            s_np = states[i].detach().cpu().numpy()
            safe_actions.append(self.act(s_np))
        safe_actions = torch.FloatTensor(np.stack(safe_actions)).to(self.device)
        imitate_loss = F.mse_loss(pred_actions, safe_actions)

        # Safety loss: predict next state via learned dynamics, penalise
        # predicted constraint violations
        with torch.no_grad():
            pred_ds = self.pinn(states, pred_actions)
        pred_next = states + pred_ds * 0.1
        omega_pred = pred_next[:, 0]
        T_pred = pred_next[:, 3]
        omega_viol = F.relu(omega_pred - self.omega_max) + F.relu(self.omega_min - omega_pred)
        T_viol = F.relu(T_pred - self.T_max)
        safe_loss = (omega_viol + T_viol).mean()

        loss = lambda_imitate * imitate_loss + lambda_safe * safe_loss

        self.policy_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_opt.step()
        return loss.item()

    def get_ode_residual_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.ode_residual(state, action)

    def update(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> float:
        """Update PINN with data."""
        pred_ds = self.pinn(states, actions)
        actual_ds = next_states - states
        data_loss = F.mse_loss(pred_ds, actual_ds)
        ode_loss = self.ode_residual(states, actions)
        loss = data_loss + ode_loss

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pinn.parameters(), 1.0)
        self.opt.step()
        return loss.item()

    def update_residual(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> float:
        """Update residual correction from NN agent data."""
        x = torch.cat([states, actions], dim=-1)
        pred = self.pinn(states, actions) + self.residual_net(x)
        loss = F.mse_loss(pred, next_states - states)
        self.res_opt.zero_grad()
        loss.backward()
        self.res_opt.step()
        return loss.item()

    def get_embedding(self, state: torch.Tensor) -> torch.Tensor:
        """Get agent-specific encoding for shared embedding."""
        action_zeros = torch.zeros(state.shape[0], self.action_dim, device=state.device)
        return self.pinn.encoder(state)

    def parameters(self):
        return self.pinn.parameters()

    def eval(self):
        self.pinn.eval()

    def train(self):
        self.pinn.train()

    def zero_grad(self):
        self.pinn.zero_grad()

    def named_parameters(self):
        return self.pinn.named_parameters()
