"""
Agent 2: Gaussian Process Agent for uncertainty quantification.

Uses a simple parametric approximation for scalability (avoids full GP cubic cost).
Provides:
  - Action recommendation μ_a
  - Epistemic uncertainty σ²_epistemic
  - RUL estimate via exponential fit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque


class UncertaintyNetwork(nn.Module):
    """Ensemble-based uncertainty estimator (scalable GP approximation)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, n_models: int = 5):
        super().__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )
            for _ in range(n_models)
        ])

    def forward(self, state: torch.Tensor) -> tuple:
        """Returns mean action and epistemic uncertainty."""
        preds = torch.stack([m(state) for m in self.models], dim=0)  # [N, B, A]
        mean = preds.mean(dim=0)       # [B, A]
        var = preds.var(dim=0)          # [B, A]
        return mean, var


class GPAgent:
    """
    Bayesian agent using ensemble-based uncertainty estimation.
    Provides uncertainty maps for SAC exploration and RUL estimation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        n_models: int = 5,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.network = UncertaintyNetwork(state_dim, action_dim, hidden_dim, n_models).to(device)
        self.opt = torch.optim.Adam(self.network.parameters(), lr=lr)

        # Encoder for shared embedding
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        ).to(device)
        self.enc_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)

        # RUL estimation: simple exponential fit on degradation history
        self.degradation_history = deque(maxlen=500)

    def act(self, state: np.ndarray) -> np.ndarray:
        """Return action recommendation from ensemble mean."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, _ = self.network(state_t)
        return torch.tanh(mean).squeeze(0).cpu().numpy()

    def get_uncertainty(self, states: torch.Tensor) -> torch.Tensor:
        """Return epistemic uncertainty σ² for given states."""
        with torch.no_grad():
            _, var = self.network(states)
        return var.mean(dim=-1)  # [B]

    def get_uncertainty_np(self, state: np.ndarray) -> float:
        """Scalar uncertainty for a single state."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.get_uncertainty(state_t).item()

    def estimate_rul(self, state: np.ndarray, delta_override: float = None) -> float:
        """Estimate RUL based on degradation trajectory."""
        delta = delta_override if delta_override is not None else (state[5] if len(state) > 5 else 0.0)
        self.degradation_history.append(delta)

        if len(self.degradation_history) < 10:
            return 1000.0  # Default high RUL when insufficient data

        # Simple linear extrapolation of degradation rate
        deltas = np.array(self.degradation_history)
        rate = np.diff(deltas[-20:]).mean() if len(deltas) >= 20 else np.diff(deltas).mean()

        if rate <= 0:
            return 1000.0

        remaining = max(0.0, (0.9 - delta) / rate)  # Steps until critical
        return float(remaining)

    def encode(self, state: np.ndarray) -> np.ndarray:
        """Encode state for shared embedding."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z = self.encoder(state_t)
        return z.squeeze(0).cpu().numpy()

    def get_embedding(self, state: torch.Tensor) -> torch.Tensor:
        """Get embedding for shared space."""
        return self.encoder(state)

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Approximate log probability under ensemble (for KL distillation)."""
        mean, var = self.network(states)
        dist = torch.distributions.Normal(torch.tanh(mean), (var + 1e-6).sqrt())
        return dist.log_prob(actions).sum(dim=-1, keepdim=True)

    def update(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> float:
        """Update ensemble models with new experience."""
        loss = torch.tensor(0.0, device=self.device)
        for model in self.network.models:
            pred = torch.tanh(model(states))
            loss = loss + F.mse_loss(pred, actions)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.opt.step()
        return loss.item()

    def parameters(self):
        return self.network.parameters()

    def eval(self):
        self.network.eval()

    def train(self):
        self.network.train()

    def zero_grad(self):
        self.network.zero_grad()

    def named_parameters(self):
        return self.network.named_parameters()
