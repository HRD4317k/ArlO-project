"""
Agent 2 (replaces GP): iPAE-based Physics-Informed Uncertainty Agent.

Wraps the Interpretable Physics-informed Autoencoder (iPAE) stack —
TemporalEncoder + variational latent + PhysicsConstraintModule + RULHead —
into an agent with the SAME public interface as GPAgent, so it drops
straight into the existing HMARL-DRT / CEMRL training loop, checkpoint
schema, and dashboard/validation tooling (checkpoint key "gp_network",
CSV columns g_gp / r_* etc. are all preserved for compatibility).

Key design decisions (documented honestly):

1. Temporal context: iPAE's encoder is a Conv1D+BiLSTM over a *window* of
   sensor readings. During online rollout (act/estimate_rul/get_uncertainty)
   we maintain a real rolling window of the last `window_size` raw states
   per-episode, so the encoder sees genuine temporal context.

2. Batched training updates: the replay buffer stores single-step
   (s, a, r, s', done) transitions, not sequences, so there is no stored
   history available for an arbitrary sampled transition. For batched
   `update()` calls we therefore build a *static pseudo-window* by tiling
   each state `window_size` times. This lets gradients flow through the
   full encoder/physics/RUL/action stack from state content alone. It is a
   pragmatic compromise — true temporal dynamics are only available in the
   online rolling-window path used during rollout.

3. Supervised RUL signal: instead of the previous ad-hoc linear
   extrapolation, RUL is trained against a genuine supervised target
   derived from the ground-truth degradation index: rul_target =
   RUL_SCALE * (1 - delta), consistent with the target used elsewhere in
   this repo's pretraining scripts (pretrain_gp.py).

4. Uncertainty: MC-Dropout variance from the RUL head (predict_with_uncertainty)
   is used as the epistemic uncertainty signal (replaces GP ensemble variance).

5. Action head: a small MLP on top of the physics-constrained latent z,
   trained to imitate the actions taken (same imitation-style loss GPAgent
   used), i.e. self-distillation from the blended/replayed policy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

from ipae.encoder import iPAEEncoder
from ipae.physics_module import PhysicsConstraintModule
from ipae.rul_head import RULHead


RUL_SCALE = 1000.0  # matches pretrain_gp.py-style supervised RUL proxy


class IPAEActionHead(nn.Module):
    """Small MLP mapping physics-constrained latent z -> action proposal."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class IPAENetwork(nn.Module):
    """Combined encoder + physics constraints + RUL head + action head."""

    def __init__(self, state_dim: int, action_dim: int, latent_dim: int = 8,
                 window_size: int = 20, encoder_hidden: int = 64,
                 lstm_hidden: int = 64, lstm_layers: int = 1,
                 rul_hidden=None, mc_dropout: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.window_size = window_size

        self.encoder = iPAEEncoder(
            n_features=state_dim,
            latent_dim=latent_dim,
            encoder_hidden=encoder_hidden,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
        )
        self.physics = PhysicsConstraintModule(latent_dim=latent_dim)
        self.rul_head = RULHead(latent_dim=latent_dim, hidden_dims=rul_hidden or [64, 32, 16], dropout=mc_dropout)
        self.action_head = IPAEActionHead(latent_dim, action_dim, dropout=mc_dropout)

        # Projection of latent z -> 32-dim embedding (matches GP's shared-embedding slot)
        self.embed_proj = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
        )

    def encode(self, window: torch.Tensor):
        """window: [B, W, F] -> z_constrained [B, latent_dim], mu, logvar"""
        z, mu, logvar = self.encoder(window)
        z_c = self.physics.apply_constraints(z)
        return z_c, mu, logvar

    def forward(self, window: torch.Tensor):
        z_c, mu, logvar = self.encode(window)
        action = torch.tanh(self.action_head(z_c))
        rul = self.rul_head(z_c)
        return action, rul, z_c, mu, logvar


class IPAEAgent:
    """
    Physics-informed uncertainty/RUL agent (drop-in replacement for GPAgent).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 8,
        window_size: int = 20,
        encoder_hidden: int = 64,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        rul_hidden=None,
        mc_dropout: float = 0.1,
        lambda_kl: float = 0.01,
        lambda_rul: float = 1.0,
        lambda_mono: float = 0.5,
        lambda_physics: float = 1.0,
        n_mc_samples: int = 10,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.window_size = window_size
        self.device = device
        self.lambda_kl = lambda_kl
        self.lambda_rul = lambda_rul
        self.lambda_mono = lambda_mono
        self.lambda_physics = lambda_physics
        self.n_mc_samples = n_mc_samples

        self.network = IPAENetwork(
            state_dim=state_dim, action_dim=action_dim, latent_dim=latent_dim,
            window_size=window_size, encoder_hidden=encoder_hidden,
            lstm_hidden=lstm_hidden, lstm_layers=lstm_layers,
            rul_hidden=rul_hidden, mc_dropout=mc_dropout,
        ).to(device)
        self.opt = torch.optim.Adam(self.network.parameters(), lr=lr)

        # Only these layers behave differently in train vs eval; toggling them
        # directly avoids recursively walking the full module tree each step.
        self._mode_layers = [
            m for m in self.network.modules()
            if isinstance(m, (nn.BatchNorm1d, nn.Dropout, nn.LSTM))
        ]

        # Online rolling window of raw states (per-episode temporal context)
        self.window_buffer = deque(maxlen=window_size)

        # Single-entry cache of the most recent online encode. The training
        # loop calls act(), get_uncertainty_np(), estimate_rul() and
        # get_embedding() on the SAME state each timestep; without this the
        # Conv1D+BiLSTM encoder would run four times per environment step.
        self._cache_key = None
        self._cache_z = None

        # RUL history (kept for API parity / diagnostics only)
        self.degradation_history = deque(maxlen=500)

    # ------------------------------------------------------------------
    # Windowing helpers
    # ------------------------------------------------------------------
    def reset_window(self):
        """Call at episode start to clear the rolling temporal context."""
        self.window_buffer.clear()
        self._cache_key = None
        self._cache_z = None

    def _online_latent(self, state: np.ndarray) -> torch.Tensor:
        """
        Encode `state` in the context of the rolling window, reusing the cached
        latent when the same state is queried repeatedly within one timestep.
        """
        arr = np.asarray(state, dtype=np.float32)
        key = arr.tobytes()
        if key == self._cache_key and self._cache_z is not None:
            return self._cache_z
        window = self._online_window_tensor(arr)
        z_c, _, _ = self._encode_eval(window)
        self._cache_key = key
        self._cache_z = z_c
        return z_c

    def _online_window_tensor(self, state: np.ndarray) -> torch.Tensor:
        """Update rolling buffer with `state` and return [1, W, F] tensor."""
        self.window_buffer.append(np.asarray(state, dtype=np.float32))
        window = list(self.window_buffer)
        if len(window) < self.window_size:
            # Pad by repeating the earliest observed state (no history yet)
            pad = [window[0]] * (self.window_size - len(window))
            window = pad + window
        arr = np.stack(window, axis=0)  # [W, F]
        return torch.FloatTensor(arr).unsqueeze(0).to(self.device)  # [1, W, F]

    @staticmethod
    def _pseudo_window_tensor(states: torch.Tensor, window_size: int) -> torch.Tensor:
        """Tile each state W times -> [B, W, F] static pseudo-window for batched training."""
        return states.unsqueeze(1).expand(-1, window_size, -1).contiguous()

    @staticmethod
    def _normalize_uncertainty(rul_var: torch.Tensor) -> torch.Tensor:
        """
        Convert raw MC-Dropout RUL variance into a dimensionless relative
        uncertainty in [0, 1].

        This matters for drop-in compatibility: the uncertainty signal is
        consumed by the reward shaper's curiosity term, the arbiter's
        degradation/uncertainty bias, and the GP->NN transfer bonus, all of
        which were tuned against the old ensemble-action variance (~1e-2..1).
        Raw RUL variance lives on the RUL scale (~1e4+) and would otherwise
        dominate the total reward by orders of magnitude.
        """
        return (rul_var.clamp(min=0.0).sqrt() / RUL_SCALE).clamp(0.0, 1.0)

    def _encode_eval(self, window: torch.Tensor):
        """
        Encode in eval mode under no_grad.

        Eval mode matters here: the temporal encoder contains BatchNorm1d, and
        online rollout calls arrive one state at a time (B=1). Running BN in
        train mode on those would both normalize using degenerate single-window
        statistics and pollute the running estimates used later during batched
        training updates.
        """
        was_training = self.network.training
        for m in self._mode_layers:
            m.training = False
        try:
            with torch.no_grad():
                z_c, mu, logvar = self.network.encode(window)
        finally:
            if was_training:
                for m in self._mode_layers:
                    m.training = True
        return z_c, mu, logvar

    # ------------------------------------------------------------------
    # Public interface (mirrors GPAgent)
    # ------------------------------------------------------------------
    def act(self, state: np.ndarray) -> np.ndarray:
        z_c = self._online_latent(state)
        with torch.no_grad():
            action = torch.tanh(self.network.action_head(z_c))
        return action.squeeze(0).cpu().numpy()

    def _window_for(self, states: torch.Tensor, window: torch.Tensor = None) -> torch.Tensor:
        """Use the real replayed history when supplied, else a tiled window."""
        return window if window is not None else self._pseudo_window_tensor(states, self.window_size)

    def get_uncertainty(self, states: torch.Tensor, window: torch.Tensor = None) -> torch.Tensor:
        """Batched epistemic uncertainty via MC-Dropout on the RUL head."""
        z_c, _, _ = self._encode_eval(self._window_for(states, window))
        with torch.no_grad():
            _, rul_var = self.network.rul_head.predict_with_uncertainty(z_c, n_samples=self.n_mc_samples)
        return self._normalize_uncertainty(rul_var)

    def get_uncertainty_np(self, state: np.ndarray) -> float:
        z_c = self._online_latent(state)
        with torch.no_grad():
            _, rul_var = self.network.rul_head.predict_with_uncertainty(z_c, n_samples=self.n_mc_samples)
        return float(self._normalize_uncertainty(rul_var).item())

    def estimate_rul(self, state: np.ndarray, delta_override: float = None) -> float:
        """Supervised-informed RUL estimate from the trained RUL head."""
        delta = delta_override if delta_override is not None else (state[5] if len(state) > 5 else 0.0)
        self.degradation_history.append(delta)
        z_c = self._online_latent(state)
        with torch.no_grad():
            rul = self.network.rul_head(z_c)
        return float(rul.item())

    def encode(self, state: np.ndarray) -> np.ndarray:
        z_c = self._online_latent(state)
        with torch.no_grad():
            z = self.network.embed_proj(z_c)
        return z.squeeze(0).cpu().numpy()

    def get_embedding(self, state: torch.Tensor, window: torch.Tensor = None) -> torch.Tensor:
        """
        Embedding for the shared embedding space (32-dim, matches GP slot).

        For the online B=1 path this goes through the rolling-window encoder,
        so the embedding shares the same temporal context (and the same cached
        latent) as act()/estimate_rul() within a timestep.
        """
        if window is None and state.dim() == 2 and state.shape[0] == 1:
            z_c = self._online_latent(state.detach().squeeze(0).cpu().numpy())
        else:
            z_c, _, _ = self._encode_eval(self._window_for(state, window))
        with torch.no_grad():
            return self.network.embed_proj(z_c)

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor, window: torch.Tensor = None) -> torch.Tensor:
        """Approximate log-prob under a unit-variance Gaussian around the action head's output."""
        z_c, _, _ = self._encode_eval(self._window_for(states, window))
        mean = torch.tanh(self.network.action_head(z_c))
        dist = torch.distributions.Normal(mean, 1.0)
        return dist.log_prob(actions).sum(dim=-1, keepdim=True)

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        window: torch.Tensor = None,
        next_window: torch.Tensor = None,
    ) -> float:
        """
        Supervised update:
          - action_loss: imitate replayed actions (self-distillation, same as GPAgent)
          - rul_loss: MSE against true-delta-derived RUL target (genuine supervision)
          - kl_loss: light regularization on the latent
          - mono/ode losses: only computable when a real next-step window is
            available, since both compare latents at t and t+1
        `rewards` unused directly (kept for signature parity with GPAgent.update).
        """
        z_c, mu, logvar = self.network.encode(self._window_for(states, window))

        pred_action = torch.tanh(self.network.action_head(z_c))
        action_loss = F.mse_loss(pred_action, actions)

        delta = states[:, 5] if states.shape[-1] > 5 else torch.zeros(states.shape[0], device=states.device)
        rul_target = RUL_SCALE * (1.0 - delta.clamp(0.0, 1.0))
        rul_pred = self.network.rul_head(z_c)
        rul_loss = F.mse_loss(rul_pred, rul_target)

        kl_loss = self.network.physics.kl_divergence(mu, logvar)

        loss = action_loss + self.lambda_rul * rul_loss + self.lambda_kl * kl_loss

        if next_window is not None:
            z_next, _, _ = self.network.encode(next_window)
            loss = loss + self.lambda_mono * self.network.physics.monotonicity_loss(z_c, z_next)
            loss = loss + self.lambda_physics * self.network.physics.ode_residual_loss(z_c, z_next)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.opt.step()
        return loss.item()

    # ------------------------------------------------------------------
    # Framework glue (parity with GPAgent)
    # ------------------------------------------------------------------
    def parameters(self):
        return self.network.parameters()

    def named_parameters(self):
        return self.network.named_parameters()

    def eval(self):
        self.network.eval()

    def train(self):
        self.network.train()

    def zero_grad(self):
        self.network.zero_grad()
