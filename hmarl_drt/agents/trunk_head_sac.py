"""
F2 – Trunk-Head SAC with Online EWC.

Architecture (from report §4.2):
    trunk: shared, EWC-protected, MAML outer-loop only
    heads: regime-specific, MAML inner-loop adapted

    π(a|s, z_shared, regime) = head_regime(trunk([s, z_shared]))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from typing import Optional


class TrunkHeadActor(nn.Module):
    """Gaussian policy with shared trunk + regime-specific heads."""

    LOG_STD_MIN, LOG_STD_MAX = -20, 2

    def __init__(self, state_dim: int, shared_dim: int, action_dim: int,
                 hidden_dim: int = 256, n_regimes: int = 4):
        super().__init__()
        # Shared trunk (EWC-protected)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + shared_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        # Regime-specific heads (MAML-adapted)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim * 2),
            ) for _ in range(n_regimes)
        ])

    def forward(self, state: torch.Tensor, z_shared: torch.Tensor,
                regime_id: Optional[int] = None,
                soft_weights: Optional[torch.Tensor] = None):
        x = torch.cat([state, z_shared], dim=-1)
        trunk_out = self.trunk(x)
        if soft_weights is not None:
            # Soft-weighted blend across all heads
            output = sum(soft_weights[:, j:j+1] * self.heads[j](trunk_out)
                        for j in range(len(self.heads)))
        else:
            output = self.heads[regime_id](trunk_out)
        mean, log_std = output.chunk(2, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor, z_shared: torch.Tensor,
               regime_id: Optional[int] = None,
               soft_weights: Optional[torch.Tensor] = None):
        mean, log_std = self.forward(state, z_shared, regime_id, soft_weights)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return y_t, log_prob

    def get_embedding(self, state: torch.Tensor) -> torch.Tensor:
        """First layer of trunk for shared embedding space."""
        return state  # simplified; real impl uses encoder


class TrunkHeadCritic(nn.Module):
    """Twin Q-networks with shared trunk + regime-specific heads."""

    def __init__(self, state_dim: int, shared_dim: int, action_dim: int,
                 hidden_dim: int = 256, n_regimes: int = 4):
        super().__init__()
        inp_dim = state_dim + shared_dim + action_dim
        self.trunk = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.q1_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_regimes)])
        self.q2_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_regimes)])

    def forward(self, state, z_shared, action, regime_id=0, soft_weights=None):
        x = torch.cat([state, z_shared, action], dim=-1)
        trunk_out = self.trunk(x)
        if soft_weights is not None:
            q1 = sum(soft_weights[:, j:j+1] * self.q1_heads[j](trunk_out) for j in range(len(self.q1_heads)))
            q2 = sum(soft_weights[:, j:j+1] * self.q2_heads[j](trunk_out) for j in range(len(self.q2_heads)))
        else:
            q1 = self.q1_heads[regime_id](trunk_out)
            q2 = self.q2_heads[regime_id](trunk_out)
        return q1, q2


class TrunkHeadSAC:
    """
    SAC with trunk-head decomposition for EWC + MAML compatibility.
    """

    def __init__(self, state_dim: int, shared_dim: int, action_dim: int,
                 hidden_dim: int = 256, n_regimes: int = 4,
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
                 alpha: float = 0.2, auto_entropy: bool = True,
                 device: str = "cpu"):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device
        self.action_dim = action_dim
        self.n_regimes = n_regimes

        self.actor = TrunkHeadActor(state_dim, shared_dim, action_dim, hidden_dim, n_regimes).to(device)
        self.critic = TrunkHeadCritic(state_dim, shared_dim, action_dim, hidden_dim, n_regimes).to(device)
        self.critic_target = TrunkHeadCritic(state_dim, shared_dim, action_dim, hidden_dim, n_regimes).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        if auto_entropy:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        self.ewc_manager = None
        self.current_regime_id = 0

    def act(self, state: np.ndarray, z_shared: torch.Tensor,
            deterministic: bool = False, regime_id: Optional[int] = None) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if z_shared.dim() == 1:
            z_shared = z_shared.unsqueeze(0)
        rid = regime_id if regime_id is not None else self.current_regime_id
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state_t, z_shared, regime_id=rid)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state_t, z_shared, regime_id=rid)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch: dict, z_shared: torch.Tensor, z_shared_next: torch.Tensor,
               z_task: torch.Tensor = None, regime_id: Optional[int] = None) -> dict:
        s = torch.FloatTensor(batch["states"]).to(self.device)
        a = torch.FloatTensor(batch["actions"]).to(self.device)
        r = torch.FloatTensor(batch["rewards"]).to(self.device)
        s_ = torch.FloatTensor(batch["next_states"]).to(self.device)
        d = torch.FloatTensor(batch["dones"]).to(self.device)
        rid = regime_id if regime_id is not None else self.current_regime_id

        with torch.no_grad():
            a_, log_pi_ = self.actor.sample(s_, z_shared_next, regime_id=rid)
            q1_t, q2_t = self.critic_target(s_, z_shared_next, a_, regime_id=rid)
            q_target = torch.min(q1_t, q2_t)
            y = r + self.gamma * (1 - d) * (q_target - self.alpha * log_pi_)

        q1, q2 = self.critic(s, z_shared, a, regime_id=rid)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        a_new, log_pi = self.actor.sample(s, z_shared.detach(), regime_id=rid)
        q1_new, q2_new = self.critic(s, z_shared.detach(), a_new, regime_id=rid)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_pi - q_new).mean()

        if self.ewc_manager is not None and z_task is not None:
            ewc_penalty = self.ewc_manager.ewc_penalty("sac_actor", self.actor, z_task)
            actor_loss = actor_loss + ewc_penalty

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        if hasattr(self, 'alpha_opt'):
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

        for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.copy_(self.tau * p.data + (1 - self.tau) * p_t.data)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(), "alpha": self.alpha}
