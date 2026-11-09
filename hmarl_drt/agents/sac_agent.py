"""
Agent 1: Soft Actor-Critic (SAC) with shared embedding + transfer hooks.

Objective: J(π) = E[Σ γ^t (r_t + α H(π(·|s_t)))]
Architecture: Gaussian Actor + Twin-Q Critics + auto entropy tuning
Transfer hooks: physics regularization input, uncertainty exploration bonus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal


class Actor(nn.Module):
    """Gaussian policy π(a|s, z_shared)."""

    LOG_STD_MIN, LOG_STD_MAX = -20, 2

    def __init__(self, state_dim: int, shared_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(hidden_dim // 2 + shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor, z_shared: torch.Tensor):
        s_enc = self.encoder(state)
        x = torch.cat([s_enc, z_shared], dim=-1)
        h = self.net(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor, z_shared: torch.Tensor):
        mean, log_std = self.forward(state, z_shared)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return y_t, log_prob

    def get_embedding(self, state: torch.Tensor) -> torch.Tensor:
        """Get agent-specific encoding for shared embedding."""
        return self.encoder(state)


class Critic(nn.Module):
    """Twin Q-networks."""

    def __init__(self, state_dim: int, shared_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        inp = state_dim + shared_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(inp, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(inp, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, z_shared, action):
        x = torch.cat([state, z_shared, action], dim=-1)
        return self.q1(x), self.q2(x)


class SACAgent:
    """
    Soft Actor-Critic Agent with shared embedding input and transfer learning hooks.
    """

    def __init__(
        self,
        state_dim: int,
        shared_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy: bool = True,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device
        self.action_dim = action_dim
        self.shared_dim = shared_dim

        self.actor = Actor(state_dim, shared_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, shared_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, shared_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Auto entropy tuning
        self.auto_entropy = auto_entropy
        if auto_entropy:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        # Transfer regularization storage
        self.physics_reg = None
        self.uncertainty_bonus = None

        # CEMRL: EWC Manager (set from training loop)
        self.ewc_manager = None

    def eval(self):
        self.actor.eval()
        self.critic.eval()
        self.critic_target.eval()

    def train(self):
        self.actor.train()
        self.critic.train()
        self.critic_target.train()

    def zero_grad(self):
        self.actor.zero_grad()
        self.critic.zero_grad()
        self.critic_target.zero_grad()

    def parameters(self):
        return list(self.actor.parameters()) + list(self.critic.parameters()) + list(self.critic_target.parameters())

    def named_parameters(self):
        named = []
        for n, p in self.actor.named_parameters():
            named.append((f"actor.{n}", p))
        for n, p in self.critic.named_parameters():
            named.append((f"critic.{n}", p))
        for n, p in self.critic_target.named_parameters():
            named.append((f"critic_target.{n}", p))
        return named

    def act(self, state: np.ndarray, z_shared: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if z_shared.dim() == 1:
            z_shared = z_shared.unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state_t, z_shared)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state_t, z_shared)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch: dict, z_shared: torch.Tensor, z_shared_next: torch.Tensor, z_task: torch.Tensor = None, regime_id: int = None) -> dict:
        s = torch.FloatTensor(batch["states"]).to(self.device)
        a = torch.FloatTensor(batch["actions"]).to(self.device)
        r = torch.FloatTensor(batch["rewards"]).to(self.device)
        s_ = torch.FloatTensor(batch["next_states"]).to(self.device)
        d = torch.FloatTensor(batch["dones"]).to(self.device)

        # -- Critic Update --
        with torch.no_grad():
            a_, log_pi_ = self.actor.sample(s_, z_shared_next)
            q1_t, q2_t = self.critic_target(s_, z_shared_next, a_)
            q_target = torch.min(q1_t, q2_t)
            y = r + self.gamma * (1 - d) * (q_target - self.alpha * log_pi_)

        q1, q2 = self.critic(s, z_shared, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        if self.physics_reg is not None:
            critic_loss = critic_loss + 0.1 * self.physics_reg

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # -- Actor Update --
        a_new, log_pi = self.actor.sample(s, z_shared.detach())
        q1_new, q2_new = self.critic(s, z_shared.detach(), a_new)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_pi - q_new).mean()

        if self.uncertainty_bonus is not None:
            actor_loss = actor_loss - 0.05 * self.uncertainty_bonus.mean()

        # CEMRL: Add EWC penalty to prevent catastrophic forgetting.
        # Uses hard regime-id weighting (ground-truth, supervised regime
        # detection) rather than soft PEARL-centroid weighting, since regime
        # centroids are never populated without a learned PEARL encoder.
        if self.ewc_manager is not None and regime_id is not None:
            ewc_penalty = self.ewc_manager.ewc_penalty("sac_actor", self.actor, regime_id=regime_id)
            actor_loss = actor_loss + ewc_penalty

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # -- Entropy Coefficient Update --
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

        # -- Soft Target Update --
        for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.copy_(self.tau * p.data + (1 - self.tau) * p_t.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
        }

    def set_physics_regularization(self, residual: torch.Tensor):
        self.physics_reg = residual

    def set_uncertainty_bonus(self, uncertainty: torch.Tensor):
        self.uncertainty_bonus = uncertainty

    def trigger_regime_fisher_computation(self, regime_id: int, data_buffer: list, ewc_manager, embedding,
                                          gp_agent=None, fm_agent=None, physics_agent=None):
        """
        Compute Fisher Information Matrix for the actor when a regime transition is detected.
        """
        if len(data_buffer) == 0:
            return

        # Convert buffer to batch format
        states = torch.FloatTensor([t[0] for t in data_buffer]).to(self.device)
        actions = torch.FloatTensor([t[1] for t in data_buffer]).to(self.device)

        # Get shared embeddings for the states using agent encoders
        with torch.no_grad():
            z_shared_list = []
            for s in states:
                z1 = self.actor.get_embedding(s.unsqueeze(0))
                z2 = gp_agent.get_embedding(s.unsqueeze(0)) if gp_agent else torch.zeros(1, 32, device=self.device)
                z3 = fm_agent.get_embedding(s.unsqueeze(0)) if fm_agent else torch.zeros(1, 128, device=self.device)
                z4 = physics_agent.get_embedding(s.unsqueeze(0)) if physics_agent else torch.zeros(1, 64, device=self.device)
                z_shared, _ = embedding([z1, z2, z3, z4])
                z_shared_list.append(z_shared)
            z_shared_batch = torch.cat(z_shared_list, dim=0)

        def actor_loss_fn(model, batch):
            """Compute actor loss for Fisher computation."""
            s_batch = batch["states"]
            z_batch = batch["z_shared"]

            # Sample actions and compute Q-values
            a_new, log_pi = model.sample(s_batch, z_batch)
            q1_new, q2_new = self.critic(s_batch, z_batch, a_new)
            q_new = torch.min(q1_new, q2_new)

            # Actor loss: maximize Q - entropy
            loss = (self.alpha * log_pi - q_new).mean()
            return loss

        # Prepare batch for Fisher computation
        fisher_batch = {
            "states": states,
            "z_shared": z_shared_batch,
        }

        # Compute Fisher Information Matrix
        ewc_manager.compute_fisher(
            component_name="sac_actor",
            model=self.actor,
            data_buffer=[fisher_batch],  # Single batch
            loss_fn=actor_loss_fn,
            regime_id=regime_id,
        )
