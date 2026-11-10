"""
Soft Actor-Critic in plain PyTorch (Haarnoja et al., 2018).

Tanh-squashed Gaussian actor, twin Q critics with a min in the target, and
automatic entropy tuning. Written as a small self-contained class (instead of
Stable-Baselines3) because FOMAML needs to clone the whole agent, run inner
SAC updates on the clone, and read the gradients back out.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def mlp(i, o, h):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, h):
        super().__init__()
        self.net = mlp(s_dim, 2 * a_dim, h)
        self.a_dim = a_dim

    def forward(self, s, deterministic=False):
        mu, log_std = self.net(s).split(self.a_dim, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        if deterministic:
            return torch.tanh(mu), None
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)  # tanh correction
        return a, logp


class TwinQ(nn.Module):
    def __init__(self, s_dim, a_dim, h):
        super().__init__()
        self.q1 = mlp(s_dim + a_dim, 1, h)
        self.q2 = mlp(s_dim + a_dim, 1, h)

    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class ReplayBuffer:
    def __init__(self, s_dim, a_dim, size=200_000):
        self.s = np.zeros((size, s_dim), np.float32)
        self.a = np.zeros((size, a_dim), np.float32)
        self.r = np.zeros(size, np.float32)
        self.s2 = np.zeros((size, s_dim), np.float32)
        self.d = np.zeros(size, np.float32)
        self.size, self.ptr, self.n = size, 0, 0

    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.ptr = (i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, batch, rng: np.random.Generator):
        idx = rng.integers(0, self.n, size=batch)
        f = torch.from_numpy
        return f(self.s[idx]), f(self.a[idx]), f(self.r[idx]), f(self.s2[idx]), f(self.d[idx])


class SAC:
    def __init__(self, s_dim=8, a_dim=2, hidden=128, lr=3e-4, gamma=0.99, tau=0.005,
                 init_alpha=0.2, batch=128, seed=0):
        torch.manual_seed(seed)
        self.actor = Actor(s_dim, a_dim, hidden)
        self.critic = TwinQ(s_dim, a_dim, hidden)
        self.critic_t = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32, requires_grad=True)
        self.target_entropy = -float(a_dim)
        self.gamma, self.tau, self.lr, self.batch = gamma, tau, lr, batch
        self.s_dim, self.a_dim = s_dim, a_dim
        self.make_optimisers()

    # ------------------------------------------------------------------
    def make_optimisers(self, lr=None):
        lr = lr or self.lr
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.opt_al = torch.optim.Adam([self.log_alpha], lr=lr)

    def meta_params(self):
        """Parameters that MAML meta-learns (target critic is tracked by Polyak)."""
        return list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_alpha]

    def clone(self) -> "SAC":
        c = copy.deepcopy(self)
        c.make_optimisers()
        return c

    @torch.no_grad()
    def act(self, s, deterministic=False):
        a, _ = self.actor(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0), deterministic)
        return a[0].numpy()

    # ------------------------------------------------------------------
    def losses(self, batch):
        s, a, r, s2, d = batch
        alpha = self.log_alpha.exp()
        with torch.no_grad():
            a2, logp2 = self.actor(s2)
            q1t, q2t = self.critic_t(s2, a2)
            y = r + self.gamma * (1 - d) * (torch.min(q1t, q2t) - alpha.detach() * logp2)
        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        a_pi, logp = self.actor(s)
        q1p, q2p = self.critic(s, a_pi)
        actor_loss = (alpha.detach() * logp - torch.min(q1p, q2p)).mean()
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        return critic_loss, actor_loss, alpha_loss

    def grads(self, batch):
        """Gradient of the SAC loss wrt meta_params (used as the FOMAML query gradient).
        The actor loss only flows into actor params (critic grads come from the critic loss)."""
        critic_loss, actor_loss, alpha_loss = self.losses(batch)
        ca = list(self.critic.parameters())
        g_c = torch.autograd.grad(critic_loss, ca)
        g_a = torch.autograd.grad(actor_loss, list(self.actor.parameters()))
        g_al = torch.autograd.grad(alpha_loss, [self.log_alpha])
        return list(g_a) + list(g_c) + list(g_al)

    def update(self, batch):
        critic_loss, _, _ = self.losses(batch)
        self.opt_c.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.opt_c.step()
        # actor + temperature with freshly computed Q (critic just changed)
        s = batch[0]
        alpha = self.log_alpha.exp().detach()
        a_pi, logp = self.actor(s)
        q1p, q2p = self.critic(s, a_pi)
        actor_loss = (alpha * logp - torch.min(q1p, q2p)).mean()
        self.opt_a.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.opt_a.step()
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_al.zero_grad()
        alpha_loss.backward()
        self.opt_al.step()
        self.soft_update()
        return critic_loss.item(), actor_loss.item(), float(alpha)

    @torch.no_grad()
    def soft_update(self):
        for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
            pt.mul_(1 - self.tau).add_(self.tau * p)

    # ------------------------------------------------------------------
    def state_dict(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "critic_t": self.critic_t.state_dict(), "log_alpha": self.log_alpha.detach().clone()}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_t.load_state_dict(sd["critic_t"])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"])
        self.make_optimisers()
