"""Minimal probabilistic dynamics model for CartPole.

The model learns one-step transitions:

    state_t, action_t -> state_{t+1}, reward_t, continue_t

It is deliberately tiny for teaching purposes: one shared MLP and three output
heads. There is no recurrent state, latent state, encoder, decoder, actor, or
critic.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class DynamicsModel(nn.Module):
    """A small MLP with separate heads for state, reward, and continuation."""

    def __init__(self, hidden_size: int = 64, min_std: float = 0.05):
        super().__init__()
        self.min_std = min_std

        self.backbone = nn.Sequential(
            nn.Linear(6, hidden_size),  # 4 state values + 2 one-hot actions
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.next_state_head = nn.Linear(hidden_size, 8)  # mean and std for 4 values
        self.reward_head = nn.Linear(hidden_size, 2)  # mean and std for reward
        self.continue_head = nn.Linear(hidden_size, 1)  # Bernoulli logit

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        features = self.backbone(x)

        next_state_mean, next_state_raw_std = self.next_state_head(features).chunk(
            2,
            dim=-1,
        )
        reward_mean, reward_raw_std = self.reward_head(features).chunk(2, dim=-1)

        return {
            "next_state_mean": next_state_mean,
            "next_state_std": F.softplus(next_state_raw_std) + self.min_std,
            "reward_mean": reward_mean,
            "reward_std": F.softplus(reward_raw_std) + self.min_std,
            "continue_logit": self.continue_head(features),
        }

    def loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        reward: torch.Tensor,
        continue_: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        prediction = self(state, action)

        next_state_dist = Normal(
            prediction["next_state_mean"],
            prediction["next_state_std"],
        )
        reward_dist = Normal(prediction["reward_mean"], prediction["reward_std"])

        state_loss = -next_state_dist.log_prob(next_state).sum(dim=-1).mean()
        reward_loss = -reward_dist.log_prob(reward).sum(dim=-1).mean()
        continue_loss = F.binary_cross_entropy_with_logits(
            prediction["continue_logit"],
            continue_,
        )
        loss = state_loss + reward_loss + continue_loss

        with torch.no_grad():
            continue_probability = torch.sigmoid(prediction["continue_logit"])
            metrics = {
                "loss": float(loss.item()),
                "state_mse": float(
                    F.mse_loss(prediction["next_state_mean"], next_state).item()
                ),
                "reward_mse": float(
                    F.mse_loss(prediction["reward_mean"], reward).item()
                ),
                "continue_accuracy": float(
                    ((continue_probability >= 0.5).float() == continue_)
                    .float()
                    .mean()
                    .item()
                ),
            }
        return loss, metrics
