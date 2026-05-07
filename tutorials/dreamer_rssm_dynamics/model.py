"""A small RSSM dynamics model for CartPole.

This is the recurrent-state version of the minimal MLP dynamics example. The
model learns short sequences:

    state_t, action_t -> state_{t+1}, reward_t, continue_t

The RSSM has a deterministic GRU state `h` and a stochastic latent state `z`.
During training, a posterior uses the observed next state. For prediction, the
model uses the prior, which does not see the next state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


class RSSMDynamicsModel(nn.Module):
    """Tiny recurrent state-space model with state, reward, and continue heads."""

    def __init__(
        self,
        stochastic_size: int = 8,
        deterministic_size: int = 32,
        hidden_size: int = 64,
        min_std: float = 0.05,
        kl_scale: float = 0.1,
        prior_loss_scale: float = 0.5,
    ):
        super().__init__()
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.min_std = min_std
        self.kl_scale = kl_scale
        self.prior_loss_scale = prior_loss_scale

        self.action_model = nn.Sequential(
            nn.Linear(stochastic_size + 2, hidden_size),
            nn.ReLU(),
        )
        self.rnn = nn.GRUCell(hidden_size, deterministic_size)

        self.prior_model = nn.Sequential(
            nn.Linear(deterministic_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * stochastic_size),
        )
        self.posterior_model = nn.Sequential(
            nn.Linear(deterministic_size + 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * stochastic_size),
        )

        self.prediction_model = nn.Sequential(
            nn.Linear(stochastic_size + deterministic_size, hidden_size),
            nn.ReLU(),
        )
        self.next_state_head = nn.Linear(hidden_size, 8)
        self.reward_head = nn.Linear(hidden_size, 2)
        self.continue_head = nn.Linear(hidden_size, 1)

    def make_normal(self, parameters: torch.Tensor) -> Normal:
        mean, raw_std = parameters.chunk(2, dim=-1)
        std = F.softplus(raw_std) + self.min_std
        return Normal(mean, std)

    def prediction_heads(
        self,
        stochastic_state: torch.Tensor,
        deterministic_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = self.prediction_model(
            torch.cat([stochastic_state, deterministic_state], dim=-1)
        )
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

    def initial(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        stochastic = torch.zeros(batch_size, self.stochastic_size, device=device)
        deterministic = torch.zeros(batch_size, self.deterministic_size, device=device)
        return stochastic, deterministic

    def observe(self, states: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the RSSM over a sequence.

        Args:
            states: Tensor with shape `[batch, sequence + 1, 4]`.
            actions: Tensor with shape `[batch, sequence, 2]`.
        """

        batch_size, sequence_plus_one, _ = states.shape
        sequence_length = sequence_plus_one - 1
        stochastic, deterministic = self.initial(batch_size, states.device)

        priors = []
        posteriors = []
        posterior_predictions = []
        prior_predictions = []

        for t in range(sequence_length):
            rnn_input = self.action_model(torch.cat([stochastic, actions[:, t]], dim=-1))
            deterministic = self.rnn(rnn_input, deterministic)

            prior = self.make_normal(self.prior_model(deterministic))
            posterior = self.make_normal(
                self.posterior_model(torch.cat([deterministic, states[:, t + 1]], dim=-1))
            )
            prior_stochastic = prior.rsample()
            stochastic = posterior.rsample()

            priors.append(prior)
            posteriors.append(posterior)
            posterior_predictions.append(
                self.prediction_heads(stochastic, deterministic)
            )
            prior_predictions.append(
                self.prediction_heads(prior_stochastic, deterministic)
            )

        return {
            "priors": priors,
            "posteriors": posteriors,
            "posterior_predictions": stack_predictions(posterior_predictions),
            "prior_predictions": stack_predictions(prior_predictions),
        }

    def loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        continues: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        output = self.observe(states, actions)
        targets = states[:, 1:]
        posterior_loss, posterior_metrics = prediction_loss(
            output["posterior_predictions"],
            targets,
            rewards,
            continues,
        )
        prior_loss, prior_metrics = prediction_loss(
            output["prior_predictions"],
            targets,
            rewards,
            continues,
        )

        kl_values = [
            kl_divergence(posterior, prior).sum(dim=-1).mean()
            for posterior, prior in zip(output["posteriors"], output["priors"])
        ]
        kl_loss = torch.stack(kl_values).mean()
        loss = posterior_loss + self.kl_scale * kl_loss + self.prior_loss_scale * prior_loss

        metrics = {
            "loss": float(loss.item()),
            "kl": float(kl_loss.item()),
            "posterior_state_mse": posterior_metrics["state_mse"],
            "posterior_reward_mse": posterior_metrics["reward_mse"],
            "posterior_continue_accuracy": posterior_metrics["continue_accuracy"],
            "state_mse": prior_metrics["state_mse"],
            "reward_mse": prior_metrics["reward_mse"],
            "continue_accuracy": prior_metrics["continue_accuracy"],
        }
        return loss, metrics


def stack_predictions(predictions: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.stack([prediction[key] for prediction in predictions], dim=1)
        for key in predictions[0]
    }


def prediction_loss(
    prediction: dict[str, torch.Tensor],
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    continues: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    next_state_dist = Normal(prediction["next_state_mean"], prediction["next_state_std"])
    reward_dist = Normal(prediction["reward_mean"], prediction["reward_std"])

    state_loss = -next_state_dist.log_prob(next_states).sum(dim=-1).mean()
    reward_loss = -reward_dist.log_prob(rewards).sum(dim=-1).mean()
    continue_loss = F.binary_cross_entropy_with_logits(
        prediction["continue_logit"],
        continues,
    )
    loss = state_loss + reward_loss + continue_loss

    with torch.no_grad():
        continue_probability = torch.sigmoid(prediction["continue_logit"])
        metrics = {
            "state_mse": float(
                F.mse_loss(prediction["next_state_mean"], next_states).item()
            ),
            "reward_mse": float(
                F.mse_loss(prediction["reward_mean"], rewards).item()
            ),
            "continue_accuracy": float(
                ((continue_probability >= 0.5).float() == continues)
                .float()
                .mean()
                .item()
            ),
        }
    return loss, metrics

