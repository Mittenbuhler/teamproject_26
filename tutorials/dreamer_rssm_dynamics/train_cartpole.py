"""Train a tiny RSSM dynamics model on CartPole.

Run from the repository root:

    python tutorials/dreamer_rssm_dynamics/train_cartpole.py
"""

from __future__ import annotations

import argparse
import importlib
import random
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from tutorials.dreamer_rssm_dynamics.model import RSSMDynamicsModel


def make_env(seed: int):
    for module_name in ("gymnasium", "gym"):
        try:
            gym = importlib.import_module(module_name)
            env = gym.make("CartPole-v1")
            env.action_space.seed(seed)
            return env
        except ImportError:
            pass
    raise ImportError(
        'Install Gymnasium first: pip install "gymnasium[classic-control]"'
    )


def reset(env, seed: int | None = None) -> np.ndarray:
    result = env.reset(seed=seed) if seed is not None else env.reset()
    observation = result[0] if isinstance(result, tuple) else result
    return np.asarray(observation, dtype=np.float32)


def step(env, action: int) -> tuple[np.ndarray, float, bool]:
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, _ = result
        done = terminated or truncated
    else:
        observation, reward, done, _ = result
    return np.asarray(observation, dtype=np.float32), float(reward), bool(done)


def one_hot(action: int) -> np.ndarray:
    action_vector = np.zeros(2, dtype=np.float32)
    action_vector[action] = 1.0
    return action_vector


def collect_episodes(env, episodes: int, seed: int) -> list[dict[str, np.ndarray]]:
    collected = []
    for episode in range(episodes):
        states = [reset(env, seed=seed + episode)]
        actions, rewards, continues = [], [], []
        done = False

        while not done:
            action = env.action_space.sample()
            next_state, reward, done = step(env, action)
            actions.append(one_hot(action))
            rewards.append([reward])
            continues.append([0.0 if done else 1.0])
            states.append(next_state)

        collected.append(
            {
                "states": np.asarray(states, dtype=np.float32),
                "actions": np.asarray(actions, dtype=np.float32),
                "rewards": np.asarray(rewards, dtype=np.float32),
                "continues": np.asarray(continues, dtype=np.float32),
            }
        )
    return collected


def sample_batch(
    episodes: list[dict[str, np.ndarray]],
    batch_size: int,
    sequence_length: int,
) -> dict[str, torch.Tensor]:
    long_enough = [
        episode for episode in episodes if len(episode["actions"]) >= sequence_length
    ]
    if not long_enough:
        raise ValueError("No collected episode is long enough for this sequence length.")

    states, actions, rewards, continues = [], [], [], []
    for _ in range(batch_size):
        episode = random.choice(long_enough)
        max_start = len(episode["actions"]) - sequence_length
        start = random.randint(0, max_start)
        end = start + sequence_length
        states.append(episode["states"][start : end + 1])
        actions.append(episode["actions"][start:end])
        rewards.append(episode["rewards"][start:end])
        continues.append(episode["continues"][start:end])

    return {
        "states": torch.tensor(np.asarray(states), dtype=torch.float32),
        "actions": torch.tensor(np.asarray(actions), dtype=torch.float32),
        "rewards": torch.tensor(np.asarray(rewards), dtype=torch.float32),
        "continues": torch.tensor(np.asarray(continues), dtype=torch.float32),
    }


def evaluate(
    model: RSSMDynamicsModel,
    episodes: list[dict[str, np.ndarray]],
    batch_size: int,
    sequence_length: int,
) -> dict[str, float]:
    model.eval()
    batch = sample_batch(episodes, batch_size, sequence_length)
    with torch.no_grad():
        _, metrics = model.loss(
            batch["states"],
            batch["actions"],
            batch["rewards"],
            batch["continues"],
        )
    model.train()
    return metrics


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env(args.seed)
    episodes = collect_episodes(env, args.episodes, args.seed)
    env.close()

    model = RSSMDynamicsModel(
        stochastic_size=args.stochastic_size,
        deterministic_size=args.deterministic_size,
        hidden_size=args.hidden_size,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    transition_count = sum(len(episode["actions"]) for episode in episodes)
    print(
        f"Collected {len(episodes)} episodes and {transition_count} transitions "
        "from CartPole-v1."
    )

    for train_step in range(1, args.train_steps + 1):
        batch = sample_batch(episodes, args.batch_size, args.sequence_length)
        loss, metrics = model.loss(
            batch["states"],
            batch["actions"],
            batch["rewards"],
            batch["continues"],
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if train_step == 1 or train_step % args.log_every == 0:
            print(
                f"step={train_step:04d} "
                f"loss={metrics['loss']:.3f} "
                f"kl={metrics['kl']:.3f} "
                f"state_mse={metrics['state_mse']:.5f} "
                f"reward_mse={metrics['reward_mse']:.5f} "
                f"continue_acc={metrics['continue_accuracy']:.3f}"
            )

    metrics = evaluate(model, episodes, args.batch_size, args.sequence_length)
    print(
        "final "
        f"state_mse={metrics['state_mse']:.5f} "
        f"reward_mse={metrics['reward_mse']:.5f} "
        f"continue_acc={metrics['continue_accuracy']:.3f} "
        f"kl={metrics['kl']:.3f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--stochastic-size", type=int, default=8)
    parser.add_argument("--deterministic-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())

