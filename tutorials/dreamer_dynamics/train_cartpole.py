"""Train a tiny CartPole dynamics model.

Run from the repository root:

    python tutorials/dreamer_dynamics/train_cartpole.py
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

from tutorials.dreamer_dynamics.model import DynamicsModel


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


def collect_transitions(env, episodes: int, seed: int) -> dict[str, torch.Tensor]:
    states, actions, next_states, rewards, continues = [], [], [], [], []

    for episode in range(episodes):
        state = reset(env, seed=seed + episode)
        done = False

        while not done:
            action = env.action_space.sample()
            next_state, reward, done = step(env, action)

            states.append(state)
            actions.append(one_hot(action))
            next_states.append(next_state)
            rewards.append([reward])
            continues.append([0.0 if done else 1.0])

            state = next_state

    return {
        "state": torch.tensor(np.asarray(states), dtype=torch.float32),
        "action": torch.tensor(np.asarray(actions), dtype=torch.float32),
        "next_state": torch.tensor(np.asarray(next_states), dtype=torch.float32),
        "reward": torch.tensor(np.asarray(rewards), dtype=torch.float32),
        "continue": torch.tensor(np.asarray(continues), dtype=torch.float32),
    }


def batch(data: dict[str, torch.Tensor], batch_size: int) -> dict[str, torch.Tensor]:
    indices = torch.randint(len(data["state"]), (batch_size,))
    return {key: value[indices] for key, value in data.items()}


def evaluate(model: DynamicsModel, data: dict[str, torch.Tensor]) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        _, metrics = model.loss(
            data["state"],
            data["action"],
            data["next_state"],
            data["reward"],
            data["continue"],
        )
    model.train()
    return metrics


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env(args.seed)
    data = collect_transitions(env, args.episodes, args.seed)
    env.close()

    model = DynamicsModel(hidden_size=args.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Collected {len(data['state'])} transitions from CartPole-v1.")

    for train_step in range(1, args.train_steps + 1):
        mini_batch = batch(data, args.batch_size)
        loss, metrics = model.loss(
            mini_batch["state"],
            mini_batch["action"],
            mini_batch["next_state"],
            mini_batch["reward"],
            mini_batch["continue"],
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if train_step == 1 or train_step % args.log_every == 0:
            print(
                f"step={train_step:04d} "
                f"loss={metrics['loss']:.3f} "
                f"state_mse={metrics['state_mse']:.5f} "
                f"reward_mse={metrics['reward_mse']:.5f} "
                f"continue_acc={metrics['continue_accuracy']:.3f}"
            )

    metrics = evaluate(model, data)
    print(
        "final "
        f"state_mse={metrics['state_mse']:.5f} "
        f"reward_mse={metrics['reward_mse']:.5f} "
        f"continue_acc={metrics['continue_accuracy']:.3f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
