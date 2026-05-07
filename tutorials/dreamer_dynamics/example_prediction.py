"""Show a short CartPole rollout predicted by the dynamics model.

Run from the repository root:

    python tutorials/dreamer_dynamics/example_prediction.py

The script trains the tiny one-step dynamics model, then repeatedly feeds the
predicted next state back into the model to compare a multi-step prediction with
the true CartPole rollout.
"""

from __future__ import annotations

import argparse
import importlib
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dreamer_dynamics_matplotlib"),
)

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from tutorials.dreamer_dynamics.model import DynamicsModel


STATE_NAMES = ("x", "x_dot", "theta", "theta_dot")

GRAVITY = 9.8
MASSCART = 1.0
MASSPOLE = 0.1
TOTAL_MASS = MASSCART + MASSPOLE
LENGTH = 0.5
POLEMASS_LENGTH = MASSPOLE * LENGTH
FORCE_MAG = 10.0
TAU = 0.02
X_THRESHOLD = 2.4
THETA_THRESHOLD_RADIANS = 12 * 2 * np.pi / 360


class ActionSpace:
    """Small replacement for Gym's discrete action space."""

    def __init__(self, seed: int):
        self.seed(seed)

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def sample(self) -> int:
        return int(self.rng.integers(2))


class LocalCartPoleEnv:
    """Tiny CartPole-v1 compatible fallback used when Gym is unavailable."""

    def __init__(self, seed: int, max_episode_steps: int = 500):
        self.action_space = ActionSpace(seed)
        self.max_episode_steps = max_episode_steps
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(4, dtype=np.float32)
        self.steps = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.state = self.rng.uniform(low=-0.05, high=0.05, size=4).astype(np.float32)
        return self.state.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self.state, reward, terminated = cartpole_step(self.state, action)
        self.steps += 1
        done = terminated or self.steps >= self.max_episode_steps
        return self.state.copy(), reward, done, {}

    def close(self) -> None:
        pass


def cartpole_step(state: np.ndarray, action: int) -> tuple[np.ndarray, float, bool]:
    """One CartPole physics step matching the classic-control equations."""

    x, x_dot, theta, theta_dot = state
    force = FORCE_MAG if action == 1 else -FORCE_MAG
    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    temp = (force + POLEMASS_LENGTH * theta_dot**2 * sintheta) / TOTAL_MASS
    theta_acc = (GRAVITY * sintheta - costheta * temp) / (
        LENGTH * (4.0 / 3.0 - MASSPOLE * costheta**2 / TOTAL_MASS)
    )
    x_acc = temp - POLEMASS_LENGTH * theta_acc * costheta / TOTAL_MASS

    next_state = np.array(
        [
            x + TAU * x_dot,
            x_dot + TAU * x_acc,
            theta + TAU * theta_dot,
            theta_dot + TAU * theta_acc,
        ],
        dtype=np.float32,
    )
    next_x, _, next_theta, _ = next_state
    done = bool(
        next_x < -X_THRESHOLD
        or next_x > X_THRESHOLD
        or next_theta < -THETA_THRESHOLD_RADIANS
        or next_theta > THETA_THRESHOLD_RADIANS
    )
    return next_state, 1.0, done


def make_env(seed: int) -> tuple[object, str]:
    for module_name in ("gymnasium", "gym"):
        try:
            gym = importlib.import_module(module_name)
            env = gym.make("CartPole-v1")
            env.action_space.seed(seed)
            return env, f"{module_name} CartPole-v1"
        except ImportError:
            pass

    return LocalCartPoleEnv(seed), "local CartPole physics fallback"


def reset(env: object, seed: int | None = None) -> np.ndarray:
    result = env.reset(seed=seed) if seed is not None else env.reset()
    observation = result[0] if isinstance(result, tuple) else result
    return np.asarray(observation, dtype=np.float32)


def step(env: object, action: int) -> tuple[np.ndarray, float, bool]:
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


def set_state(env: object, state: np.ndarray) -> None:
    target = getattr(env, "unwrapped", env)
    target.state = np.asarray(state, dtype=np.float32).copy()
    if hasattr(target, "steps"):
        target.steps = 0


def collect_transitions(
    env: object,
    episodes: int,
    seed: int,
) -> dict[str, torch.Tensor]:
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


def sample_batch(
    data: dict[str, torch.Tensor],
    batch_size: int,
) -> dict[str, torch.Tensor]:
    indices = torch.randint(len(data["state"]), (batch_size,))
    return {key: value[indices] for key, value in data.items()}


def train_model(args: argparse.Namespace) -> tuple[DynamicsModel, object, str]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env, source = make_env(args.seed)
    data = collect_transitions(env, args.episodes, args.seed)

    model = DynamicsModel(hidden_size=args.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for _ in range(args.train_steps):
        mini_batch = sample_batch(data, args.batch_size)
        loss, _ = model.loss(
            mini_batch["state"],
            mini_batch["action"],
            mini_batch["next_state"],
            mini_batch["reward"],
            mini_batch["continue"],
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, env, source


def predict_transition(
    model: DynamicsModel,
    state: np.ndarray,
    action: int,
) -> dict[str, np.ndarray | float]:
    model.eval()
    with torch.no_grad():
        prediction = model(
            torch.tensor(state, dtype=torch.float32).unsqueeze(0),
            torch.tensor(one_hot(action), dtype=torch.float32).unsqueeze(0),
        )
        continue_probability = torch.sigmoid(prediction["continue_logit"])

    return {
        "next_state_mean": prediction["next_state_mean"].squeeze(0).numpy(),
        "next_state_std": prediction["next_state_std"].squeeze(0).numpy(),
        "reward_mean": float(prediction["reward_mean"].item()),
        "reward_std": float(prediction["reward_std"].item()),
        "continue_probability": float(continue_probability.item()),
    }


def rollout_prediction(
    model: DynamicsModel,
    start_state: np.ndarray,
    actions: list[int],
) -> dict[str, list[np.ndarray] | list[float]]:
    states = [start_state.copy()]
    rewards = []
    reward_stds = []
    continue_probabilities = []
    state = start_state.copy()

    for action in actions:
        prediction = predict_transition(model, state, action)
        state = np.asarray(prediction["next_state_mean"], dtype=np.float32)

        states.append(state)
        rewards.append(float(prediction["reward_mean"]))
        reward_stds.append(float(prediction["reward_std"]))
        continue_probabilities.append(float(prediction["continue_probability"]))

    return {
        "states": states,
        "rewards": rewards,
        "reward_stds": reward_stds,
        "continue_probabilities": continue_probabilities,
    }


def rollout_truth(
    env: object,
    start_state: np.ndarray,
    actions: list[int],
) -> dict[str, list[np.ndarray] | list[float]]:
    set_state(env, start_state)

    states = [start_state.copy()]
    rewards = []
    continues = []
    done = False

    for action in actions:
        if done:
            states.append(states[-1].copy())
            rewards.append(0.0)
            continues.append(0.0)
            continue

        next_state, reward, done = step(env, action)
        states.append(next_state)
        rewards.append(reward)
        continues.append(0.0 if done else 1.0)

    return {
        "states": states,
        "rewards": rewards,
        "continues": continues,
    }


def draw_cartpole(
    ax,
    state: np.ndarray,
    title: str,
    color: str,
    subtitle: str,
    x_limit: float,
) -> None:
    import matplotlib.patches as patches

    x, _, theta, _ = state
    cart_y = 0.0
    cart_width = 0.42
    cart_height = 0.22
    pole_length = 1.0
    pole_x = x + pole_length * np.sin(theta)
    pole_y = cart_y + cart_height / 2 + pole_length * np.cos(theta)

    ax.plot([-x_limit, x_limit], [-0.15, -0.15], color="#555555", linewidth=1.5)
    ax.add_patch(
        patches.Rectangle(
            (x - cart_width / 2, cart_y - cart_height / 2),
            cart_width,
            cart_height,
            facecolor="#e8edf3",
            edgecolor="#1f2937",
            linewidth=1.5,
        )
    )
    ax.add_patch(
        patches.Circle(
            (x - cart_width * 0.28, cart_y - cart_height / 2),
            0.055,
            facecolor="#1f2937",
        )
    )
    ax.add_patch(
        patches.Circle(
            (x + cart_width * 0.28, cart_y - cart_height / 2),
            0.055,
            facecolor="#1f2937",
        )
    )
    ax.plot([x, pole_x], [cart_y + cart_height / 2, pole_y], color=color, linewidth=5)
    ax.scatter([x], [cart_y + cart_height / 2], s=40, color="#1f2937", zorder=3)

    ax.set_title(title, fontsize=11)
    ax.text(
        0.0,
        -0.32,
        subtitle,
        ha="center",
        va="top",
        fontsize=6,
        family="monospace",
        linespacing=1.1,
    )
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-1.1, 1.3)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def save_plot(
    path: Path,
    actions: list[int],
    predicted_rollout: dict[str, list[np.ndarray] | list[float]],
    true_rollout: dict[str, list[np.ndarray] | list[float]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    predicted_states = predicted_rollout["states"]
    predicted_rewards = predicted_rollout["rewards"]
    predicted_continues = predicted_rollout["continue_probabilities"]
    true_states = true_rollout["states"]
    true_rewards = true_rollout["rewards"]
    true_continues = true_rollout["continues"]

    all_states = list(predicted_states) + list(true_states)
    max_abs_x = max(abs(float(state[0])) for state in all_states)
    x_limit = min(X_THRESHOLD + 0.2, max(0.9, max_abs_x + 0.65))

    columns = len(actions) + 1
    fig_width = max(16.0, columns * 1.6)
    fig, axes = plt.subplots(
        2,
        columns,
        figsize=(fig_width, 6.6),
        constrained_layout=True,
        squeeze=False,
    )

    for column in range(columns):
        if column == 0:
            draw_cartpole(
                axes[0, column],
                predicted_states[column],
                "Predicted\nstart",
                "#2563eb",
                format_state_for_plot(predicted_states[column]),
                x_limit,
            )
            draw_cartpole(
                axes[1, column],
                true_states[column],
                "True\nstart",
                "#2563eb",
                format_state_for_plot(true_states[column]),
                x_limit,
            )
            continue

        action = actions[column - 1]
        draw_cartpole(
            axes[0, column],
            predicted_states[column],
            f"Pred t={column}\na={action}",
            "#d97706",
            format_state_for_plot(predicted_states[column])
            + f"\nr={predicted_rewards[column - 1]:+.2f} p={predicted_continues[column - 1]:.2f}",
            x_limit,
        )
        draw_cartpole(
            axes[1, column],
            true_states[column],
            f"True t={column}\na={action}",
            "#059669",
            format_state_for_plot(true_states[column])
            + f"\nr={true_rewards[column - 1]:+.1f} c={int(true_continues[column - 1])}",
            x_limit,
        )

    fig.suptitle("10-step CartPole dynamics rollout", fontsize=14)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def print_comparison(
    source: str,
    start_state: np.ndarray,
    actions: list[int],
    predicted_rollout: dict[str, list[np.ndarray] | list[float]],
    true_rollout: dict[str, list[np.ndarray] | list[float]],
    plot_path: Path | None,
) -> None:
    predicted_states = predicted_rollout["states"]
    predicted_rewards = predicted_rollout["rewards"]
    predicted_continues = predicted_rollout["continue_probabilities"]
    true_states = true_rollout["states"]
    true_rewards = true_rollout["rewards"]
    true_continues = true_rollout["continues"]

    print(f"Environment source: {source}")
    print(f"Start state order: {', '.join(STATE_NAMES)}")
    print(f"Start state: {format_array(start_state)}")
    print(f"Actions: {actions}")
    print()

    for index, action in enumerate(actions, start=1):
        predicted_state = np.asarray(predicted_states[index])
        true_state = np.asarray(true_states[index])
        error = predicted_state - true_state
        rmse = float(np.sqrt(np.mean(error**2)))

        print(f"Step {index} after action {action}:")
        print(f"  predicted state: {format_array(predicted_state)}")
        print(f"  true state:      {format_array(true_state)}")
        print(f"  error:           {format_array(error)}")
        print(
            "  reward: "
            f"predicted={predicted_rewards[index - 1]:.5f}, "
            f"true={true_rewards[index - 1]:.1f}"
        )
        print(
            "  continue: "
            f"predicted p={predicted_continues[index - 1]:.5f}, "
            f"true={int(true_continues[index - 1])}, "
            f"state_rmse={rmse:.5f}"
        )

    if plot_path is not None:
        print(f"Saved plot: {plot_path}")


def format_array(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.5f}" for value in values) + "]"


def format_state_for_plot(values: np.ndarray) -> str:
    short_names = ("x", "xd", "th", "thd")
    return "\n".join(
        f"{name}={value:+.3f}" for name, value in zip(short_names, values)
    )


def build_action_sequence(args: argparse.Namespace) -> list[int]:
    if args.actions is None:
        return [args.action for _ in range(args.horizon)]

    actions = []
    for raw_action in args.actions.split(","):
        raw_action = raw_action.strip()
        if raw_action == "":
            continue
        if raw_action not in {"0", "1"}:
            raise ValueError("actions must contain only 0 and 1")
        actions.append(int(raw_action))

    if len(actions) != args.horizon:
        raise ValueError(
            f"--actions contains {len(actions)} actions, but --horizon is {args.horizon}"
        )
    return actions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--action",
        type=int,
        choices=(0, 1),
        default=1,
        help="Action to repeat when --actions is omitted: 0 pushes left, 1 pushes right.",
    )
    parser.add_argument(
        "--actions",
        type=str,
        default=None,
        help="Comma-separated action sequence, such as 1,1,0,1,0,0,1,1,0,1.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Number of future steps to predict.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("tutorials/dreamer_dynamics/example_prediction.png"),
        help="Where to save the CartPole comparison plot.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print the numeric comparison.",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    model, env, source = train_model(args)
    actions = build_action_sequence(args)

    start_state = np.array([0.020, 0.150, 0.080, -0.120], dtype=np.float32)
    reset(env, seed=args.seed + 10_000)
    true_rollout = rollout_truth(env, start_state, actions)
    predicted_rollout = rollout_prediction(model, start_state, actions)

    plot_path = None if args.no_plot else args.plot_path
    if plot_path is not None:
        save_plot(
            plot_path,
            actions,
            predicted_rollout,
            true_rollout,
        )

    print_comparison(
        source,
        start_state,
        actions,
        predicted_rollout,
        true_rollout,
        plot_path,
    )
    env.close()


if __name__ == "__main__":
    main(build_parser().parse_args())
