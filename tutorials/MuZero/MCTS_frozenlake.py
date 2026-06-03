"""Solve Gymnasium FrozenLake with Monte Carlo Tree Search.

This follows the MCTS structure from ``MCTS.ipynb``: selection with UCB,
expansion, random rollout, backpropagation, and choosing the most visited child.

FrozenLake is a discrete environment, so the search can use the transition
table exposed by ``env.unwrapped.P`` instead of cloning an environment object.
The reward schedule below uses Gymnasium's configurable FrozenLake rewards:

    reward_schedule=(goal_reward, hole_reward, step_reward)
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Optional

try:
    import gymnasium as gym
except ModuleNotFoundError:  # pragma: no cover - lets the file import cleanly
    gym = None


ACTION_NAMES = ("LEFT", "DOWN", "RIGHT", "UP")
ACTION_ARROWS = ("<", "v", ">", "^")


def make_frozenlake_env(
    *,
    map_name: str = "4x4",
    is_slippery: bool = False,
    goal_reward: float = 1.0,
    hole_reward: float = -1.0,
    step_reward: float = -0.01,
    seed: Optional[int] = None,
):
    """Create the Gymnasium FrozenLake variant with configurable rewards."""

    if gym is None:
        raise RuntimeError(
            'Gymnasium is required. Install it with: pip install "gymnasium[toy-text]"'
        )

    try:
        env = gym.make(
            "FrozenLake-v1",
            map_name=map_name,
            is_slippery=is_slippery,
            reward_schedule=(goal_reward, hole_reward, step_reward),
        )
    except TypeError as exc:
        raise RuntimeError(
            "This FrozenLake-v1 version does not accept reward_schedule. "
            "Use the Gymnasium version from the FrozenLake A2C notebook."
        ) from exc

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env


class FrozenLakeModel:
    """Small generative model around Gymnasium FrozenLake's transition table."""

    def __init__(self, env, *, gamma: float = 1.0, seed: Optional[int] = None):
        self.env = env
        self.gamma = gamma
        self.rng = random.Random(seed)
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        self.transitions = env.unwrapped.P
        self.max_steps = env.spec.max_episode_steps or 100
        self.desc = self._decode_desc(env.unwrapped.desc)

    @staticmethod
    def _decode_desc(desc):
        return [
            [
                tile.decode("utf-8") if isinstance(tile, bytes) else str(tile)
                for tile in row
            ]
            for row in desc
        ]

    def sample_transition(self, state: int, action: int):
        """Sample ``(next_state, reward, done)`` from the model."""

        outcomes = self.transitions[int(state)][int(action)]
        if len(outcomes) == 1:
            _, next_state, reward, done = outcomes[0]
            return int(next_state), float(reward), bool(done)

        threshold = self.rng.random()
        cumulative_probability = 0.0
        for probability, next_state, reward, done in outcomes:
            cumulative_probability += probability
            if threshold <= cumulative_probability:
                return int(next_state), float(reward), bool(done)

        _, next_state, reward, done = outcomes[-1]
        return int(next_state), float(reward), bool(done)

    def random_action(self) -> int:
        return self.rng.randrange(self.n_actions)

    def tile_at(self, state: int) -> str:
        width = len(self.desc[0])
        row, column = divmod(int(state), width)
        return self.desc[row][column]

    def is_terminal_state(self, state: int) -> bool:
        return self.tile_at(state) in {"H", "G"}


@dataclass
class Node:
    """MCTS node for a FrozenLake state."""

    model: FrozenLakeModel
    state: int
    done: bool
    parent: Optional["Node"] = None
    action_index: Optional[int] = None
    reward_from_parent: float = 0.0
    depth: int = 0
    child: dict[int, "Node"] = field(default_factory=dict)
    T: float = 0.0
    N: int = 0

    @property
    def average_value(self) -> float:
        return self.T / self.N if self.N else 0.0

    def action_value_from_parent(self) -> float:
        return self.reward_from_parent + self.model.gamma * self.average_value

    def getUCBscore(self, exploration_constant: float) -> float:
        """UCB score used during the selection phase."""

        if self.N == 0:
            return float("inf")

        parent_visits = max(1, self.parent.N if self.parent else 1)
        exploration_bonus = exploration_constant * math.sqrt(
            math.log(parent_visits + 1) / self.N
        )
        return self.action_value_from_parent() + exploration_bonus

    def detach_parent(self) -> None:
        self.parent = None

    def create_child(self) -> None:
        """Expand this node by creating one child for each possible action."""

        if self.done or self.child:
            return

        for action in range(self.model.n_actions):
            next_state, reward, done = self.model.sample_transition(self.state, action)
            next_depth = self.depth + 1
            truncated = next_depth >= self.model.max_steps
            self.child[action] = Node(
                model=self.model,
                state=next_state,
                done=done or truncated,
                parent=self,
                action_index=action,
                reward_from_parent=reward,
                depth=next_depth,
            )

    def _best_ucb_child(self, exploration_constant: float) -> "Node":
        max_ucb = max(
            child.getUCBscore(exploration_constant) for child in self.child.values()
        )
        candidates = [
            child
            for child in self.child.values()
            if child.getUCBscore(exploration_constant) == max_ucb
        ]
        return self.model.rng.choice(candidates)

    def explore(self, exploration_constant: float = math.sqrt(2.0)) -> None:
        """Run one complete MCTS simulation from this node."""

        current = self

        while current.child:
            current = current._best_ucb_child(exploration_constant)

        if current.N > 0 and not current.done:
            current.create_child()
            if current.child:
                current = self.model.rng.choice(list(current.child.values()))

        rollout_value = current.rollout()
        current.backpropagate(rollout_value)

    def rollout(self) -> float:
        """Play randomly from this node and return the discounted future reward."""

        if self.done:
            return 0.0

        state = self.state
        depth = self.depth
        discount = 1.0
        total_reward = 0.0

        while depth < self.model.max_steps:
            action = self.model.random_action()
            next_state, reward, done = self.model.sample_transition(state, action)
            total_reward += discount * reward

            depth += 1
            if done or depth >= self.model.max_steps:
                break

            state = next_state
            discount *= self.model.gamma

        return total_reward

    def backpropagate(self, value: float) -> None:
        """Backpropagate a state value up to the root."""

        node: Optional[Node] = self
        while node is not None:
            node.N += 1
            node.T += value

            if node.parent is None:
                break

            value = node.reward_from_parent + node.model.gamma * value
            node = node.parent

    def next(self) -> tuple["Node", int]:
        """Choose the next real action after enough search."""

        if self.done:
            raise ValueError("game has ended")

        if not self.child:
            raise ValueError("no children found and game has not ended")

        best_child = max(
            self.child.values(),
            key=lambda child: (child.N, child.action_value_from_parent()),
        )
        return best_child, int(best_child.action_index)


MCTS_POLICY_EXPLORE = 1000


def Policy_Player_MCTS(
    mytree: Node,
    *,
    simulations: int = MCTS_POLICY_EXPLORE,
    exploration_constant: float = math.sqrt(2.0),
) -> tuple[Node, int]:
    """Notebook-style MCTS policy: search, pick best child, reuse subtree."""

    for _ in range(simulations):
        mytree.explore(exploration_constant)

    next_tree, next_action = mytree.next()
    next_tree.detach_parent()
    return next_tree, next_action


@dataclass
class EpisodeResult:
    total_reward: float
    actions: list[int]
    states: list[int]
    solved: bool


def run_episode(
    *,
    seed: int,
    simulations: int,
    exploration_constant: float,
    map_name: str,
    is_slippery: bool,
    goal_reward: float,
    hole_reward: float,
    step_reward: float,
    gamma: float,
) -> EpisodeResult:
    env = make_frozenlake_env(
        map_name=map_name,
        is_slippery=is_slippery,
        goal_reward=goal_reward,
        hole_reward=hole_reward,
        step_reward=step_reward,
        seed=seed,
    )
    model = FrozenLakeModel(env, gamma=gamma, seed=seed)
    reset_result = env.reset(seed=seed)
    state = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    done = False
    total_reward = 0.0
    states = [int(state)]
    actions = []
    tree = Node(model=model, state=int(state), done=False)

    for _ in range(model.max_steps):
        tree, action = Policy_Player_MCTS(
            tree,
            simulations=simulations,
            exploration_constant=exploration_constant,
        )
        step_result = env.step(action)
        next_state, reward, terminated, truncated, _ = step_result
        done = terminated or truncated

        actions.append(action)
        states.append(int(next_state))
        total_reward += float(reward)

        if tree.state != int(next_state) or tree.done != done:
            tree = Node(
                model=model,
                state=int(next_state),
                done=done,
                depth=len(actions),
            )

        if done:
            break

    solved = bool(done and model.tile_at(states[-1]) == "G")
    env.close()
    return EpisodeResult(total_reward, actions, states, solved)


def build_policy_grid(
    model: FrozenLakeModel,
    *,
    simulations: int,
    exploration_constant: float,
) -> str:
    rows = []
    state = 0

    for row in model.desc:
        cells = []
        for tile in row:
            if tile in {"H", "G"}:
                cells.append(tile)
            else:
                root = Node(model=model, state=state, done=False)
                for _ in range(simulations):
                    root.explore(exploration_constant)
                _, action = root.next()
                cells.append(ACTION_ARROWS[action])
            state += 1
        rows.append(" ".join(cells))

    return "\n".join(rows)


def action_path(actions: list[int]) -> str:
    if not actions:
        return "(no actions)"
    return " -> ".join(ACTION_NAMES[action] for action in actions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=MCTS_POLICY_EXPLORE)
    parser.add_argument("--policy-simulations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--map-name", default="4x4")
    parser.add_argument("--is-slippery", action="store_true")
    parser.add_argument("--goal-reward", type=float, default=1.0)
    parser.add_argument("--hole-reward", type=float, default=-1.0)
    parser.add_argument("--step-reward", type=float, default=-0.01)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--exploration-constant", type=float, default=math.sqrt(2.0))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = [
        run_episode(
            seed=args.seed + episode,
            simulations=args.simulations,
            exploration_constant=args.exploration_constant,
            map_name=args.map_name,
            is_slippery=args.is_slippery,
            goal_reward=args.goal_reward,
            hole_reward=args.hole_reward,
            step_reward=args.step_reward,
            gamma=args.gamma,
        )
        for episode in range(args.episodes)
    ]

    mean_reward = sum(result.total_reward for result in results) / len(results)
    solved_count = sum(result.solved for result in results)
    first = results[0]

    print(
        "FrozenLake MCTS "
        f"reward_schedule=({args.goal_reward}, {args.hole_reward}, {args.step_reward})"
    )
    print(f"episodes: {args.episodes}")
    print(f"simulations per move: {args.simulations}")
    print(f"success rate: {solved_count}/{args.episodes}")
    print(f"mean reward: {mean_reward:.3f}")
    print(f"first episode reward: {first.total_reward:.3f}")
    print(f"first episode states: {first.states}")
    print(f"first episode actions: {action_path(first.actions)}")

    env = make_frozenlake_env(
        map_name=args.map_name,
        is_slippery=args.is_slippery,
        goal_reward=args.goal_reward,
        hole_reward=args.hole_reward,
        step_reward=args.step_reward,
        seed=args.seed,
    )
    model = FrozenLakeModel(env, gamma=args.gamma, seed=args.seed)
    print("policy grid:")
    print(
        build_policy_grid(
            model,
            simulations=args.policy_simulations,
            exploration_constant=args.exploration_constant,
        )
    )
    env.close()


if __name__ == "__main__":
    main()
