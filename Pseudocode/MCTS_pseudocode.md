# Conceptual Pseudocode for `MCTS.ipynb`

This is a high-level view of the Monte Carlo Tree Search algorithm used in the
notebook. It leaves out implementation details and focuses on the algorithmic
idea.

## Main Idea

```text
Start from the current game state.

Build a search tree where:
    each node represents a game state
    each edge represents an action
    each node stores:
        how often it was visited
        how much reward was collected through it

Repeat many simulations from the current root.

Use the simulation results to choose the most promising next action.
```

## Search Node

```text
Node:
    state
    parent
    children
    action_that_led_here
    visit_count
    total_reward
    terminal_state?
```

Each node is one possible future state of the environment.

## One MCTS Simulation

```text
function explore(root):
    node = root

    # 1. Selection
    while node is already expanded:
        choose the child with the best UCB score
        node = chosen child

    # 2. Expansion
    if node is not terminal and has been visited before:
        create one child for each possible action
        choose one of the new children

    # 3. Rollout
    simulate random actions from this node until the game ends
    collect the rollout reward

    # 4. Backpropagation
    walk back from the simulated node to the root
    for each node on this path:
        increase visit_count
        add the rollout reward to total_reward
```

## UCB Score

```text
UCB(node) =
    average reward of the node
    +
    exploration bonus for nodes visited less often
```

Conceptually:

```text
Prefer actions that look good,
but still try actions that have not been explored much.
```

## Rollout

```text
function rollout(node):
    copy the environment state stored in the node

    while the copied game is not finished:
        choose a random action
        apply it
        accumulate reward

    return accumulated reward
```

The rollout gives a noisy estimate of how good a state might be.

## Choosing the Real Action

```text
function choose_action(root):
    repeat many times:
        explore(root)

    choose the child of the root with the highest visit count
    return that child's action
```

The notebook keeps the chosen child as the new root, so the next decision can
reuse the search statistics already collected below that state.

## Overall Algorithm

```text
root = node for current game state

repeat for each move:
    run many MCTS simulations from root
    action = most visited action from root
    play action
    root = child reached by action
```
