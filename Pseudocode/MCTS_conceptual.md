# MCTS Pseudocode, Version 2

This version describes the algorithm from the outside in: first the repeated
search loop, then the pieces used inside one call to `explore`.

## Main Search Loop

```text
root = Node(current_game_state)

for n iterations:
    explore(root)

action = choose_action_from(root)
```

The root node represents the current real game state. Each call to `explore`
adds more statistical information to the search tree below that root.

After enough iterations, the algorithm chooses an action from the root, usually
the action leading to the most visited child node.

## Node Statistics

Each node stores information collected by previous simulations:

```text
Node:
    game_state
    children
    parent
    action_from_parent
    visit_count
    total_reward
    terminal?
```

The important statistics are:

```text
visit_count:
    how often this node was reached during search

total_reward:
    sum of rollout rewards that passed through this node

average_reward:
    total_reward / visit_count
```

## explore(root)

One call to `explore` is one complete MCTS simulation.

```text
function explore(root):
    node = root

    # 1. Selection
    while node has children:
        node = child of node with highest UCB score

    # 2. Expansion or direct simulation
    if node has never been visited:
        reward = rollout(node)
    else if node is not terminal:
        expand(node)
        node = one newly created child
        reward = rollout(node)
    else:
        reward = 0

    # 3. Backpropagation
    while node exists:
        node.visit_count += 1
        node.total_reward += reward
        node = node.parent
```

Conceptually, `explore` does three things:

```text
Selection:
    move down the existing tree toward a promising node

Expansion / Simulation:
    create new possible futures if needed,
    then estimate one future by playing randomly

Backpropagation:
    send the simulation result back up the path to the root
```

## Selection with UCB

During selection, the algorithm repeatedly chooses the child with the highest
UCB score.

```text
UCB(child) =
    average_reward(child)
    +
    exploration_constant * sqrt(log(parent.visit_count) / child.visit_count)
```

The two terms have different roles:

```text
average_reward(child):
    favors actions that have produced good rewards so far

exploration bonus:
    favors actions that have been tried less often
```

If a child has never been visited, its UCB score is treated as infinitely good.
This makes sure every action gets tried at least once.

## Expansion

```text
function expand(node):
    for each possible action:
        copy the game state stored in node
        apply the action to the copied state
        create a child node for the resulting state
```

Expansion adds all immediate next states to the tree. It does not decide that
one action is best; it only makes those actions available for future selection.

## Rollout

```text
function rollout(node):
    state = copy of node.game_state
    reward_sum = 0

    while state is not terminal:
        action = random legal action
        state, reward = apply action
        reward_sum += reward

    return reward_sum
```

A rollout is a rough estimate of the value of a node. One rollout may be very
noisy, but many rollouts make the node's average reward more meaningful.

## Choosing the Action

```text
function choose_action_from(root):
    choose the child with the highest visit_count
    return the action that leads to that child
```

The most visited child is chosen because MCTS has spent the most search effort
there. In the notebook, that child becomes the new root so future searches can
reuse the subtree already built below it.
