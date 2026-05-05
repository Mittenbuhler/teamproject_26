# Conceptual Pseudocode for `RL_A2C.ipynb`

This is a high-level view of the Advantage Actor-Critic algorithm used in the
notebook. It leaves out rendering and implementation details and focuses on the
learning idea.

## Main Idea

```text
Train one neural network with two outputs:
    an actor that chooses actions
    a critic that estimates how good states are

For each episode:
    let the actor interact with the environment
    store rewards, chosen actions, and critic predictions
    compare what actually happened with what the critic expected
    update both actor and critic from that comparison
```

## Actor-Critic Model

```text
ActorCritic(state):
    convert state into hidden features

    actor output:
        probability of each possible action

    critic output:
        estimated value of the state
```

The actor decides what to do. The critic judges how good the current state is.

## One Training Episode

```text
function run_episode():
    reset the environment

    while the episode is not finished:
        ask the model for:
            action probabilities
            predicted state value

        sample an action from the action probabilities
        play the action in the environment

        store:
            reward
            log probability of the chosen action
            predicted state value
```

The stored trajectory is used after the episode to improve the model.

## Returns

```text
For each time step:
    compute the discounted sum of rewards from that point onward
```

The return is the learning target for the critic: it says how much reward was
actually obtained after visiting a state.

## Advantage

```text
advantage = actual return - critic prediction
```

Conceptually:

```text
If advantage is positive:
    the action worked better than expected

If advantage is negative:
    the action worked worse than expected
```

The advantage is the feedback signal for the actor.

## Updating the Actor

```text
Increase the probability of actions with positive advantage.
Decrease the probability of actions with negative advantage.
```

This makes the policy more likely to repeat actions that performed better than
expected.

## Updating the Critic

```text
Move the critic prediction closer to the actual return.
```

This helps the critic become a better baseline for judging future actions.

## Overall Algorithm

```text
initialize actor-critic network

repeat for many episodes:
    collect one trajectory using the current actor
    compute discounted returns
    compute advantages
    compute actor loss
    compute critic loss
    update the network
```

After training, choose actions from the actor's learned action probabilities.
