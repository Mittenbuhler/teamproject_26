# A2C Pseudocode

This file describes the core Advantage Actor-Critic algorithm used in
`RL_A2C.ipynb`. It focuses on the learning algorithm, not on rendering,
visualization, or notebook helper functions.

## Main Training Loop

```text
initialize actor-critic network
initialize optimizer

for each episode:
    trajectory = collect one episode using the current policy

    returns = compute discounted future rewards from trajectory.rewards
    advantages = returns - trajectory.predicted_values

    actor_loss = policy loss using action log probabilities and advantages
    critic_loss = value prediction loss using returns

    update network using actor_loss + critic_loss
```

A2C trains two parts of one model at the same time:

```text
Actor:
    learns which actions to choose

Critic:
    learns how valuable each state is
```

## Actor-Critic Network

```text
function network(state):
    hidden = transform state with a neural network layer

    action_probabilities = actor_head(hidden)
    state_value = critic_head(hidden)

    return action_probabilities, state_value
```

The actor output is a probability distribution over actions. The critic output
is a single number estimating the expected future reward from the current state.

## Collecting a Trajectory

```text
function collect_episode(environment, policy):
    reset environment

    while episode is not done:
        action_probabilities, predicted_value = network(current_state)

        action = sample from action_probabilities
        log_probability = log probability of sampled action

        next_state, reward, done = environment.step(action)

        store:
            reward
            log_probability
            predicted_value

        current_state = next_state

    return collected rewards, log probabilities, and predicted values
```

The agent samples actions during training so it can keep exploring while it
learns.

## Discounted Returns

```text
function compute_returns(rewards, gamma):
    future_return = 0
    returns = empty list

    for each reward, moving backward through the episode:
        future_return = reward + gamma * future_return
        insert future_return at the front of returns

    return returns
```

Each return estimates how much reward was obtained from that time step onward.
The discount factor `gamma` controls how much future rewards matter.

## Advantage

```text
advantage = return - predicted_value
```

Conceptually:

```text
positive advantage:
    the action led to better results than the critic expected

negative advantage:
    the action led to worse results than the critic expected
```

The advantage tells the actor whether the sampled action should become more or
less likely in similar states.

## Actor Loss

```text
actor_loss = -mean(log_probability_of_action * advantage)
```

If the advantage is positive, the loss encourages the actor to increase the
probability of that action. If the advantage is negative, the loss encourages
the actor to decrease that probability.

The advantage is treated as a fixed signal for the actor update, so the actor
does not directly change the critic through this term.

## Critic Loss

```text
critic_loss = mean squared error(predicted_value, return)
```

The critic is trained to make its value estimate closer to the observed
discounted return.

## Network Update

```text
total_loss = actor_loss + critic_loss

clear old gradients
backpropagate total_loss
take one optimizer step
```

Both actor and critic share the same network body, so one update improves the
policy and the value estimates together.

## Choosing Actions After Training

```text
function choose_action(state):
    action_probabilities, predicted_value = network(state)
    choose the action with the highest probability
```

During evaluation, the notebook uses the most likely action instead of sampling,
so the trained policy behaves deterministically.
