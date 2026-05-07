# Conceptual Pseudocode for `tutorials/dreamer_dynamics`

This file describes the high-level learning idea in the minimal CartPole
dynamics tutorial. It focuses on learning a one-step world model, not on the
full Dreamer agent with latent recurrent states, imagination rollouts, actor, or
critic.

## Main Idea

```text
Learn a model of the environment:
    current state + action -> next state, reward, continue flag

Use random CartPole episodes to create supervised training data.
Train a small neural network to predict what happens after each action.
```

The model does not learn which action is best. It only learns to predict the
environment's next step.

## Data Collection

```text
function collect_transitions(environment, number_of_episodes):
    empty dataset

    for each episode:
        state = reset environment

        while episode is not finished:
            action = sample random action
            next_state, reward, done = environment.step(action)

            store transition:
                state
                one-hot action
                next_state
                reward
                continue = 0 if done else 1

            state = next_state

    return dataset of transitions
```

The tutorial uses random actions because the goal is only to learn the dynamics
from observed transitions.

## Dynamics Model

```text
function dynamics_model(state, action):
    input = concatenate state and one-hot action
    hidden_features = shared neural network(input)

    next_state_prediction:
        mean and standard deviation for each next-state value

    reward_prediction:
        mean and standard deviation for the reward

    continue_prediction:
        probability that the episode continues

    return all predictions
```

The next state and reward are modeled as Gaussian distributions. The continue
flag is modeled as a binary prediction.

## Training Loss

```text
function model_loss(batch):
    predictions = dynamics_model(batch.state, batch.action)

    state_loss =
        negative log probability of the true next state
        under the predicted next-state distribution

    reward_loss =
        negative log probability of the true reward
        under the predicted reward distribution

    continue_loss =
        binary classification loss for the true continue flag

    total_loss = state_loss + reward_loss + continue_loss

    return total_loss
```

The model is rewarded for assigning high probability to the transition that
actually happened.

## Main Training Loop

```text
set random seeds
create CartPole environment

dataset = collect_transitions(environment, episodes)
initialize dynamics model
initialize optimizer

for each training step:
    mini_batch = sample random transitions from dataset

    loss = model_loss(mini_batch)

    clear old gradients
    backpropagate loss
    update model parameters

    occasionally print:
        total loss
        next-state mean squared error
        reward mean squared error
        continue prediction accuracy
```

Training is supervised: each stored transition already contains the target next
state, reward, and continue flag.

## Evaluation

```text
function evaluate(model, dataset):
    run model_loss on all collected transitions without updating the model

    report:
        state_mse
        reward_mse
        continue_accuracy
```

Lower state and reward errors mean the model better predicts CartPole's
one-step behavior. Higher continue accuracy means it better predicts when an
episode has ended.

## Where This Fits in Dreamer

```text
Full Dreamer:
    learns a world model
    imagines future trajectories inside that model
    trains an actor and critic from imagined futures

This tutorial:
    only learns the simplest one-step world model
```

This makes the dynamics-learning piece visible before adding the more complex
parts of Dreamer.
