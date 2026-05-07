# Minimal CartPole Dynamics Model

This is a deliberately small pedagogical example of model-based RL dynamics
learning. It uses Gym/Gymnasium `CartPole-v1` and learns one-step predictions:

```text
state_t, action_t -> state_{t+1}, reward_t, continue_t
```

The whole model is a single MLP class:

- input: the 4D CartPole state plus a 2D one-hot action,
- shared hidden layers,
- next-state head: Gaussian mean/std for the next 4D state,
- reward head: Gaussian mean/std for the next reward,
- continue head: Bernoulli logit for whether the episode continues.

There is no RNN, latent state, replay buffer class, actor, critic, encoder, or
decoder. This is meant as the smallest useful foundation before introducing the
extra machinery in Dreamer.

## Run

Install Gymnasium if needed:

```bash
pip install "gymnasium[classic-control]"
```

From the repository root:

```bash
python tutorials/dreamer_dynamics/train_cartpole.py
```

A shorter smoke run:

```bash
python tutorials/dreamer_dynamics/train_cartpole.py \
  --episodes 20 \
  --train-steps 20 \
  --batch-size 32
```

The script prints:

- `state_mse`: next-state prediction error,
- `reward_mse`: reward prediction error,
- `continue_acc`: accuracy for predicting whether the episode continues.

## Ten-step Prediction Example

To see one concrete 10-step rollout, run:

```bash
python tutorials/dreamer_dynamics/example_prediction.py
```

The script trains the small dynamics model, starts from one fixed CartPole
state, applies the same action for 10 steps, and prints:

- each predicted state versus the true state,
- each predicted reward versus the true reward,
- each predicted continue probability versus the true continue flag.

It also saves a plot to:

```text
tutorials/dreamer_dynamics/example_prediction.png
```

You can provide a custom action sequence:

```bash
python tutorials/dreamer_dynamics/example_prediction.py \
  --actions 1,1,0,1,0,0,1,1,0,1
```

## Files

- `model.py`: the single `DynamicsModel` class.
- `train_cartpole.py`: collect random CartPole transitions and train the model.
- `example_prediction.py`: show and plot a 10-step predicted rollout.
