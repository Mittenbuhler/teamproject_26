# Minimal CartPole RSSM Dynamics Model

This is a small recurrent state-space model version of the minimal CartPole
dynamics example.

It learns short sequences:

```text
state_t, action_t -> state_{t+1}, reward_t, continue_t
```

The model keeps only the RSSM pieces:

- deterministic recurrent state `h` from a `GRUCell`,
- stochastic latent state `z`,
- prior `p(z_{t+1} | h_{t+1})`,
- posterior `q(z_{t+1} | h_{t+1}, state_{t+1})`,
- prediction heads for next state, reward, and continue.

There is still no actor, critic, image encoder, image decoder, or full Dreamer
training loop. This folder is meant to sit between the plain MLP dynamics model
and the full Dreamer implementation.

## Run

Install Gymnasium if needed:

```bash
pip install "gymnasium[classic-control]"
```

From the repository root:

```bash
python tutorials/dreamer_rssm_dynamics/train_cartpole.py
```

A short smoke run:

```bash
python tutorials/dreamer_rssm_dynamics/train_cartpole.py \
  --episodes 20 \
  --train-steps 20 \
  --batch-size 16 \
  --sequence-length 8
```

The printed `state_mse`, `reward_mse`, and `continue_acc` are computed from the
prior predictions, so they measure prediction without peeking at the next state.

## Files

- `model.py`: the `RSSMDynamicsModel` class.
- `train_cartpole.py`: collect random CartPole episodes, sample short chunks,
  and train the RSSM.

