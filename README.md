# TVC Research Controller

This repository hosts an advanced 2D thrust-vector-control (TVC) research stack
that combines:

- a MuJoCo-based planar rocket environment tuned for high-fidelity TVC studies,
- a differentiable JAX dynamics model used by a gradient-based MPC solver,
- a PPO actor-critic implemented with Flax and Optax,
- evolutionary perturbations with curriculum learning to generalise across
  deployment scenarios.

The controller outputs two lateral offsets $(x, y)$ positioning the engine
carriage beneath the rocket, mirroring real-world TVC hardware. The MPC module
provides a physics-consistent feed-forward solution, while PPO learns residual
corrections in tandem with evolutionary elitism.

## Setup

1. **Install MuJoCo and GPU-enabled JAX.** For CUDA 12 for example:

   ```powershell
   pip install mujoco==3.1.4
   pip install --upgrade "jax[cuda12_pip]"==0.4.28 --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

2. Install the remaining dependencies via the project metadata:

   ```powershell
   pip install -e .
   ```

3. Verify the environment:

   ```powershell
   pytest
   ```

## Project layout

- `src/tvc/env.py` – MuJoCo environment with GPU-ready MJX export helpers.
- `src/tvc/dynamics.py` – JAX dynamics, linearisation, and batched rollouts.
- `src/tvc/mpc.py` – Gradient-based MPC utilities for TVC optimisation.
- `src/tvc/policies.py` – Flax actor-critic with evolutionary mutations.
- `src/tvc/curriculum.py` – Scenario scheduling for curriculum learning.
- `src/tvc/training.py` – PPO + evolution training loop integrating MPC.
- `assets/tvc_2d.xml` – MJCF model describing the planar rocket and TVC gear.

## Usage

```python
import jax
from tvc import Tvc2DEnv, train_controller

key = jax.random.key(0)
env = Tvc2DEnv()
result = train_controller(env, total_episodes=200, rng=key)
```

The returned `result` contains the optimised parameters, optimiser state, elite
population, and RNG handle for continued training.
