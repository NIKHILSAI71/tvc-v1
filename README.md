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

## Installation

1. Install system dependencies (MuJoCo and Python 3.10+).
2. Install project requirements in a virtual environment:
   ```bash
   pip install -e .
   ```
3. Download MuJoCo assets if needed (see `assets/` folder for details).

### GPU runtimes (JAX)

If you plan to use NVIDIA GPUs, ensure the JAX CUDA plugin stack matches the
installed `jaxlib` version. When CUDA packages are out of sync, the CLI will
temporarily force CPU execution and print upgrade instructions similar to:

```bash
pip install --upgrade "jax==0.7.2" "jaxlib==0.7.2" \
    "jax-cuda12-plugin[with-cuda]==0.7.2" "jax-cuda12-pjrt==0.7.2"
```

After upgrading you can opt back into GPU execution (if desired) by setting:

```bash
set JAX_PLATFORMS=gpu  # Windows PowerShell
```

On Unix shells use `export` instead of `set`.

When installing the package with GPU support on Linux, use the optional extra to
pull in matching CUDA plugins automatically:

```bash
pip install -e .[gpu]
```

## Running Tests

Run the test suite using `pytest`:

```bash
pytest
```

## Usage

To train the controller:

```bash
python -m tvc train --episodes 100 --seed 42
```

To run the smoke tests:

```bash
python -m tvc test
```
