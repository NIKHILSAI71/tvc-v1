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

### Training defaults

The training stack now includes stability-oriented defaults:

- Softer curriculum thresholds (`reward_threshold ≥ -110`) with gentler disturbances for earlier stages.
- Reward shaping bonuses for low pitch and lateral error, plus explicit control smoothness penalties.
- Progressive MPC-to-policy action blending (70→20% MPC) over the first ~300 stage episodes.
- PPO hyperparameters tuned for negative-reward regimes (`reward_scale=1.0`, `clip ε=0.15`, `rollout_length=512`, `entropy_coef=5e-3`).
- A deeper policy network (512-512-256-128) with lower dropout and conservative log-std initialisation.

#### CLI overrides

`python -m tvc train` now exposes key knobs so runs can be customised without editing code:

- `--learning-rate`, `--entropy-coef`, `--reward-scale`, `--rollout-length` override the PPO loop.
- `--policy-weight`, `--mpc-weight`, `--policy-warmup-weight`, `--mpc-warmup-weight`, `--blend-transition-episodes`, `--no-progressive-blend` control the MPC/policy blend schedule.
- Existing plateau-management flags still apply; combine them to experiment with different LR reduction strategies.

Each training invocation now creates a timestamped run directory under `training/` by default, e.g. `training/run-2025-09-24-04-30PM/`. The folder contains:

- `training.log` – the full textual log for the run.
- `metrics.csv` / `metrics.json` – per-episode statistics (returns, elite scores, environment diagnostics).
- `metrics.png` – an aggregated visualization of rewards, stabilisation metrics, and the learning-rate schedule.
- `config.json` – the serialisable subset of the training configuration used for the run.
- `policy_final.msgpack` – the trained PPO parameters corresponding to the final episode.
- `policy_elite_best.msgpack` – the highest-scoring elite retained by the evolutionary phase (when available).
- `observation_normalizer.npz` – running mean/variance statistics for normalising observations at inference time.
- `policy_metadata.json` – handy metadata describing the best recorded episode and learning-rate state.

Pass `--output-root` to change the base directory or `--run-tag` to supply a custom run name. The new warmup cosine learning-rate schedule can be disabled with `--no-lr-schedule` if desired.

To run the smoke tests:

```bash
python -m tvc test
```
