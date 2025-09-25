"""Training pipeline combining PPO, evolutionary strategies, and MPC."""
from __future__ import annotations

if __package__ in (None, ""):
    import pathlib
    import sys

    package_root = pathlib.Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.append(str(package_root))
    __package__ = "tvc"

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .curriculum import CurriculumStage, build_curriculum, select_stage
from .dynamics import RocketParams
from .env import StepResult, Tvc2DEnv
from .mpc import MpcConfig, compute_tvc_mpc_action
from .policies import PolicyConfig, build_policy_network, mutate_parameters


@dataclass(frozen=True)
class PpoEvolutionConfig:
    """Aggregates PPO and evolutionary hyperparameters."""

    gamma: float = 0.985
    lam: float = 0.92
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.18
    rollout_length: int = 1024
    num_epochs: int = 4
    minibatch_size: int = 256
    mutation_scale: float = 0.015
    population_size: int = 16
    elite_keep: int = 4
    policy_config: PolicyConfig = PolicyConfig()
    mpc_config: MpcConfig = MpcConfig()
    params: RocketParams = RocketParams()


@dataclass
class TrainingState:
    params: Dict[str, object]
    opt_state: optax.OptState
    elites: List[Dict[str, object]]
    rng: jax.random.KeyArray


def _gather_state(env: Tvc2DEnv) -> jnp.ndarray:
    x = float(env.data.qpos[0])
    z = float(env.data.qpos[2])
    vx = float(env.data.qvel[0])
    vz = float(env.data.qvel[2])
    theta = env._pitch()
    omega = env._pitch_rate()
    return jnp.array([x, z, vx, vz, theta, omega], dtype=jnp.float32)


def _gaussian_log_prob(mean: jnp.ndarray, log_std: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    var = jnp.exp(2.0 * log_std)
    return -0.5 * (jnp.sum(((action - mean) ** 2) / var + 2 * log_std + jnp.log(2 * jnp.pi)))


def _compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float,
    lam: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    def scan_fn(carry, inputs):
        reward, value, next_value, done = inputs
        delta = reward + gamma * (1.0 - done) * next_value - value
        advantage = delta + gamma * lam * (1.0 - done) * carry
        return advantage, advantage

    _, advantages_rev = jax.lax.scan(
        scan_fn,
        0.0,
        (rewards[::-1], values[:-1][::-1], values[1:][::-1], dones[::-1]),
    )
    advantages = advantages_rev[::-1]
    returns = advantages + values[:-1]
    return advantages, returns


def train_controller(
    env: Tvc2DEnv,
    total_episodes: int,
    rng: jax.random.KeyArray,
    config: PpoEvolutionConfig = PpoEvolutionConfig(),
) -> TrainingState:
    """Runs PPO with evolutionary refinement and MPC guided actions."""

    policy_funcs = build_policy_network(config.policy_config)
    sample_obs = jnp.zeros((3,), dtype=jnp.float32)
    params = policy_funcs.init(rng, sample_obs)

    optimiser = optax.adam(config.learning_rate)
    opt_state = optimiser.init(params)

    curriculum = build_curriculum()
    elites: List[Dict[str, object]] = []
    state = TrainingState(params=params, opt_state=opt_state, elites=elites, rng=rng)
    logger = logging.getLogger("tvc.training")

    logger.info("Beginning training run for %s episodes", total_episodes)

    for episode in range(total_episodes):
        stage = select_stage(curriculum, episode)
        trajectories = _collect_rollout(env, stage, state, policy_funcs, config)
        state = _ppo_update(state, trajectories, optimiser, policy_funcs, config)
        episode_return = float(jnp.sum(trajectories["rewards"]))
        reward_mean = float(jnp.mean(trajectories["rewards"]))
        state, elite_best, elite_mean = _evolutionary_update(env, stage, state, policy_funcs, config)

        logger.info(
            "Episode %s/%s | stage=%s (disturbance=%.2f) | rollout_return=%.3f | reward_mean=%.3f | elite_best=%.3f | elite_mean=%.3f | elites=%s",
            episode + 1,
            total_episodes,
            stage.name,
            stage.disturbance_scale,
            episode_return,
            reward_mean,
            elite_best,
            elite_mean,
            len(state.elites),
        )

    logger.info("Training loop complete. Elites maintained: %s", len(state.elites))
    return state


def _collect_rollout(
    env: Tvc2DEnv,
    stage: CurriculumStage,
    state: TrainingState,
    funcs,
    config: PpoEvolutionConfig,
):
    obs_buffer: List[jnp.ndarray] = []
    action_buffer: List[jnp.ndarray] = []
    logprob_buffer: List[float] = []
    reward_buffer: List[float] = []
    value_buffer: List[float] = []
    done_buffer: List[float] = []

    observation = jnp.asarray(env.reset(disturbance_scale=stage.disturbance_scale))
    for _ in range(config.rollout_length):
        state.rng, policy_key = jax.random.split(state.rng)
        mean, log_std, value = funcs.distribution(state.params, observation)
        std = jnp.exp(log_std)
        epsilon = jax.random.normal(policy_key, shape=mean.shape)
        policy_action = mean + std * epsilon
        rocket_state = _gather_state(env)
        mpc_action, _ = compute_tvc_mpc_action(rocket_state, stage.target_state, config.params, config.mpc_config)
        combined_action = jnp.clip(policy_action + mpc_action, -config.mpc_config.control_limit, config.mpc_config.control_limit)

        log_prob = _gaussian_log_prob(mean, log_std, policy_action)

        result: StepResult = env.step(np.array(combined_action))

        obs_buffer.append(observation)
        action_buffer.append(policy_action)
        logprob_buffer.append(float(log_prob))
        reward_buffer.append(result.reward)
        value_buffer.append(float(value))
        done_buffer.append(float(result.done))

        observation = jnp.asarray(result.observation)
        if result.done:
            observation = jnp.asarray(env.reset(disturbance_scale=stage.disturbance_scale))

    # Bootstrap value for final state.
    value_buffer.append(float(funcs.value(state.params, observation)))

    batch = {
        "observations": jnp.stack(obs_buffer),
        "actions": jnp.stack(action_buffer),
        "log_probs": jnp.array(logprob_buffer),
        "rewards": jnp.array(reward_buffer),
        "values": jnp.array(value_buffer),
        "dones": jnp.array(done_buffer),
    }
    return batch


def _ppo_update(
    state: TrainingState,
    batch: Dict[str, jnp.ndarray],
    optimiser: optax.GradientTransformation,
    funcs,
    config: PpoEvolutionConfig,
) -> TrainingState:
    advantages, returns = _compute_gae(
        batch["rewards"],
        batch["values"],
        batch["dones"],
        config.gamma,
        config.lam,
    )

    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    dataset = {
        "obs": batch["observations"],
        "actions": batch["actions"],
        "log_probs": batch["log_probs"],
        "advantages": advantages,
        "returns": returns,
    }

    def loss_fn(params, minibatch):
        mean, log_std, values = funcs.distribution(params, minibatch["obs"])
        log_prob = jax.vmap(_gaussian_log_prob, in_axes=(0, None, 0))(mean, log_std, minibatch["actions"])
        ratio = jnp.exp(log_prob - minibatch["log_probs"])
        clipped = jnp.clip(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
        actor_loss = -jnp.mean(jnp.minimum(ratio * minibatch["advantages"], clipped * minibatch["advantages"]))
        value_loss = 0.5 * jnp.mean((minibatch["returns"] - values) ** 2)
        entropy = 0.5 * jnp.sum(1.0 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
        return actor_loss + value_loss - 0.001 * entropy

    params = state.params
    opt_state = state.opt_state
    num_samples = dataset["obs"].shape[0]

    for _ in range(config.num_epochs):
        permutation = np.random.permutation(num_samples)
        for start in range(0, num_samples, config.minibatch_size):
            idx = permutation[start : start + config.minibatch_size]
            minibatch = {k: v[idx] for k, v in dataset.items()}
            grads = jax.grad(loss_fn)(params, minibatch)
            updates, opt_state = optimiser.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

    state.params = params
    state.opt_state = opt_state
    return state


def _evolutionary_update(
    env: Tvc2DEnv,
    stage: CurriculumStage,
    state: TrainingState,
    funcs,
    config: PpoEvolutionConfig,
) -> Tuple[TrainingState, float, float]:
    rng = state.rng
    population = [state.params] + state.elites
    while len(population) < config.population_size:
        rng, key = jax.random.split(rng)
        population.append(mutate_parameters(key, state.params, config.mutation_scale))

    scores = []
    for candidate in population:
        reward = _evaluate_candidate(env, stage, candidate, funcs)
        scores.append(reward)

    elite_indices = np.argsort(scores)[-config.elite_keep :][::-1]
    elites = [population[i] for i in elite_indices]
    state.elites = elites
    state.rng = rng

    best_reward = float(np.max(scores)) if scores else float("nan")
    mean_reward = float(np.mean(scores)) if scores else float("nan")
    return state, best_reward, mean_reward


def _evaluate_candidate(
    env: Tvc2DEnv,
    stage: CurriculumStage,
    params,
    funcs,
    episodes: int = 3,
) -> float:
    total = 0.0
    for _ in range(episodes):
        obs = jnp.asarray(env.reset(disturbance_scale=stage.disturbance_scale))
        done = False
        steps = 0
        while not done and steps < 400:
            action = funcs.actor(params, obs, None, True)
            result = env.step(np.array(action))
            total += result.reward
            obs = jnp.asarray(result.observation)
            done = result.done
            steps += 1
    return total / episodes
