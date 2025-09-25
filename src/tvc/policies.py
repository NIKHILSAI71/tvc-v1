"""Policy architectures for PPO with evolutionary perturbations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class PolicyConfig:
    """Defines the actor-critic network structure and initialisation settings."""

    hidden_dims: Tuple[int, ...] = (128, 128, 128)
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    log_std_init: float = -0.5


class ActorCritic(nn.Module):
    """Shared-body actor-critic network handling Gaussian policy outputs."""

    config: PolicyConfig

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x = obs
        for width in self.config.hidden_dims:
            x = nn.Dense(width)(x)
            x = self.config.activation(x)
        mean = nn.Dense(2, name="policy_head")(x)
        value = nn.Dense(1, name="value_head")(x)
        log_std = self.param("log_std", lambda k: jnp.full((2,), self.config.log_std_init))
        return mean, log_std, value.squeeze(-1)


@dataclass(frozen=True)
class PolicyFunctions:
    """Callable collection for policy initialisation and inference."""

    init: Callable[[jax.random.KeyArray, jnp.ndarray], Dict[str, Any]]
    actor: Callable[[Dict[str, Any], jnp.ndarray, jax.random.KeyArray | None, bool], jnp.ndarray]
    value: Callable[[Dict[str, Any], jnp.ndarray], jnp.ndarray]
    distribution: Callable[[Dict[str, Any], jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]


def build_policy_network(config: PolicyConfig = PolicyConfig()) -> PolicyFunctions:
    """Constructs the actor-critic network and returns convenient callables."""

    module = ActorCritic(config)

    def init_fn(key: jax.random.KeyArray, sample_obs: jnp.ndarray) -> Dict[str, Any]:
        variables = module.init(key, sample_obs)
        return variables

    def actor_fn(
        variables: Dict[str, Any],
        observation: jnp.ndarray,
        key: jax.random.KeyArray | None = None,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        mean, log_std, _ = module.apply(variables, observation)
        if deterministic or key is None:
            return mean
        std = jnp.exp(log_std)
        return mean + std * jax.random.normal(key, shape=mean.shape)

    def value_fn(variables: Dict[str, Any], observation: jnp.ndarray) -> jnp.ndarray:
        _, _, value = module.apply(variables, observation)
        return value

    def distribution_fn(
        variables: Dict[str, Any], observation: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return module.apply(variables, observation)

    return PolicyFunctions(
        init=init_fn,
        actor=actor_fn,
        value=value_fn,
        distribution=distribution_fn,
    )


def evaluate_policy(
    variables: Dict[str, Any],
    observation: jnp.ndarray,
    rng: jax.random.KeyArray | None,
    deterministic: bool = False,
    config: PolicyConfig = PolicyConfig(),
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Evaluates the policy and returns the action with diagnostics."""

    funcs = build_policy_network(config)
    action = funcs.actor(variables, observation, rng, deterministic)
    mean, log_std, value = funcs.distribution(variables, observation)
    diagnostics = {"mean": mean, "log_std": log_std, "value": value}
    return action, diagnostics


def mutate_parameters(
    rng: jax.random.KeyArray,
    variables: Dict[str, Any],
    scale: float = 0.02,
) -> Dict[str, Any]:
    """Applies isotropic Gaussian mutations to policy parameters for evolution."""

    leaves, tree_def = jax.tree_util.tree_flatten(variables)
    keys = jax.random.split(rng, len(leaves))

    mutated_leaves = [
        leaf + scale * jax.random.normal(key, shape=leaf.shape, dtype=leaf.dtype)
        for leaf, key in zip(leaves, keys)
    ]
    return jax.tree_util.tree_unflatten(tree_def, mutated_leaves)
