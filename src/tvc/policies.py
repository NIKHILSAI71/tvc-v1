"""Policy architectures for PPO with evolutionary perturbations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, cast
from typing import Protocol

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import FrozenDict

PRNGKey = jax.Array
Variables = FrozenDict[str, Any]


class InitFn(Protocol):
    def __call__(self, key: PRNGKey, sample_obs: jnp.ndarray) -> Variables: ...


class ActorFn(Protocol):
    def __call__(
        self,
        variables: Variables,
        observation: jnp.ndarray,
        key: PRNGKey | None = ...,
        deterministic: bool = ...,
    ) -> jnp.ndarray: ...


class ValueFn(Protocol):
    def __call__(self, variables: Variables, observation: jnp.ndarray) -> jnp.ndarray: ...


class DistributionFn(Protocol):
    def __call__(
        self,
        variables: Variables,
        observation: jnp.ndarray,
        key: PRNGKey | None = ...,
        deterministic: bool = ...,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: ...


@dataclass(frozen=True)
class PolicyConfig:
    """Defines the actor-critic network structure and initialisation settings."""

    hidden_dims: Tuple[int, ...] = (512, 512, 256, 128)
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    log_std_init: float = -0.3
    dropout_rate: float = 0.05
    use_layer_norm: bool = True


class ActorCritic(nn.Module):
    """Shared-body actor-critic network handling Gaussian policy outputs."""

    config: PolicyConfig

    @nn.compact
    def __call__(self, obs: jnp.ndarray, *, deterministic: bool) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x = obs
        for idx, width in enumerate(self.config.hidden_dims):
            x = nn.Dense(width, name=f"mlp_{idx}_dense")(x)
            if self.config.use_layer_norm:
                x = nn.LayerNorm(name=f"mlp_{idx}_norm")(x)
            x = self.config.activation(x)
            if self.config.dropout_rate > 0.0:
                x = nn.Dropout(rate=self.config.dropout_rate, name=f"mlp_{idx}_dropout")(x, deterministic=deterministic)
        mean = nn.Dense(2, name="policy_head")(x)
        value = nn.Dense(1, name="value_head")(x)
        log_std = self.param("log_std", lambda k: jnp.full((2,), self.config.log_std_init))
        return mean, log_std, value.squeeze(-1)


@dataclass(frozen=True)
class PolicyFunctions:
    """Callable collection for policy initialisation and inference."""

    init: InitFn
    actor: ActorFn
    value: ValueFn
    distribution: DistributionFn


def build_policy_network(config: PolicyConfig = PolicyConfig()) -> PolicyFunctions:
    """Constructs the actor-critic network and returns convenient callables."""

    module = ActorCritic(config)

    def init_fn(key: PRNGKey, sample_obs: jnp.ndarray) -> Variables:
        variables = module.init(key, sample_obs, deterministic=True)
        return cast(Variables, variables)

    def _apply(
        variables: Variables,
        observation: jnp.ndarray,
        *,
        deterministic: bool,
        rng: PRNGKey | None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if config.dropout_rate > 0.0 and not deterministic and rng is not None:
            outputs = module.apply(variables, observation, deterministic=deterministic, rngs={"dropout": rng})
        else:
            outputs = module.apply(variables, observation, deterministic=deterministic)
        return cast(Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], outputs)

    def actor_fn(
        variables: Variables,
        observation: jnp.ndarray,
        key: PRNGKey | None = None,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        dropout_key: PRNGKey | None = None
        sample_key: PRNGKey | None = None
        if key is not None:
            dropout_key, sample_key = jax.random.split(key, num=2)
        mean, log_std, _ = _apply(variables, observation, deterministic=deterministic or sample_key is None, rng=dropout_key)
        if deterministic or sample_key is None:
            return mean
        std = jnp.exp(log_std)
        return mean + std * jax.random.normal(sample_key, shape=mean.shape)

    def value_fn(variables: Variables, observation: jnp.ndarray) -> jnp.ndarray:
        _, _, value = _apply(variables, observation, deterministic=True, rng=None)
        return value

    def distribution_fn(
        variables: Variables,
        observation: jnp.ndarray,
        key: PRNGKey | None = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return _apply(variables, observation, deterministic=deterministic, rng=key)

    return PolicyFunctions(
        init=init_fn,
        actor=actor_fn,
        value=value_fn,
        distribution=distribution_fn,
    )


def evaluate_policy(
    variables: Variables,
    observation: jnp.ndarray,
    rng: PRNGKey | None,
    deterministic: bool = False,
    config: PolicyConfig = PolicyConfig(),
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Evaluates the policy and returns the action with diagnostics."""

    funcs = build_policy_network(config)
    action = funcs.actor(variables, observation, rng, deterministic)
    mean, log_std, value = funcs.distribution(variables, observation, key=None, deterministic=True)
    diagnostics = {"mean": mean, "log_std": log_std, "value": value}
    return action, diagnostics


def mutate_parameters(
    rng: PRNGKey,
    variables: Variables,
    scale: float = 0.02,
) -> Variables:
    """Applies isotropic Gaussian mutations to policy parameters for evolution."""

    leaves, tree_def = jax.tree_util.tree_flatten(variables)
    keys = jax.random.split(rng, num=len(leaves))

    mutated_leaves = [
        leaf + scale * jax.random.normal(key, shape=leaf.shape, dtype=leaf.dtype)
        for leaf, key in zip(leaves, keys)
    ]
    return cast(Variables, jax.tree_util.tree_unflatten(tree_def, mutated_leaves))
