"""Policy networks for 3D TVC control."""

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
    """Policy network configuration for 3D TVC."""
    hidden_dims: Tuple[int, ...] = (512, 512, 256, 128)
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    log_std_init: float = -0.3
    log_std_min: float = -2.5   # Allow tighter control when converged
    log_std_max: float = 0.3    # Allow higher exploration (std up to 1.35)
    dropout_rate: float = 0.0
    use_layer_norm: bool = True
    action_limit: float = 0.14  # Gimbal angle limit (±8° = 0.14 rad)
    thrust_limit: float = 1.0   # Thrust fraction [0, 1]
    action_dims: int = 3        # 3D: [gimbal_x_angle, gimbal_y_angle, thrust]


class ActorCritic(nn.Module):
    """Shared-body actor-critic network for 3D control."""

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

        # Policy head: [gimbal_x, gimbal_y, thrust]
        mean_raw = nn.Dense(3, name="policy_head")(x)
        gimbal_mean = jnp.tanh(mean_raw[:2]) * self.config.action_limit
        thrust_mean = nn.sigmoid(mean_raw[2:3]) * self.config.thrust_limit
        mean = jnp.concatenate([gimbal_mean, thrust_mean])

        # Value head
        value = nn.Dense(1, name="value_head")(x)

        # Log std head
        log_std_bias = nn.initializers.constant(self.config.log_std_init)
        log_std_head = nn.Dense(
            3,
            name="log_std_head",
            kernel_init=nn.initializers.zeros,
            bias_init=log_std_bias,
        )(x)
        log_std = jnp.clip(log_std_head, self.config.log_std_min, self.config.log_std_max)

        return mean, log_std, value.squeeze(-1)


@dataclass(frozen=True)
class PolicyFunctions:
    """Callable collection for policy operations."""
    init: InitFn
    actor: ActorFn
    value: ValueFn
    distribution: DistributionFn


def build_policy_network(config: PolicyConfig = PolicyConfig()) -> PolicyFunctions:
    """Build actor-critic network and return interface functions."""
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


def mutate_parameters(
    rng: PRNGKey,
    variables: Variables,
    scale: float = 0.02,
    mutation_prob: float = 0.8,
) -> Variables:
    """Apply selective Gaussian mutations to policy parameters.

    Based on rocket stabilization neuroevolution:
    - Only mutate a fraction of parameters (controlled by mutation_prob)
    - Use Gaussian noise scaled by parameter magnitude
    - Helps maintain network structure while exploring

    Args:
        rng: JAX random key
        variables: Policy parameters to mutate
        scale: Mutation scale factor
        mutation_prob: Probability of mutating each parameter

    Returns:
        Mutated parameters
    """
    leaves, tree_def = jax.tree_util.tree_flatten(variables)
    rng_keys = jax.random.split(rng, num=len(leaves) * 2)
    mut_keys = rng_keys[:len(leaves)]
    mask_keys = rng_keys[len(leaves):]

    mutated_leaves = []
    for leaf, mut_key, mask_key in zip(leaves, mut_keys, mask_keys):
        # Generate mutation mask (which parameters to mutate)
        mutation_mask = jax.random.uniform(mask_key, shape=leaf.shape) < mutation_prob

        # Generate Gaussian noise scaled by parameter magnitude + scale
        noise = jax.random.normal(mut_key, shape=leaf.shape, dtype=leaf.dtype)
        adaptive_scale = scale * (1.0 + jnp.abs(leaf))  # Larger params get larger mutations

        # Apply masked mutation
        mutation = jnp.where(mutation_mask, noise * adaptive_scale, 0.0)
        mutated_leaves.append(leaf + mutation)

    return cast(Variables, jax.tree_util.tree_unflatten(tree_def, mutated_leaves))
