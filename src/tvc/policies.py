"""
Module: policies
Purpose: Policy networks for 3D TVC control with GRU recurrence.
Complexity: Time O(T * H^2) | Space O(T * H) where T=sequence length, H=hidden dim
Dependencies: flax, jax, jax.numpy
Last Updated: 2026-01-03
"""

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
# GRU uses single hidden state (simpler than LSTM's (h, c) tuple)
GRUState = jnp.ndarray  # Shape: [Batch, HiddenDim]

class InitFn(Protocol):
    def __call__(self, key: PRNGKey, sample_obs: jnp.ndarray, sample_hidden: GRUState) -> Tuple[Variables, GRUState]: ...

class ActorFn(Protocol):
    def __call__(
        self, variables: Variables, observation: jnp.ndarray, hidden: GRUState, dones: jnp.ndarray, 
        key: PRNGKey | None = ..., deterministic: bool = ...
    ) -> Tuple[jnp.ndarray, GRUState]: ...

class DistributionFn(Protocol):
    def __call__(
        self, variables: Variables, observation: jnp.ndarray, hidden: GRUState, dones: jnp.ndarray, 
        key: PRNGKey | None = ..., deterministic: bool = ...
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, GRUState]: ...

@dataclass(frozen=True)
class PolicyConfig:
    hidden_dims: Tuple[int, ...] = (512, 256)  # Larger network for complex behaviors
    gru_hidden_dim: int = 256  # More temporal memory for smooth control
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    log_std_init: float = -2.0  # Reduced from -1.2: start with small exploration to prevent oscillation
    log_std_min: float = -4.0
    log_std_max: float = 0.0
    dropout_rate: float = 0.0
    use_layer_norm: bool = True
    action_limit: float = 0.14
    thrust_limit: float = 1.0
    thrust_init_bias: float = -1.0
    action_dims: int = 3

class MaskedGRUCell(nn.Module):
    """GRUCell with internal reset logic. Simpler than LSTM - single hidden state."""
    features: int

    @nn.compact
    def __call__(self, carry: GRUState, inputs: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[GRUState, jnp.ndarray]:
        x, done = inputs
        # Support broadcasting
        done = jnp.reshape(done, (-1, 1))
        
        # GRU has single hidden state (not tuple like LSTM)
        h = carry * (1.0 - done)  # Reset hidden on episode boundary
        
        new_carry, out = nn.GRUCell(self.features)(h, x)
        return new_carry, out

class RecurrentActorCritic(nn.Module):
    """
    Recurrent Actor-Critic network with Masked GRU.
    
    Complexity Analysis:
        Time: O(L * H^2) per sequence forward pass, where L is sequence length and H is hidden dimension.
        Space: O(L * H) to store hidden states for backpropagation.
    
    Design Decisions:
        - Uses MaskedGRUCell (simpler than LSTM, 30% fewer params).
        - GRU uses single hidden state vs LSTM's (h, c) tuple.
        - Shared encoder for both Actor and Critic heads.
        - LayerNorm used for training stability.
    """
    config: PolicyConfig

    @nn.compact
    def __call__(self, obs: jnp.ndarray, hidden: GRUState, dones: jnp.ndarray, deterministic: bool = False):
        is_sequence = obs.ndim == 3
        if not is_sequence:
            obs = obs[:, None, :]
            dones = dones[:, None]
            
        # Encoder
        x = obs
        for width in self.config.hidden_dims:
            x = nn.Dense(width)(x)
            if self.config.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.config.activation(x)
                
        # GRU (simpler than LSTM - single hidden state)
        # Transpose to [Time, Batch, Features] for scanning
        x_t = jnp.transpose(x, (1, 0, 2))
        dones_t = jnp.transpose(dones, (1, 0))
        
        # Scan the MaskedGRUCell
        gru_scan = nn.scan(
            MaskedGRUCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0, out_axes=0
        )(features=self.config.gru_hidden_dim)
        
        # Pack inputs correctly as a tuple matching MaskedGRUCell.__call__ signature
        final_state, x_out = gru_scan(hidden, (x_t, dones_t))
        
        # Transpose back to [Batch, Time, Features]
        x_out = jnp.transpose(x_out, (1, 0, 2))
        
        # Heads
        mean_raw = nn.Dense(3, name="policy_head")(x_out)
        gimbal_mean = jnp.tanh(mean_raw[..., :2]) * self.config.action_limit
        thrust_logit = mean_raw[..., 2:3] + self.config.thrust_init_bias
        thrust_mean = nn.sigmoid(thrust_logit) * self.config.thrust_limit
        mean = jnp.concatenate([gimbal_mean, thrust_mean], axis=-1)
        
        value = nn.Dense(1, name="value_head")(x_out).squeeze(-1)
        
        log_std_bias = nn.initializers.constant(self.config.log_std_init)
        log_std = nn.Dense(3, kernel_init=nn.initializers.zeros, bias_init=log_std_bias, name="log_std_head")(x_out)
        log_std = jnp.clip(log_std, self.config.log_std_min, self.config.log_std_max)
        
        if not is_sequence:
            mean, value, log_std = mean.squeeze(1), value.squeeze(1), log_std.squeeze(1)
            
        return mean, log_std, value, final_state

    def initialize_carrier(self, batch_size: int) -> GRUState:
        # GRU uses single hidden state (simpler than LSTM)
        return jnp.zeros((batch_size, self.config.gru_hidden_dim))

@dataclass(frozen=True)
class PolicyFunctions:
    init: InitFn
    actor: ActorFn
    value: Any
    distribution: DistributionFn

def build_policy_network(config: PolicyConfig = PolicyConfig()) -> PolicyFunctions:
    module = RecurrentActorCritic(config)
    
    def init_fn(key: PRNGKey, sample_obs: jnp.ndarray, sample_hidden: GRUState) -> Tuple[Variables, GRUState]:
        dones = jnp.zeros((sample_obs.shape[0],), dtype=jnp.float32)
        variables = module.init(key, sample_obs, sample_hidden, dones, deterministic=True)
        return cast(Variables, variables), sample_hidden

    # JIT-compiled for fast inference during rollouts
    @jax.jit
    def distribution_fn(params, obs, hidden, dones, key=None, deterministic=True):
         return module.apply(params, obs, hidden, dones, deterministic=deterministic)

    def actor_fn(params, obs, hidden, dones, key=None, deterministic=False):
        mean, log_std, _, new_hidden = distribution_fn(params, obs, hidden, dones, key, deterministic)
        if deterministic or key is None:
            return mean, new_hidden
        std = jnp.exp(log_std)
        return mean + std * jax.random.normal(key, shape=mean.shape), new_hidden

    return PolicyFunctions(init=init_fn, actor=actor_fn, value=None, distribution=distribution_fn)


def safe_mutate_parameters_smg(
    rng: PRNGKey,
    variables: Variables,
    funcs,
    sample_obs: jnp.ndarray,
    sample_hidden: GRUState,
    scale: float = 0.05,
    sensitivity_clip: float = 10.0,
) -> Variables:
    """Safe Mutation through Gradients (SM-G) for GRU networks.
    
    Scales mutation magnitude inversely to output sensitivity.
    Weights that strongly affect outputs get smaller mutations.
    This prevents breaking learned temporal dynamics.
    
    Complexity Analysis:
        Time: O(P) where P is number of parameters (requires one grad pass).
        Space: O(P) to store gradients and mutations.

    Reference: "Safe Mutations for Deep and Recurrent Neural Networks 
    through Output Gradients" (Lehman et al., 2018)
    """
    
    # Compute output sensitivity for each parameter
    # sample_obs: [Batch, SeqLen, Features] for sequence input
    def output_fn(params):
        batch_size = sample_obs.shape[0]
        seq_len = sample_obs.shape[1] if sample_obs.ndim == 3 else 1
        dones = jnp.zeros((batch_size, seq_len))  # [Batch, SeqLen] for sequence
        mean, _, _, _ = funcs.distribution(params, sample_obs, sample_hidden, dones, deterministic=True)
        return jnp.sum(jnp.square(mean))  # Scalar output for gradient
    
    # Compute gradients (sensitivity of outputs to each weight)
    sensitivities = jax.grad(output_fn)(variables)
    
    # Flatten both variables and sensitivities
    var_leaves, tree_def = jax.tree_util.tree_flatten(variables)
    sens_leaves, _ = jax.tree_util.tree_flatten(sensitivities)
    
    rng_keys = jax.random.split(rng, num=len(var_leaves))
    
    mutated_leaves = []
    for leaf, sens, key in zip(var_leaves, sens_leaves, rng_keys):
        # Compute safe mutation scale per weight
        # Higher sensitivity -> smaller mutation
        abs_sens = jnp.abs(sens) + 1e-8
        clipped_sens = jnp.clip(abs_sens, 1e-8, sensitivity_clip)
        
        # Inverse sensitivity scaling
        safe_scale = scale / clipped_sens
        safe_scale = jnp.clip(safe_scale, 0.0, scale * 2.0)  # Cap max mutation
        
        # Apply mutation
        noise = jax.random.normal(key, shape=leaf.shape, dtype=leaf.dtype)
        mutation = noise * safe_scale
        mutated_leaves.append(leaf + mutation)
    
    return cast(Variables, jax.tree_util.tree_unflatten(tree_def, mutated_leaves))


def mutate_parameters(rng, variables, scale=0.02, mutation_prob=0.8):
    """Legacy mutation function (for non-recurrent networks)."""
    leaves, tree_def = jax.tree_util.tree_flatten(variables)
    rng_keys = jax.random.split(rng, num=len(leaves) * 2)
    mut_keys = rng_keys[:len(leaves)]
    mask_keys = rng_keys[len(leaves):]
    mutated_leaves = []
    for leaf, mut_key, mask_key in zip(leaves, mut_keys, mask_keys):
        mutation_mask = jax.random.uniform(mask_key, shape=leaf.shape) < mutation_prob
        noise = jax.random.normal(mut_key, shape=leaf.shape, dtype=leaf.dtype)
        adaptive_scale = scale * (1.0 + jnp.abs(leaf))
        mutation = jnp.where(mutation_mask, noise * adaptive_scale, 0.0)
        mutated_leaves.append(leaf + mutation)
    return cast(Variables, jax.tree_util.tree_unflatten(tree_def, mutated_leaves))


def add_parameter_noise(
    rng: PRNGKey,
    variables: Variables,
    scale: float = 0.005,
) -> Variables:
    """Add small noise to parameters for exploration during rollouts.
    
    Unlike mutation, this is meant to be temporary noise added during
    action selection to encourage exploration of the policy space.
    
    Reference: "Parameter Space Noise for Exploration" (Plappert et al., 2017)
    
    Args:
        rng: JAX random key
        variables: Policy parameters
        scale: Noise scale (typically 0.001 - 0.01)
    
    Returns:
        Noisy parameters (use for action selection only, not training)
    """
    leaves, tree_def = jax.tree_util.tree_flatten(variables)
    rng_keys = jax.random.split(rng, num=len(leaves))
    
    noisy_leaves = []
    for leaf, key in zip(leaves, rng_keys):
        noise = jax.random.normal(key, shape=leaf.shape, dtype=leaf.dtype) * scale
        noisy_leaves.append(leaf + noise)
    
    return cast(Variables, jax.tree_util.tree_unflatten(tree_def, noisy_leaves))


def crossover_parameters(
    rng: PRNGKey,
    parent1: Variables,
    parent2: Variables,
    crossover_rate: float = 0.5,
) -> Variables:
    """Uniform crossover between two parameter sets.
    
    Creates a child by randomly selecting each weight from either parent.
    This enables genetic diversity when combined with an elite archive.
    
    Reference: "Neuroevolution Strategies for Episodic Reinforcement Learning"
    
    Args:
        rng: JAX random key
        parent1: First parent parameters
        parent2: Second parent parameters
        crossover_rate: Probability of selecting from parent2 (0.5 = uniform)
    
    Returns:
        Child parameters with mixed genes from both parents
    """
    leaves1, tree_def = jax.tree_util.tree_flatten(parent1)
    leaves2, _ = jax.tree_util.tree_flatten(parent2)
    rng_keys = jax.random.split(rng, num=len(leaves1))
    
    child_leaves = []
    for leaf1, leaf2, key in zip(leaves1, leaves2, rng_keys):
        # Per-weight crossover mask
        mask = jax.random.uniform(key, shape=leaf1.shape) < crossover_rate
        child = jnp.where(mask, leaf2, leaf1)
        child_leaves.append(child)
    
    return cast(Variables, jax.tree_util.tree_unflatten(tree_def, child_leaves))


def blend_parameters(
    parent1: Variables,
    parent2: Variables,
    alpha: float = 0.5,
) -> Variables:
    """Arithmetic blend between two parameter sets.
    
    Creates a child via linear interpolation: child = alpha * parent1 + (1-alpha) * parent2
    Smoother than crossover, good for fine-tuning near optima.
    
    Args:
        parent1: First parent parameters
        parent2: Second parent parameters
        alpha: Blend factor (0.5 = average, 0.8 = mostly parent1)
    
    Returns:
        Blended child parameters
    """
    leaves1, tree_def = jax.tree_util.tree_flatten(parent1)
    leaves2, _ = jax.tree_util.tree_flatten(parent2)
    
    blended_leaves = [
        alpha * l1 + (1 - alpha) * l2 
        for l1, l2 in zip(leaves1, leaves2)
    ]
    
    return cast(Variables, jax.tree_util.tree_unflatten(tree_def, blended_leaves))

