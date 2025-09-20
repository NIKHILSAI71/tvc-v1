"""
Learning module for TVC system
"""

from .networks import (
    SimpleControlNetwork, ResidualPolicyNetwork, ValueNetwork,
    NetworkConfig, create_evolution_network, create_residual_policy, create_value_network
)
from .evolution import (
    EvolutionaryTrainer, EvolutionParameters, Individual,
    evaluate_network_fitness
)
try:
    from .ppo import (
        ResidualPPOTrainer, PPOParameters, TVCResidualEnv
    )
except Exception:
    # Optional dependency (torchrl). These will be None if unavailable.
    ResidualPPOTrainer = None  # type: ignore
    PPOParameters = None  # type: ignore
    TVCResidualEnv = None  # type: ignore

__all__ = [
    # Networks
    'SimpleControlNetwork', 'ResidualPolicyNetwork', 'ValueNetwork', 'NetworkConfig',
    'create_evolution_network', 'create_residual_policy', 'create_value_network',
    
    # Evolution
    'EvolutionaryTrainer', 'EvolutionParameters', 'Individual', 'evaluate_network_fitness',
]

# Conditionally extend exports with PPO-related symbols if available
if 'ResidualPPOTrainer' in globals() and ResidualPPOTrainer is not None:
    __all__.extend(['ResidualPPOTrainer'])
if 'PPOParameters' in globals() and PPOParameters is not None:
    __all__.extend(['PPOParameters'])
if 'TVCResidualEnv' in globals() and TVCResidualEnv is not None:
    __all__.extend(['TVCResidualEnv'])