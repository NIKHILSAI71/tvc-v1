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
from .ppo import (
    ResidualPPOTrainer, PPOParameters, TVCResidualEnv
)

__all__ = [
    # Networks
    'SimpleControlNetwork', 'ResidualPolicyNetwork', 'ValueNetwork', 'NetworkConfig',
    'create_evolution_network', 'create_residual_policy', 'create_value_network',
    
    # Evolution
    'EvolutionaryTrainer', 'EvolutionParameters', 'Individual', 'evaluate_network_fitness',
    
    # PPO
    'ResidualPPOTrainer', 'PPOParameters', 'TVCResidualEnv'
]