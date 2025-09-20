# TVC Configuration Files

This directory contains YAML configuration files for different TVC (Thrust Vector Control) scenarios and training setups. Each configuration file can be used with the training and evaluation scripts to customize the behavior of the system.

## Usage

### Basic Usage
```bash
# Use a specific configuration file
python train.py --method evolution --config configs/small_rocket.yaml
python train.py --method ppo --config configs/large_rocket.yaml
python evaluate.py --config configs/comparison.yaml

# List available experiments
python evaluate.py --list

# Create example configs (regenerates this folder)
python train.py --create-configs
python evaluate.py --create-configs
```

### Command Line Override
The configuration files provide default values, but you can override them via command line arguments or create custom configurations.

## Available Configurations

### 🚀 **Rocket Type Configurations**

#### `default.yaml`
- **Purpose**: Standard TVC configuration with balanced parameters
- **Use Case**: General-purpose testing and baseline experiments
- **Key Features**: 1kg mass, 30° gimbal range, 1000 population evolution

#### `small_rocket.yaml`
- **Purpose**: Small model rocket (Estes Big Bertha size)
- **Use Case**: Educational rockets, model rocketry, hobbyist applications
- **Key Features**: 0.5kg mass, 20° gimbal, conservative limits, faster training

#### `large_rocket.yaml`
- **Purpose**: Large launch vehicle (Falcon 9 first stage scale)
- **Use Case**: Commercial launch vehicles, heavy-lift rockets
- **Key Features**: 5kg mass, 15° gimbal, tight safety limits, extended training

#### `heavy_payload.yaml`
- **Purpose**: Rockets carrying heavy payloads
- **Use Case**: Satellite deployment, cargo missions, heavy lift
- **Key Features**: 2.5kg mass, structural load considerations, conservative control

#### `aggressive_rocket.yaml`
- **Purpose**: High-performance, agile rocket control
- **Use Case**: Military applications, rapid maneuvering, aerobatic flights
- **Key Features**: 40° gimbal range, 60° angle envelope, high angular rates

### 🎯 **Mission-Specific Configurations**

#### `precision_landing.yaml`
- **Purpose**: High-precision landing applications
- **Use Case**: SpaceX-style propulsive landing, planetary landers
- **Key Features**: Fine control tuning, extended simulation, precision rewards

#### `windy_environment.yaml`
- **Purpose**: Robustness testing against wind disturbances
- **Use Case**: Outdoor flights, weather resistance testing
- **Key Features**: Disturbance parameters, robust control, larger populations

### 🧪 **Training-Focused Configurations**

#### `evolution_only.yaml`
- **Purpose**: Pure neuro-evolutionary training (as per video)
- **Use Case**: Comparing evolution vs other methods, replicating video results
- **Key Features**: 1000 population, no safety filter, extended generations

#### `ppo_only.yaml`
- **Purpose**: Pure residual PPO training
- **Use Case**: Comparing PPO vs other methods, RL research
- **Key Features**: Extended PPO training, no safety filter, large batches

#### `comparison.yaml`
- **Purpose**: Balanced comparison of all methods
- **Use Case**: Research papers, method comparison, benchmarking
- **Key Features**: Fair parameters for all methods, comprehensive evaluation

#### `high_performance.yaml`
- **Purpose**: Maximum performance and accuracy
- **Use Case**: Competition entries, demonstration systems, research validation
- **Key Features**: 2000 population, 5M frames, high-fidelity simulation

#### `safety_focused.yaml`
- **Purpose**: Maximum safety constraints
- **Use Case**: Critical applications, certification testing, fail-safe systems
- **Key Features**: Conservative limits, aggressive barriers, extended validation

### ⚡ **Development Configurations**

#### `quick_test.yaml`
- **Purpose**: Rapid testing and development
- **Use Case**: Code debugging, feature testing, CI/CD pipelines
- **Key Features**: 20 episodes, 100 population, minimal training

#### `development.yaml`
- **Purpose**: Development and debugging
- **Use Case**: Code development, algorithm debugging, prototyping
- **Key Features**: Minimal settings, verbose logging, frequent checkpoints

## Configuration Structure

Each YAML file contains the following sections:

### Experiment Settings
```yaml
experiment_name: "my_experiment"    # Name for results folder
output_dir: "./results/my_exp"      # Output directory
use_safety_filter: true             # Enable/disable safety filter
save_models: true                   # Save trained models
save_plots: true                    # Generate visualizations
eval_episodes: 100                  # Number of evaluation episodes
eval_seed: 42                       # Random seed for reproducibility
```

### Plant Parameters
```yaml
plant:
  mass: 1.0                         # Vehicle mass (kg)
  moment_of_inertia: 0.1           # Moment of inertia (kg⋅m²)
  length: 1.0                      # Vehicle length (m)
  nominal_thrust: 15.0             # Nominal thrust (N)
  max_gimbal_angle: 0.524          # Max gimbal angle (rad)
  max_angle: 0.785                 # Max vehicle angle (rad)
  max_angular_rate: 6.28           # Max angular rate (rad/s)
```

### Control Parameters
```yaml
lqr:
  Q: null                          # State weight matrix (null = auto)
  R: 1.0                           # Control weight
  clg_gamma: 2.0                   # CLF decay rate

safety:
  angle_cbf_alpha: 3.0             # Angle barrier coefficient
  rate_cbf_alpha: 2.0              # Rate barrier coefficient
  clf_penalty: 1000.0              # CLF violation penalty
```

### Learning Parameters
```yaml
evolution:
  population_size: 1000            # Population size
  max_generations: 500             # Maximum generations
  sim_duration: 5.0                # Simulation time per episode
  mutation_strength: 0.1           # Mutation standard deviation

ppo:
  total_frames: 1000000            # Total training frames
  frames_per_batch: 2048           # Frames per batch
  clip_epsilon: 0.2                # PPO clipping parameter
  lr_actor: 0.0003                 # Actor learning rate
  lr_critic: 0.0003                # Critic learning rate
```

### Simulation Parameters
```yaml
simulation:
  timestep: 0.001                  # Physics timestep (s)
  control_timestep: 0.01           # Control update rate (s)
  max_episode_steps: 1000          # Max steps per episode
```

## Creating Custom Configurations

### Method 1: Copy and Modify
```bash
# Copy an existing config
cp configs/default.yaml configs/my_custom.yaml

# Edit the file
# Then use it
python train.py --method evolution --config configs/my_custom.yaml
```

### Method 2: Programmatic Creation
```python
from src.utils import create_default_config, save_config

# Create base config
config = create_default_config()

# Modify parameters
config.experiment_name = "my_experiment"
config.plant.mass = 2.0
config.evolution.population_size = 1500

# Save to file
save_config(config, "configs/my_custom.yaml")
```

## Parameter Guidelines

### Mass and Inertia
- **Small rockets**: 0.3-0.8 kg, I = 0.03-0.08 kg⋅m²
- **Medium rockets**: 0.8-2.0 kg, I = 0.08-0.3 kg⋅m²  
- **Large rockets**: 2.0-10.0 kg, I = 0.3-3.0 kg⋅m²

### Gimbal Angles
- **Conservative**: 10-20° (0.175-0.349 rad)
- **Standard**: 20-30° (0.349-0.524 rad)
- **Aggressive**: 30-45° (0.524-0.785 rad)

### Safety Limits
- **Tight**: ±20-30° angle, ±180-240°/s rate
- **Standard**: ±30-45° angle, ±240-360°/s rate
- **Loose**: ±45-60° angle, ±360-600°/s rate

### Population Sizes
- **Quick testing**: 20-100
- **Development**: 100-500
- **Research**: 500-1000
- **Publication**: 1000-2000

### Training Frames (PPO)
- **Quick testing**: 10K-100K
- **Development**: 100K-500K
- **Research**: 500K-2M
- **Publication**: 2M-5M

## Tips for Configuration Selection

1. **Start with `quick_test.yaml`** for initial development
2. **Use `default.yaml`** for standard experiments
3. **Choose rocket-specific configs** based on your vehicle
4. **Use `safety_focused.yaml`** for critical applications
5. **Use `high_performance.yaml`** for final results
6. **Use `comparison.yaml`** for method comparisons

## Troubleshooting

### Common Issues
- **Training too slow**: Use `quick_test.yaml` or reduce population/frames
- **Poor performance**: Increase population size or training frames
- **Safety violations**: Use `safety_focused.yaml` or increase barrier coefficients
- **Instability**: Reduce gimbal angles or increase control penalties

### Debug Settings
- Set `save_models: false` for faster iteration
- Reduce `eval_episodes` for quick testing
- Use `development.yaml` for detailed logging
- Increase learning rates for faster convergence (less stable)

## Configuration Validation

The system automatically validates configurations and will report errors for:
- Invalid parameter ranges
- Missing required fields
- Incompatible parameter combinations
- File format errors

Use `python test_system.py --quick` to validate configurations before long training runs.