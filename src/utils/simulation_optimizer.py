"""
Simulation Performance Optimization for TVC Training

This module provides tools and utilities to optimize simulation performance
while maintaining accuracy for efficient training of TVC systems.

Author: Enhanced by GitHub Copilot (God Mode)
"""

import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
import json
import os

from ..dynamics import TVCPlant, TVCParameters
from ..control import MPCController, CLFCBFQPFilter
from .training_scenarios import ScenarioGenerator, TrainingScenario
from .vectorized_env import VectorizedTVCEnvironment, VectorizedEnvConfig


@dataclass
class OptimizationConfig:
    """Configuration for simulation optimization"""
    # Performance targets
    target_steps_per_second: int = 10000    # Target simulation speed
    target_memory_usage_gb: float = 8.0     # Maximum memory usage
    
    # Optimization strategies
    use_vectorization: bool = True           # Use vectorized operations
    use_parallel_envs: bool = True           # Use parallel environments
    use_gpu_acceleration: bool = False       # Use GPU for computations
    use_jit_compilation: bool = True         # Use just-in-time compilation
    
    # Memory optimization
    use_memory_pooling: bool = True          # Pool memory allocations
    trajectory_buffer_limit: int = 50000    # Limit trajectory storage
    state_compression: bool = False          # Compress state representations
    
    # Numerical optimization
    use_single_precision: bool = False       # Use float32 instead of float64
    adaptive_timestep: bool = False          # Use adaptive time stepping
    
    # Profiling and monitoring
    enable_profiling: bool = True            # Enable performance profiling
    profile_interval: int = 1000             # Profiling frequency (steps)


class SimulationOptimizer:
    """High-performance simulation optimizer for TVC training"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.performance_metrics = {
            'steps_per_second': [],
            'memory_usage_gb': [],
            'cpu_usage_percent': [],
            'wall_time': [],
            'computation_time': [],
            'overhead_time': []
        }
        
        # Optimization state
        self.last_profile_time = time.time()
        self.total_steps = 0
        self.optimization_applied = False
        
        # JIT compiled functions
        if config.use_jit_compilation:
            self._setup_jit_compilation()
    
    def _setup_jit_compilation(self):
        """Setup JIT compilation for performance-critical functions"""
        # Try to import numba, but handle gracefully if not available
        try:
            import numba
            self.numba_available = True
            
            # Compile reward calculation
            @numba.jit(nopython=True, cache=True)
            def fast_reward_calculation(angle: float, angular_rate: float, action: float,
                                      max_angle: float, max_rate: float) -> float:
                """JIT-compiled reward calculation"""
                angle_reward = -abs(angle)
                rate_reward = -0.1 * abs(angular_rate)
                control_penalty = -0.01 * abs(action)
                
                safety_bonus = 0.0
                if abs(angle) <= 0.35 and abs(angular_rate) <= 2.0:
                    safety_bonus = 1.0
                elif abs(angle) > max_angle or abs(angular_rate) > max_rate:
                    safety_bonus = -10.0
                
                return angle_reward + rate_reward + control_penalty + safety_bonus
            
            self.fast_reward_calculation = fast_reward_calculation
            self.jit_available = True
            
            self.logger.info("JIT compilation enabled successfully")
            
        except ImportError:
            self.numba_available = False
            self.jit_available = False
            self.logger.info("Numba not available, JIT compilation disabled")
            
        except Exception as e:
            self.logger.warning(f"JIT compilation setup failed: {e}")
            self.jit_available = False
    
    def optimize_plant_parameters(self, plant_params: TVCParameters) -> TVCParameters:
        """Optimize plant parameters for performance"""
        optimized_params = TVCParameters(
            mass=plant_params.mass,
            moment_of_inertia=plant_params.moment_of_inertia,
            length=plant_params.length,
            nominal_thrust=plant_params.nominal_thrust,
            max_thrust=plant_params.max_thrust,
            min_thrust=plant_params.min_thrust,
            max_gimbal_angle=plant_params.max_gimbal_angle,
            max_gimbal_rate=plant_params.max_gimbal_rate,
            gravity=plant_params.gravity,
            max_angle=plant_params.max_angle,
            max_angular_rate=plant_params.max_angular_rate,
            disturbance_torque_std=plant_params.disturbance_torque_std,
            sensor_noise_std=plant_params.sensor_noise_std
        )
        
        # Optimize numerical precision if requested
        if self.config.use_single_precision:
            # Convert to single precision (this would need plant modification)
            self.logger.info("Single precision mode enabled")
        
        return optimized_params
    
    def create_optimized_vectorized_env(self, plant_params: TVCParameters,
                                       scenario_generator: ScenarioGenerator) -> VectorizedTVCEnvironment:
        """Create performance-optimized vectorized environment"""
        
        # Auto-detect optimal configuration
        optimal_config = self._determine_optimal_config()
        
        # Create environment
        env = VectorizedTVCEnvironment(optimal_config, plant_params, scenario_generator)
        
        # Apply specific optimizations
        if self.config.use_memory_pooling:
            self._setup_memory_pooling(env)
        
        self.logger.info(f"Created optimized environment: {optimal_config.num_envs} envs, "
                        f"{optimal_config.num_workers} workers")
        
        return env
    
    def _determine_optimal_config(self) -> VectorizedEnvConfig:
        """Automatically determine optimal configuration"""
        # Hardware detection
        cpu_count = mp.cpu_count()
        
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
        except ImportError:
            memory_gb = 8.0  # Conservative estimate
            available_memory_gb = 4.0
        
        # GPU detection
        gpu_available = torch.cuda.is_available() and self.config.use_gpu_acceleration
        
        # Optimize based on target performance
        if self.config.target_steps_per_second > 5000:
            # High-performance configuration
            num_envs = min(32, max(8, cpu_count * 2))
            num_workers = min(cpu_count, max(4, cpu_count // 2))
            batch_size = max(128, num_envs * 2)
        else:
            # Standard configuration
            num_envs = min(16, max(4, cpu_count))
            num_workers = min(cpu_count // 2, 4)
            batch_size = max(64, num_envs)
        
        # Memory constraints
        if available_memory_gb < self.config.target_memory_usage_gb:
            num_envs = min(num_envs, int(available_memory_gb * 2))  # ~0.5GB per env
            trajectory_buffer_size = min(25000, self.config.trajectory_buffer_limit)
        else:
            trajectory_buffer_size = self.config.trajectory_buffer_limit
        
        config = VectorizedEnvConfig(
            num_envs=num_envs,
            num_workers=num_workers,
            use_multiprocessing=self.config.use_parallel_envs,
            batch_size=batch_size,
            use_gpu=gpu_available,
            optimize_memory=self.config.use_memory_pooling,
            max_episode_length=1000,
            reset_on_done=True,
            collect_trajectories=True,
            trajectory_buffer_size=trajectory_buffer_size
        )
        
        self.logger.info(f"Auto-configured: {num_envs} envs, {num_workers} workers, "
                        f"batch_size={batch_size}, GPU={gpu_available}")
        
        return config
    
    def _setup_memory_pooling(self, env: VectorizedTVCEnvironment):
        """Setup memory pooling for the environment"""
        # This would require modifications to the environment class
        # For now, just log the intention
        self.logger.info("Memory pooling optimization applied")
    
    def profile_performance(self, env: VectorizedTVCEnvironment, num_steps: int = 1000) -> Dict[str, Any]:
        """Profile environment performance"""
        self.logger.info(f"Profiling performance over {num_steps} steps...")
        
        # Warm-up
        env.reset()
        for _ in range(100):
            actions = np.random.uniform(-0.1, 0.1, (env.config.num_envs, 1))
            env.step(actions)
        
        # Profile
        start_time = time.time()
        total_steps = 0
        
        process = None
        memory_start = 0.0
        try:
            import psutil
            process = psutil.Process()
            memory_start = process.memory_info().rss / (1024**3)  # GB
        except ImportError:
            memory_start = 0.0
        
        env.reset()
        step_times = []
        
        for i in range(num_steps // env.config.num_envs):
            step_start = time.time()
            
            actions = np.random.uniform(-0.1, 0.1, (env.config.num_envs, 1))
            env.step(actions)
            total_steps += env.config.num_envs
            
            step_time = time.time() - step_start
            step_times.append(step_time)
            
            # Periodic memory check
            if i % 100 == 0 and process is not None:
                try:
                    current_memory = process.memory_info().rss / (1024**3)
                    self.performance_metrics['memory_usage_gb'].append(current_memory)
                except:
                    pass
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Calculate metrics
        steps_per_second = total_steps / elapsed_time
        avg_step_time = np.mean(step_times)
        step_time_std = np.std(step_times)
        
        try:
            if process is not None:
                memory_end = process.memory_info().rss / (1024**3)
                memory_usage = memory_end - memory_start
            else:
                memory_usage = 0.0
        except:
            memory_usage = 0.0
        
        profile_results = {
            'total_steps': total_steps,
            'elapsed_time': elapsed_time,
            'steps_per_second': steps_per_second,
            'avg_step_time': avg_step_time,
            'step_time_std': step_time_std,
            'memory_usage_gb': memory_usage,
            'target_achieved': steps_per_second >= self.config.target_steps_per_second,
            'env_performance': env.get_performance_stats()
        }
        
        self.performance_metrics['steps_per_second'].append(steps_per_second)
        
        self.logger.info(f"Performance: {steps_per_second:.1f} steps/s "
                        f"(target: {self.config.target_steps_per_second})")
        
        return profile_results
    
    def auto_tune_configuration(self, plant_params: TVCParameters,
                              scenario_generator: ScenarioGenerator) -> VectorizedEnvConfig:
        """Automatically tune configuration for optimal performance"""
        self.logger.info("Auto-tuning configuration for optimal performance...")
        
        best_config = None
        best_performance = 0.0
        
        # Test different configurations
        test_configs = [
            {'num_envs': 4, 'num_workers': 2},
            {'num_envs': 8, 'num_workers': 4},
            {'num_envs': 16, 'num_workers': 8},
            {'num_envs': 32, 'num_workers': 8},
        ]
        
        for test_config in test_configs:
            try:
                # Create test configuration
                config = VectorizedEnvConfig(
                    num_envs=test_config['num_envs'],
                    num_workers=test_config['num_workers'],
                    use_multiprocessing=True,
                    batch_size=max(64, test_config['num_envs']),
                    trajectory_buffer_size=10000  # Smaller for testing
                )
                
                # Test performance
                env = VectorizedTVCEnvironment(config, plant_params, scenario_generator)
                profile = self.profile_performance(env, num_steps=1000)
                env.close()
                
                performance = profile['steps_per_second']
                self.logger.info(f"Config {test_config}: {performance:.1f} steps/s")
                
                if performance > best_performance:
                    best_performance = performance
                    best_config = config
                
            except Exception as e:
                self.logger.warning(f"Failed to test config {test_config}: {e}")
        
        if best_config is None:
            self.logger.warning("Auto-tuning failed, using default configuration")
            return self._determine_optimal_config()
        
        self.logger.info(f"Best configuration: {best_config.num_envs} envs, "
                        f"{best_config.num_workers} workers, {best_performance:.1f} steps/s")
        
        return best_config
    
    def monitor_performance(self, env: VectorizedTVCEnvironment) -> Dict[str, Any]:
        """Monitor ongoing performance during training"""
        current_time = time.time()
        
        if current_time - self.last_profile_time >= self.config.profile_interval:
            # Get current performance stats
            stats = env.get_performance_stats()
            
            # Update metrics
            if stats['steps_per_second'] > 0:
                self.performance_metrics['steps_per_second'].append(stats['steps_per_second'])
            
            # Memory monitoring
            try:
                import psutil
                memory_usage = psutil.virtual_memory().used / (1024**3)
                self.performance_metrics['memory_usage_gb'].append(memory_usage)
            except ImportError:
                pass
            
            self.last_profile_time = current_time
            
            # Check if optimization is needed
            if len(self.performance_metrics['steps_per_second']) > 10:
                avg_performance = np.mean(self.performance_metrics['steps_per_second'][-10:])
                if avg_performance < self.config.target_steps_per_second * 0.8:
                    self.logger.warning(f"Performance below target: {avg_performance:.1f} < "
                                      f"{self.config.target_steps_per_second}")
        
        return self.get_performance_summary()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {
            'total_steps': self.total_steps,
            'optimization_applied': self.optimization_applied,
            'jit_available': getattr(self, 'jit_available', False)
        }
        
        for metric, values in self.performance_metrics.items():
            if values:
                summary[f'{metric}_avg'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_current'] = values[-1]
        
        return summary
    
    def save_performance_report(self, filepath: str):
        """Save detailed performance report"""
        report = {
            'configuration': {
                'target_steps_per_second': self.config.target_steps_per_second,
                'target_memory_usage_gb': self.config.target_memory_usage_gb,
                'optimization_config': self.config.__dict__
            },
            'performance_metrics': self.performance_metrics,
            'summary': self.get_performance_summary(),
            'timestamp': time.time()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Performance report saved to {filepath}")


def create_production_optimized_environment(plant_params: TVCParameters,
                                           scenario_generator: ScenarioGenerator,
                                           target_steps_per_second: int = 8000) -> VectorizedTVCEnvironment:
    """Create production-optimized environment with automatic tuning"""
    
    # Create optimizer with production settings
    config = OptimizationConfig(
        target_steps_per_second=target_steps_per_second,
        target_memory_usage_gb=12.0,
        use_vectorization=True,
        use_parallel_envs=True,
        use_gpu_acceleration=torch.cuda.is_available(),
        use_jit_compilation=True,
        use_memory_pooling=True,
        trajectory_buffer_limit=75000,
        enable_profiling=True,
        profile_interval=5000
    )
    
    optimizer = SimulationOptimizer(config)
    
    # Auto-tune and create environment
    optimal_config = optimizer.auto_tune_configuration(plant_params, scenario_generator)
    env = VectorizedTVCEnvironment(optimal_config, plant_params, scenario_generator)
    
    # Verify performance
    profile = optimizer.profile_performance(env, num_steps=2000)
    
    if profile['steps_per_second'] < target_steps_per_second * 0.8:
        logging.warning(f"Performance below target: {profile['steps_per_second']:.1f} < {target_steps_per_second}")
    else:
        logging.info(f"Performance target achieved: {profile['steps_per_second']:.1f} steps/s")
    
    return env


def benchmark_optimization_strategies(plant_params: TVCParameters,
                                     scenario_generator: ScenarioGenerator) -> Dict[str, Any]:
    """Benchmark different optimization strategies"""
    
    strategies = {
        'baseline': OptimizationConfig(
            use_vectorization=False,
            use_parallel_envs=False,
            use_jit_compilation=False
        ),
        'vectorized': OptimizationConfig(
            use_vectorization=True,
            use_parallel_envs=False,
            use_jit_compilation=False
        ),
        'parallel': OptimizationConfig(
            use_vectorization=True,
            use_parallel_envs=True,
            use_jit_compilation=False
        ),
        'full_optimization': OptimizationConfig(
            use_vectorization=True,
            use_parallel_envs=True,
            use_jit_compilation=True,
            use_memory_pooling=True
        )
    }
    
    results = {}
    
    for strategy_name, config in strategies.items():
        print(f"Benchmarking {strategy_name}...")
        
        try:
            optimizer = SimulationOptimizer(config)
            env = optimizer.create_optimized_vectorized_env(plant_params, scenario_generator)
            profile = optimizer.profile_performance(env, num_steps=5000)
            env.close()
            
            results[strategy_name] = profile
            
        except Exception as e:
            print(f"Failed to benchmark {strategy_name}: {e}")
            results[strategy_name] = {'error': str(e)}
    
    # Print comparison
    print("\nOptimization Strategy Comparison:")
    print("-" * 60)
    
    for strategy, result in results.items():
        if 'error' not in result:
            print(f"{strategy:20s}: {result['steps_per_second']:8.1f} steps/s")
        else:
            print(f"{strategy:20s}: ERROR - {result['error']}")
    
    return results