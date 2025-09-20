"""
Vectorized Environment Implementation for High-Performance TVC Training

This module implements vectorized environments using parallel processing
for significantly faster training while maintaining simulation accuracy.

Author: Enhanced by GitHub Copilot (God Mode)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
import logging

from ..dynamics import TVCPlant, TVCParameters
from ..control import MPCController, CLFCBFQPFilter
from .training_scenarios import ScenarioGenerator, TrainingScenario


@dataclass
class VectorizedEnvConfig:
    """Configuration for vectorized environments"""
    # Parallelization
    num_envs: int = 8                    # Number of parallel environments
    num_workers: int = 4                 # Number of worker processes
    use_multiprocessing: bool = True     # Use multiprocessing vs threading
    
    # Performance optimization
    batch_size: int = 64                 # Batch size for vectorized operations
    use_gpu: bool = False                # Use GPU acceleration (if available)
    optimize_memory: bool = True         # Memory optimization
    
    # Simulation parameters
    max_episode_length: int = 1000       # Maximum episode length
    reset_on_done: bool = True           # Auto-reset environments
    
    # Data collection
    collect_trajectories: bool = True    # Collect full trajectories
    trajectory_buffer_size: int = 10000  # Maximum trajectory buffer size


class VectorizedTVCEnvironment:
    """Vectorized TVC environment for parallel training"""
    
    def __init__(self, config: VectorizedEnvConfig, plant_params: TVCParameters,
                 scenario_generator: ScenarioGenerator):
        self.config = config
        self.plant_params = plant_params
        self.scenario_generator = scenario_generator
        
        # Initialize environments
        self.plants = [TVCPlant(plant_params) for _ in range(config.num_envs)]
        self.current_scenarios: List[Optional[TrainingScenario]] = [None] * config.num_envs
        self.episode_steps = np.zeros(config.num_envs, dtype=int)
        self.episode_rewards = np.zeros(config.num_envs)
        self.done_flags = np.zeros(config.num_envs, dtype=bool)
        
        # State tracking
        self.states = np.zeros((config.num_envs, 2))  # [angle, angular_rate]
        self.actions = np.zeros((config.num_envs, 1))  # [gimbal_angle]
        self.rewards = np.zeros(config.num_envs)
        
        # Trajectory collection
        if config.collect_trajectories:
            self.trajectory_buffer = []
            self.current_trajectories = [[] for _ in range(config.num_envs)]
        
        # Performance tracking
        self.step_times = []
        self.reset_times = []
        
        # Setup parallel processing
        if config.use_multiprocessing:
            self.executor = ProcessPoolExecutor(max_workers=config.num_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=config.num_workers)
        
        self.logger = logging.getLogger(__name__)
    
    def reset(self, env_indices: Optional[List[int]] = None) -> np.ndarray:
        """Reset environments (vectorized)"""
        start_time = time.time()
        
        if env_indices is None:
            env_indices = list(range(self.config.num_envs))
        
        # Generate scenarios and initial states for environments
        reset_data = []
        for i in env_indices:
            scenario = self.scenario_generator.scenarios[i % len(self.scenario_generator.scenarios)]
            initial_state = self.scenario_generator.generate_initial_state(scenario)
            reset_data.append((i, scenario, initial_state))
        
        # Parallel reset
        if len(reset_data) > 1 and self.config.use_multiprocessing:
            futures = []
            for i, scenario, initial_state in reset_data:
                future = self.executor.submit(self._reset_single_env, i, scenario, initial_state)
                futures.append((i, future))
            
            # Collect results
            for i, future in futures:
                state = future.result()
                self.states[i] = state
                self.episode_steps[i] = 0
                self.episode_rewards[i] = 0.0
                self.done_flags[i] = False
        else:
            # Sequential reset for small numbers of environments
            for i, scenario, initial_state in reset_data:
                state = self._reset_single_env(i, scenario, initial_state)
                self.states[i] = state
                self.episode_steps[i] = 0
                self.episode_rewards[i] = 0.0
                self.done_flags[i] = False
        
        reset_time = time.time() - start_time
        self.reset_times.append(reset_time)
        
        return self.states.copy()
    
    def _reset_single_env(self, env_idx: int, scenario: TrainingScenario, 
                         initial_state: Dict[str, Any]) -> np.ndarray:
        """Reset a single environment"""
        # Set up plant
        plant_state = np.array([initial_state['angle'], initial_state['angular_rate']])
        self.plants[env_idx].reset(plant_state)
        
        # Store scenario
        self.current_scenarios[env_idx] = scenario
        
        # Initialize trajectory if collecting
        if self.config.collect_trajectories:
            self.current_trajectories[env_idx] = []
        
        return plant_state
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments (vectorized)"""
        start_time = time.time()
        
        # Ensure actions are properly shaped
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)
        
        # Prepare step data
        step_data = []
        for i in range(self.config.num_envs):
            if not self.done_flags[i]:
                step_data.append((i, actions[i, 0]))
        
        # Parallel stepping
        if len(step_data) > 1 and self.config.use_multiprocessing:
            futures = []
            for i, action in step_data:
                future = self.executor.submit(self._step_single_env, i, action)
                futures.append((i, future))
            
            # Collect results
            infos = [{}] * self.config.num_envs
            for i, future in futures:
                next_state, reward, done, info = future.result()
                self.states[i] = next_state
                self.rewards[i] = reward
                self.done_flags[i] = done
                infos[i] = info
                self.episode_steps[i] += 1
                self.episode_rewards[i] += reward
        else:
            # Sequential stepping
            infos = [{}] * self.config.num_envs
            for i, action in step_data:
                next_state, reward, done, info = self._step_single_env(i, action)
                self.states[i] = next_state
                self.rewards[i] = reward
                self.done_flags[i] = done
                infos[i] = info
                self.episode_steps[i] += 1
                self.episode_rewards[i] += reward
        
        # Handle episode termination
        if self.config.reset_on_done:
            done_envs = np.where(self.done_flags)[0].tolist()
            if done_envs:
                # Store completed trajectories
                if self.config.collect_trajectories:
                    for env_idx in done_envs:
                        if len(self.trajectory_buffer) < self.config.trajectory_buffer_size:
                            self.trajectory_buffer.append(self.current_trajectories[env_idx])
                        else:
                            # Replace oldest trajectory
                            self.trajectory_buffer[len(self.trajectory_buffer) % self.config.trajectory_buffer_size] = \
                                self.current_trajectories[env_idx]
                
                # Reset done environments
                self.reset(done_envs)
        
        step_time = time.time() - start_time
        self.step_times.append(step_time)
        
        return self.states.copy(), self.rewards.copy(), self.done_flags.copy(), infos
    
    def _step_single_env(self, env_idx: int, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step a single environment"""
        plant = self.plants[env_idx]
        scenario = self.current_scenarios[env_idx]
        
        # Clip action to safe range
        action = np.clip(action, -self.plant_params.max_gimbal_angle, 
                        self.plant_params.max_gimbal_angle)
        
        # Step plant
        next_state = plant.step(action, dt=0.005, add_disturbance=True)
        
        # Calculate reward
        reward = self._calculate_reward(next_state, action, scenario) if scenario else 0.0
        
        # Check termination
        done = (self.episode_steps[env_idx] >= self.config.max_episode_length or
               not plant.is_safe())
        
        # Collect trajectory data
        if self.config.collect_trajectories:
            step_data = {
                'state': next_state.copy(),
                'action': action,
                'reward': reward,
                'time': self.episode_steps[env_idx] * 0.005
            }
            self.current_trajectories[env_idx].append(step_data)
        
        # Info dictionary
        info = {
            'scenario': scenario.name if scenario else 'unknown',
            'episode_step': self.episode_steps[env_idx],
            'episode_reward': self.episode_rewards[env_idx],
            'safety_violation': not plant.is_safe()
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, state: np.ndarray, action: float, 
                         scenario: Optional[TrainingScenario]) -> float:
        """Calculate reward for current state and action"""
        angle, angular_rate = state
        
        # Primary objective: minimize angle error
        angle_reward = -np.abs(angle)
        
        # Secondary objective: minimize angular rate
        rate_reward = -0.1 * np.abs(angular_rate)
        
        # Control effort penalty
        control_penalty = -0.01 * np.abs(action)
        
        # Safety bonus/penalty
        safety_bonus = 0.0
        max_stable_angle = 0.35 if scenario else 0.35  # Default 20 degrees
        max_stable_rate = 2.0 if scenario else 2.0    # Default rate limit
        
        if scenario:
            max_stable_angle = scenario.max_stable_angle if hasattr(scenario, 'max_stable_angle') else 0.35
            max_stable_rate = scenario.max_stable_rate if hasattr(scenario, 'max_stable_rate') else 2.0
        
        if np.abs(angle) <= max_stable_angle and np.abs(angular_rate) <= max_stable_rate:
            safety_bonus = 1.0
        elif np.abs(angle) > self.plant_params.max_angle or np.abs(angular_rate) > self.plant_params.max_angular_rate:
            safety_bonus = -10.0  # Large penalty for safety violations
        
        # Scenario-specific rewards
        scenario_bonus = 0.0
        if scenario and hasattr(scenario, 'scenario_type'):
            if scenario.scenario_type.value == 'precision_landing':
                # Extra reward for precision
                if np.abs(angle) <= 0.02:  # 1 degree precision
                    scenario_bonus = 2.0
            elif scenario.scenario_type.value == 'high_dynamics':
                # Reward for handling high rates
                if np.abs(angular_rate) > 1.0 and np.abs(angle) <= max_stable_angle:
                    scenario_bonus = 0.5
        
        total_reward = angle_reward + rate_reward + control_penalty + safety_bonus + scenario_bonus
        
        return total_reward
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            'avg_step_time': np.mean(self.step_times) if self.step_times else 0.0,
            'avg_reset_time': np.mean(self.reset_times) if self.reset_times else 0.0,
            'steps_per_second': 1.0 / np.mean(self.step_times) if self.step_times else 0.0,
            'total_steps': len(self.step_times),
            'total_resets': len(self.reset_times),
            'num_trajectories': len(self.trajectory_buffer) if self.config.collect_trajectories else 0
        }
        
        if self.step_times:
            stats['step_time_std'] = np.std(self.step_times)
            stats['min_step_time'] = np.min(self.step_times)
            stats['max_step_time'] = np.max(self.step_times)
        
        return stats
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


class ParallelTrainingAccelerator:
    """High-performance training accelerator for TVC systems"""
    
    def __init__(self, num_envs: int = 16, num_workers: int = 8):
        self.num_envs = num_envs
        self.num_workers = num_workers
        
        # Performance optimization settings
        self.config = VectorizedEnvConfig(
            num_envs=num_envs,
            num_workers=num_workers,
            use_multiprocessing=True,
            batch_size=max(64, num_envs),
            use_gpu=torch.cuda.is_available(),
            optimize_memory=True,
            max_episode_length=1000,
            reset_on_done=True,
            collect_trajectories=True,
            trajectory_buffer_size=50000
        )
        
        self.logger = logging.getLogger(__name__)
    
    def create_vectorized_env(self, plant_params: TVCParameters,
                            scenario_generator: ScenarioGenerator) -> VectorizedTVCEnvironment:
        """Create optimized vectorized environment"""
        env = VectorizedTVCEnvironment(self.config, plant_params, scenario_generator)
        
        self.logger.info(f"Created vectorized environment with {self.num_envs} parallel environments")
        self.logger.info(f"Using {self.num_workers} workers, GPU: {self.config.use_gpu}")
        
        return env
    
    def optimize_for_hardware(self):
        """Optimize configuration for current hardware"""
        # CPU optimization
        cpu_count = mp.cpu_count()
        if self.num_workers > cpu_count:
            self.logger.warning(f"Reducing workers from {self.num_workers} to {cpu_count} (CPU cores)")
            self.config.num_workers = cpu_count
        
        # Memory optimization
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            
            if available_memory < 4:
                self.logger.warning("Low memory detected, reducing environment count and buffer sizes")
                self.config.num_envs = min(self.config.num_envs, 8)
                self.config.trajectory_buffer_size = min(self.config.trajectory_buffer_size, 10000)
            elif available_memory > 16:
                self.logger.info("High memory available, increasing buffer sizes")
                self.config.trajectory_buffer_size = min(self.config.trajectory_buffer_size * 2, 100000)
        
        except ImportError:
            self.logger.info("psutil not available, using default memory settings")
        
        # GPU optimization
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            self.logger.info(f"GPU detected with {gpu_memory:.1f}GB memory")
            
            if gpu_memory < 4:
                self.logger.warning("Limited GPU memory, using CPU for environment simulation")
                self.config.use_gpu = False
        
        self.logger.info(f"Optimized configuration: {self.config.num_envs} envs, {self.config.num_workers} workers")


def create_high_performance_training_setup(plant_params: TVCParameters,
                                          scenario_generator: ScenarioGenerator,
                                          num_envs: int = 16) -> VectorizedTVCEnvironment:
    """Create high-performance training setup with automatic optimization"""
    
    # Auto-detect optimal configuration
    cpu_count = mp.cpu_count()
    optimal_workers = min(cpu_count, max(4, cpu_count // 2))
    
    accelerator = ParallelTrainingAccelerator(num_envs=num_envs, num_workers=optimal_workers)
    accelerator.optimize_for_hardware()
    
    return accelerator.create_vectorized_env(plant_params, scenario_generator)


def benchmark_performance(plant_params: TVCParameters, scenario_generator: ScenarioGenerator,
                         num_steps: int = 10000) -> Dict[str, Any]:
    """Benchmark vectorized environment performance"""
    
    print("Benchmarking vectorized environment performance...")
    
    # Test different configurations
    configs = [
        (4, 2),   # 4 envs, 2 workers
        (8, 4),   # 8 envs, 4 workers
        (16, 8),  # 16 envs, 8 workers
    ]
    
    results = {}
    
    for num_envs, num_workers in configs:
        print(f"Testing {num_envs} environments with {num_workers} workers...")
        
        accelerator = ParallelTrainingAccelerator(num_envs=num_envs, num_workers=num_workers)
        env = accelerator.create_vectorized_env(plant_params, scenario_generator)
        
        # Warm up
        env.reset()
        for _ in range(100):
            actions = np.random.uniform(-0.1, 0.1, (num_envs, 1))
            env.step(actions)
        
        # Benchmark
        start_time = time.time()
        total_steps = 0
        
        env.reset()
        for _ in range(num_steps // num_envs):
            actions = np.random.uniform(-0.1, 0.1, (num_envs, 1))
            env.step(actions)
            total_steps += num_envs
        
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        steps_per_second = total_steps / elapsed_time
        
        stats = env.get_performance_stats()
        
        results[f"{num_envs}envs_{num_workers}workers"] = {
            'total_steps': total_steps,
            'elapsed_time': elapsed_time,
            'steps_per_second': steps_per_second,
            'performance_stats': stats
        }
        
        env.close()
        
        print(f"  Steps per second: {steps_per_second:.1f}")
    
    # Find best configuration
    best_config = max(results.keys(), key=lambda k: results[k]['steps_per_second'])
    print(f"\nBest configuration: {best_config}")
    print(f"Best performance: {results[best_config]['steps_per_second']:.1f} steps/second")
    
    return results