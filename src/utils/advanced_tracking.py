"""
Advanced Tracking and Monitoring for TVC Training

This module provides comprehensive experiment tracking, hyperparameter optimization,
and monitoring capabilities for TVC rocket control training systems.

Author: Enhanced by GitHub Copilot (God Mode)
"""

import numpy as np
import torch
import time
import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
import datetime

# Optional imports for advanced features
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

from ..dynamics import TVCPlant, TVCParameters
from ..control import MPCController, CLFCBFQPFilter
from .training_scenarios import ScenarioGenerator, TrainingScenario
from .curriculum_learning import CurriculumManager
from .comprehensive_evaluation import ComprehensiveEvaluator
from .vectorized_env import VectorizedTVCEnvironment


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking"""
    # Project settings
    project_name: str = "tvc_rocket_control"
    experiment_name: str = "default_experiment"
    run_name: Optional[str] = None
    tags: Optional[List[str]] = None
    
    # Tracking settings
    log_frequency: int = 100        # Steps between logs
    save_frequency: int = 1000      # Steps between model saves
    eval_frequency: int = 5000      # Steps between evaluations
    
    # Monitoring
    track_gradients: bool = True    # Track gradient norms
    track_weights: bool = False     # Track weight distributions
    track_performance: bool = True  # Track performance metrics
    
    # Hyperparameter optimization
    enable_hpo: bool = False        # Enable hyperparameter optimization
    hpo_trials: int = 100          # Number of HPO trials
    hpo_timeout: int = 3600        # HPO timeout in seconds


class AdvancedTracker:
    """Advanced experiment tracking and monitoring system"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize tracking backend
        self.wandb_run = None
        self.local_logs = []
        self.experiment_start_time = time.time()
        
        # Performance tracking
        self.step_count = 0
        self.episode_count = 0
        self.last_log_time = time.time()
        
        # Initialize WandB if available
        if WANDB_AVAILABLE and config.project_name:
            self._init_wandb()
        else:
            self.logger.warning("WandB not available, using local logging only")
        
        # Create local log directory
        self.log_dir = f"./logs/{config.experiment_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.log_dir, exist_ok=True)
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking"""
        if not WANDB_AVAILABLE or wandb is None:
            self.logger.warning("WandB not available, skipping initialization")
            return
            
        try:
            self.wandb_run = wandb.init(
                project=self.config.project_name,
                name=self.config.run_name or self.config.experiment_name,
                tags=self.config.tags or [],
                config=asdict(self.config)
            )
            self.logger.info(f"WandB initialized: {self.wandb_run.url}")
        except Exception as e:
            self.logger.error(f"Failed to initialize WandB: {e}")
            self.wandb_run = None
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        if self.wandb_run and WANDB_AVAILABLE and wandb is not None:
            wandb.config.update(params)
        
        # Local logging
        param_file = os.path.join(self.log_dir, "hyperparameters.json")
        with open(param_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        self.logger.info(f"Logged hyperparameters: {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """Log training metrics"""
        if step is None:
            step = self.step_count
        
        # Add timestamp
        metrics['timestamp'] = time.time()
        metrics['wall_time'] = time.time() - self.experiment_start_time
        
        # WandB logging
        if self.wandb_run and WANDB_AVAILABLE and wandb is not None:
            wandb.log(metrics, step=step)
        
        # Local logging
        log_entry = {'step': step, 'metrics': metrics}
        self.local_logs.append(log_entry)
        
        # Periodic local save
        if len(self.local_logs) % 100 == 0:
            self._save_local_logs()
        
        self.step_count = max(self.step_count, step + 1)
    
    def log_episode_summary(self, episode_data: Dict[str, Any]):
        """Log episode summary statistics"""
        episode_metrics = {
            f"episode/{key}": value for key, value in episode_data.items()
        }
        episode_metrics['episode_count'] = self.episode_count
        
        self.log_metrics(episode_metrics)
        self.episode_count += 1
    
    def log_model_performance(self, performance_data: Dict[str, Any]):
        """Log model performance metrics"""
        perf_metrics = {
            f"performance/{key}": value for key, value in performance_data.items()
        }
        self.log_metrics(perf_metrics)
    
    def log_gradients(self, model: torch.nn.Module):
        """Log gradient statistics"""
        if not self.config.track_gradients:
            return
        
        total_norm = 0.0
        grad_metrics = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.norm().item()
                total_norm += param_norm ** 2
                grad_metrics[f"gradients/{name}_norm"] = param_norm
        
        total_norm = total_norm ** 0.5
        grad_metrics['gradients/total_norm'] = total_norm
        
        self.log_metrics(grad_metrics)
    
    def log_weights(self, model: torch.nn.Module):
        """Log weight statistics"""
        if not self.config.track_weights:
            return
        
        weight_metrics = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                weight_metrics[f"weights/{name}_mean"] = param.data.mean().item()
                weight_metrics[f"weights/{name}_std"] = param.data.std().item()
                weight_metrics[f"weights/{name}_norm"] = param.data.norm().item()
        
        self.log_metrics(weight_metrics)
    
    def log_scenario_performance(self, scenario: TrainingScenario, performance: Dict[str, Any]):
        """Log performance on specific training scenario"""
        scenario_metrics = {
            f"scenario/{scenario.name}/{key}": value 
            for key, value in performance.items()
        }
        scenario_metrics['scenario/current'] = scenario.name
        scenario_metrics['scenario/difficulty'] = scenario.difficulty_weight
        
        self.log_metrics(scenario_metrics)
    
    def save_model_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                            step: int, metrics: Dict[str, Any]):
        """Save model checkpoint with tracking"""
        checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pth")
        
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'metrics': metrics,
            'timestamp': time.time()
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Log to WandB as artifact
        if self.wandb_run and WANDB_AVAILABLE and wandb is not None:
            artifact = wandb.Artifact(f"model_checkpoint_step_{step}", type="model")
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)
        
        self.logger.info(f"Saved checkpoint at step {step}: {checkpoint_path}")
    
    def log_video(self, video_path: str, caption: str = "Training Video"):
        """Log video to tracking system"""
        if self.wandb_run and WANDB_AVAILABLE and wandb is not None and os.path.exists(video_path):
            wandb.log({"videos/training": wandb.Video(video_path, caption=caption)})
    
    def _save_local_logs(self):
        """Save local logs to file"""
        log_file = os.path.join(self.log_dir, "training_logs.jsonl")
        
        with open(log_file, 'a') as f:
            for log_entry in self.local_logs[-100:]:  # Save last 100 entries
                f.write(json.dumps(log_entry) + '\n')
    
    def finish(self):
        """Finalize tracking and cleanup"""
        # Save final logs
        self._save_local_logs()
        
        # Finish WandB run
        if self.wandb_run and WANDB_AVAILABLE and wandb is not None:
            wandb.finish()
        
        # Create experiment summary
        summary = {
            'experiment_name': self.config.experiment_name,
            'total_steps': self.step_count,
            'total_episodes': self.episode_count,
            'total_time': time.time() - self.experiment_start_time,
            'log_dir': self.log_dir
        }
        
        summary_file = os.path.join(self.log_dir, "experiment_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Experiment tracking completed: {summary}")


class HyperparameterOptimizer:
    """Automated hyperparameter optimization using Optuna"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if not OPTUNA_AVAILABLE:
            self.logger.error("Optuna not available for hyperparameter optimization")
            self.enabled = False
        else:
            self.enabled = config.enable_hpo
            self.study = None
    
    def create_study(self, study_name: str, direction: str = "maximize"):
        """Create Optuna study for optimization"""
        if not self.enabled or not OPTUNA_AVAILABLE or optuna is None:
            return None
        
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            pruner=optuna.pruners.MedianPruner()
        )
        
        self.logger.info(f"Created HPO study: {study_name}")
        return self.study
    
    def suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial"""
        if not self.enabled:
            return {}
        
        # PPO hyperparameters
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'clip_ratio': trial.suggest_float('clip_ratio', 0.1, 0.3),
            'entropy_coef': trial.suggest_float('entropy_coef', 0.0, 0.1),
            'value_coef': trial.suggest_float('value_coef', 0.1, 1.0),
            'gamma': trial.suggest_float('gamma', 0.95, 0.999),
            
            # Network architecture
            'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256, 512]),
            'num_layers': trial.suggest_int('num_layers', 2, 4),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'elu']),
            
            # Training parameters
            'max_episodes': trial.suggest_int('max_episodes', 1000, 10000),
            'episode_length': trial.suggest_int('episode_length', 500, 2000),
            
            # Environment parameters
            'control_frequency': trial.suggest_categorical('control_frequency', [100, 200, 500]),
            'scenario_difficulty': trial.suggest_float('scenario_difficulty', 0.3, 0.8),
        }
        
        return params
    
    def optimize(self, objective_function: Callable, n_trials: Optional[int] = None) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        if not self.enabled or self.study is None:
            self.logger.warning("HPO not enabled or study not created")
            return {}
        
        n_trials = n_trials or self.config.hpo_trials
        
        self.logger.info(f"Starting HPO with {n_trials} trials...")
        
        self.study.optimize(
            objective_function,
            n_trials=n_trials,
            timeout=self.config.hpo_timeout
        )
        
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        self.logger.info(f"HPO completed. Best value: {best_value}")
        self.logger.info(f"Best parameters: {best_params}")
        
        return best_params
    
    def get_optimization_history(self) -> Dict[str, Any]:
        """Get optimization history and statistics"""
        if not self.enabled or self.study is None:
            return {}
        
        trials_df = self.study.trials_dataframe()
        
        return {
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'n_trials': len(self.study.trials),
            'optimization_history': trials_df.to_dict('records')
        }


class ComprehensiveMonitor:
    """Comprehensive monitoring system combining tracking and optimization"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tracker = AdvancedTracker(config)
        self.optimizer = HyperparameterOptimizer(config)
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self.last_eval_step = 0
        self.last_save_step = 0
        self.training_start_time = time.time()
        
        # Performance tracking
        self.episode_rewards = []
        self.success_rates = []
        self.scenario_performances = {}
    
    def start_experiment(self, experiment_config: Dict[str, Any]):
        """Start comprehensive experiment monitoring"""
        self.tracker.log_hyperparameters(experiment_config)
        
        self.logger.info(f"Started experiment: {self.config.experiment_name}")
        self.logger.info(f"Tracking enabled: WandB={WANDB_AVAILABLE}, HPO={self.optimizer.enabled}")
    
    def log_training_step(self, step: int, metrics: Dict[str, Any], 
                         model: Optional[torch.nn.Module] = None,
                         optimizer: Optional[torch.optim.Optimizer] = None):
        """Log comprehensive training step data"""
        
        # Basic metrics
        self.tracker.log_metrics(metrics, step)
        
        # Model-specific logging
        if model is not None:
            if step % (self.config.log_frequency * 5) == 0:
                self.tracker.log_gradients(model)
                if self.config.track_weights:
                    self.tracker.log_weights(model)
        
        # Periodic evaluation
        if step - self.last_eval_step >= self.config.eval_frequency:
            self._run_periodic_evaluation(step)
            self.last_eval_step = step
        
        # Periodic model saving
        if (step - self.last_save_step >= self.config.save_frequency and 
            model is not None and optimizer is not None):
            self.tracker.save_model_checkpoint(model, optimizer, step, metrics)
            self.last_save_step = step
    
    def log_episode_completion(self, episode_data: Dict[str, Any]):
        """Log episode completion with comprehensive analysis"""
        reward = episode_data.get('total_reward', 0.0)
        success = episode_data.get('success', False)
        scenario = episode_data.get('scenario', 'unknown')
        
        # Track episode performance
        self.episode_rewards.append(reward)
        self.success_rates.append(float(success))
        
        # Track scenario-specific performance
        if scenario not in self.scenario_performances:
            self.scenario_performances[scenario] = {'rewards': [], 'successes': []}
        
        self.scenario_performances[scenario]['rewards'].append(reward)
        self.scenario_performances[scenario]['successes'].append(float(success))
        
        # Calculate rolling statistics
        window_size = 100
        if len(self.episode_rewards) >= window_size:
            recent_rewards = self.episode_rewards[-window_size:]
            recent_successes = self.success_rates[-window_size:]
            
            rolling_stats = {
                'episode/reward_mean': np.mean(recent_rewards),
                'episode/reward_std': np.std(recent_rewards),
                'episode/success_rate': np.mean(recent_successes),
                'episode/total_episodes': len(self.episode_rewards)
            }
            
            self.tracker.log_metrics(rolling_stats)
        
        # Log individual episode
        self.tracker.log_episode_summary(episode_data)
    
    def _run_periodic_evaluation(self, step: int):
        """Run periodic evaluation and logging"""
        # This would integrate with the ComprehensiveEvaluator
        self.logger.info(f"Running periodic evaluation at step {step}")
        
        # Calculate training progress metrics
        training_time = time.time() - self.training_start_time
        steps_per_second = step / training_time if training_time > 0 else 0
        
        eval_metrics = {
            'evaluation/training_time': training_time,
            'evaluation/steps_per_second': steps_per_second,
            'evaluation/total_episodes': len(self.episode_rewards)
        }
        
        # Scenario performance summary
        for scenario, data in self.scenario_performances.items():
            if data['rewards']:
                eval_metrics[f'evaluation/scenario_{scenario}_mean_reward'] = np.mean(data['rewards'])
                eval_metrics[f'evaluation/scenario_{scenario}_success_rate'] = np.mean(data['successes'])
        
        self.tracker.log_metrics(eval_metrics, step)
    
    def create_hpo_objective(self, training_function: Callable) -> Callable:
        """Create objective function for hyperparameter optimization"""
        def objective(trial):
            # Get suggested hyperparameters
            params = self.optimizer.suggest_hyperparameters(trial)
            
            # Run training with these parameters
            result = training_function(params, trial)
            
            # Return metric to optimize (e.g., success rate, reward)
            return result.get('success_rate', 0.0)
        
        return objective
    
    def run_hyperparameter_optimization(self, training_function: Callable):
        """Run full hyperparameter optimization"""
        if not self.optimizer.enabled:
            self.logger.warning("Hyperparameter optimization not enabled")
            return {}
        
        # Create study
        study_name = f"{self.config.experiment_name}_hpo"
        self.optimizer.create_study(study_name, direction="maximize")
        
        # Create objective function
        objective = self.create_hpo_objective(training_function)
        
        # Run optimization
        best_params = self.optimizer.optimize(objective)
        
        # Log optimization results
        hpo_history = self.optimizer.get_optimization_history()
        self.tracker.log_hyperparameters(hpo_history)
        
        return best_params
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        report = {
            'experiment_config': asdict(self.config),
            'training_summary': {
                'total_episodes': len(self.episode_rewards),
                'total_training_time': time.time() - self.training_start_time,
                'final_success_rate': np.mean(self.success_rates[-100:]) if self.success_rates else 0.0,
                'final_mean_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
            },
            'scenario_performances': {}
        }
        
        # Scenario-specific performance
        for scenario, data in self.scenario_performances.items():
            if data['rewards']:
                report['scenario_performances'][scenario] = {
                    'mean_reward': np.mean(data['rewards']),
                    'success_rate': np.mean(data['successes']),
                    'num_episodes': len(data['rewards'])
                }
        
        # Save report
        report_file = os.path.join(self.tracker.log_dir, "final_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def finish_experiment(self):
        """Finish experiment with comprehensive cleanup"""
        # Generate final report
        final_report = self.generate_final_report()
        
        # Log final summary
        self.tracker.log_metrics({
            'final/total_episodes': len(self.episode_rewards),
            'final/mean_reward': final_report['training_summary']['final_mean_reward'],
            'final/success_rate': final_report['training_summary']['final_success_rate'],
            'final/training_time': final_report['training_summary']['total_training_time']
        })
        
        # Cleanup tracking
        self.tracker.finish()
        
        self.logger.info("Experiment monitoring completed successfully")
        return final_report


def create_production_monitor(experiment_name: str, 
                            enable_wandb: bool = True,
                            enable_hpo: bool = False) -> ComprehensiveMonitor:
    """Create production-ready monitoring system"""
    
    project_name = "tvc_rocket_control_production" if enable_wandb else None
    
    config = ExperimentConfig(
        project_name=project_name or "tvc_rocket_control_local",
        experiment_name=experiment_name,
        run_name=f"{experiment_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags=["production", "tvc", "rocket_control"],
        log_frequency=50,
        save_frequency=5000,
        eval_frequency=2500,
        track_gradients=True,
        track_weights=False,
        track_performance=True,
        enable_hpo=enable_hpo,
        hpo_trials=50,
        hpo_timeout=7200  # 2 hours
    )
    
    monitor = ComprehensiveMonitor(config)
    
    logging.info(f"Created production monitor: {experiment_name}")
    logging.info(f"WandB: {enable_wandb and WANDB_AVAILABLE}, HPO: {enable_hpo and OPTUNA_AVAILABLE}")
    
    return monitor