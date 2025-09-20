"""
Curriculum Learning Implementation for TVC Training

This module implements progressive difficulty training using the comprehensive
scenario system to gradually increase training complexity.

Author: Enhanced by GitHub Copilot (God Mode)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import json

from .training_scenarios import ScenarioGenerator, TrainingScenario, ScenarioType


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning"""
    # Progression parameters
    initial_success_threshold: float = 0.8      # Success rate to advance
    min_episodes_per_stage: int = 500           # Minimum episodes before advance
    max_episodes_per_stage: int = 2000          # Maximum episodes per stage
    success_window: int = 100                   # Episodes to evaluate success rate
    
    # Difficulty progression
    difficulty_increment: float = 0.1           # How much to increase difficulty
    max_difficulty: float = 1.0                 # Maximum difficulty level
    
    # Adaptive parameters
    adaptive_thresholds: bool = True            # Adapt thresholds based on performance
    threshold_decay: float = 0.95               # Reduce threshold over time
    min_threshold: float = 0.6                  # Minimum success threshold
    
    # Rollback settings
    enable_rollback: bool = True                # Roll back if performance drops
    rollback_threshold: float = 0.5             # Performance drop threshold
    rollback_episodes: int = 200                # Episodes to trigger rollback


class CurriculumManager:
    """Manages curriculum learning progression"""
    
    def __init__(self, config: CurriculumConfig, scenario_generator: ScenarioGenerator,
                 log_dir: Optional[str] = None):
        self.config = config
        self.scenario_generator = scenario_generator
        self.log_dir = Path(log_dir) if log_dir else Path("./curriculum_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize curriculum state
        self.current_difficulty = 0.1
        self.current_stage = 0
        self.episodes_in_stage = 0
        self.success_history = []
        self.performance_history = []
        self.stage_history = []
        
        # Available scenarios by difficulty
        self.scenarios_by_difficulty = self._organize_scenarios_by_difficulty()
        
        # Current scenario pool
        self.current_scenarios = self._get_scenarios_for_difficulty(self.current_difficulty)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup curriculum learning logging"""
        log_file = self.log_dir / "curriculum.log"
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def _organize_scenarios_by_difficulty(self) -> Dict[float, List[TrainingScenario]]:
        """Organize scenarios by difficulty levels"""
        scenarios_by_diff = {}
        
        # Create difficulty bins
        difficulty_levels = np.arange(0.1, 1.1, 0.1)
        
        for difficulty in difficulty_levels:
            # Get scenarios within difficulty range
            scenarios = self.scenario_generator.get_scenarios_by_difficulty(
                float(max(0.0, difficulty - 0.1)), float(difficulty + 0.05)
            )
            if scenarios:
                scenarios_by_diff[round(difficulty, 1)] = scenarios
                
        return scenarios_by_diff
    
    def _get_scenarios_for_difficulty(self, difficulty: float) -> List[TrainingScenario]:
        """Get scenarios for current difficulty level"""
        scenarios = []
        
        # Include all scenarios up to current difficulty
        for diff_level in self.scenarios_by_difficulty:
            if diff_level <= difficulty + 0.05:  # Small tolerance
                scenarios.extend(self.scenarios_by_difficulty[diff_level])
        
        # Ensure we have at least basic scenarios
        if not scenarios:
            scenarios = self.scenario_generator.get_scenarios_by_difficulty(0.0, 0.2)
            
        return scenarios
    
    def get_next_scenario(self) -> TrainingScenario:
        """Get next training scenario based on curriculum"""
        if not self.current_scenarios:
            self.current_scenarios = self._get_scenarios_for_difficulty(self.current_difficulty)
        
        # Weight scenarios by difficulty (prefer current level)
        weights = []
        for scenario in self.current_scenarios:
            # Higher weight for scenarios near current difficulty
            diff_distance = abs(scenario.difficulty_weight - self.current_difficulty)
            weight = max(0.1, 1.0 - diff_distance * 2)  # Closer = higher weight
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Select scenario
        scenario_idx = np.random.choice(len(self.current_scenarios), p=weights)
        return self.current_scenarios[scenario_idx]
    
    def record_episode_result(self, scenario_name: str, success: bool, 
                            performance_metrics: Dict[str, float]):
        """Record the result of an episode"""
        self.episodes_in_stage += 1
        self.success_history.append(success)
        self.performance_history.append(performance_metrics)
        
        # Limit history size
        max_history = self.config.success_window * 3
        if len(self.success_history) > max_history:
            self.success_history = self.success_history[-max_history:]
            self.performance_history = self.performance_history[-max_history:]
        
        # Check for progression
        if self.episodes_in_stage >= self.config.min_episodes_per_stage:
            self._check_progression()
        
        # Log progress
        if self.episodes_in_stage % 100 == 0:
            self._log_progress()
    
    def _check_progression(self):
        """Check if we should progress to next difficulty level"""
        if len(self.success_history) < self.config.success_window:
            return
        
        # Calculate recent success rate
        recent_successes = self.success_history[-self.config.success_window:]
        success_rate = np.mean(recent_successes)
        
        # Get current threshold
        threshold = self._get_current_threshold()
        
        # Check for advancement
        if (success_rate >= threshold and 
            self.episodes_in_stage >= self.config.min_episodes_per_stage):
            self._advance_difficulty()
        
        # Check for rollback
        elif (self.config.enable_rollback and 
              success_rate < self.config.rollback_threshold and
              self.episodes_in_stage >= self.config.rollback_episodes):
            self._rollback_difficulty()
        
        # Force advancement if too many episodes
        elif self.episodes_in_stage >= self.config.max_episodes_per_stage:
            self.logger.warning(f"Forcing advancement after {self.episodes_in_stage} episodes")
            self._advance_difficulty()
    
    def _get_current_threshold(self) -> float:
        """Get current success threshold (adaptive)"""
        threshold = self.config.initial_success_threshold
        
        if self.config.adaptive_thresholds:
            # Gradually reduce threshold as difficulty increases
            decay_factor = self.config.threshold_decay ** self.current_stage
            threshold = max(self.config.min_threshold, threshold * decay_factor)
        
        return threshold
    
    def _advance_difficulty(self):
        """Advance to next difficulty level"""
        old_difficulty = self.current_difficulty
        
        # Increase difficulty
        self.current_difficulty = min(
            self.config.max_difficulty,
            self.current_difficulty + self.config.difficulty_increment
        )
        
        # Update scenarios
        self.current_scenarios = self._get_scenarios_for_difficulty(self.current_difficulty)
        
        # Reset stage tracking
        self.current_stage += 1
        self.episodes_in_stage = 0
        
        # Log advancement
        recent_success_rate = np.mean(self.success_history[-self.config.success_window:])
        self.logger.info(
            f"Advanced difficulty from {old_difficulty:.1f} to {self.current_difficulty:.1f} "
            f"(Stage {self.current_stage}) - Success rate: {recent_success_rate:.3f}"
        )
        
        # Save state
        self._save_curriculum_state()
    
    def _rollback_difficulty(self):
        """Roll back to previous difficulty level"""
        if self.current_difficulty <= 0.1:
            return  # Can't roll back further
        
        old_difficulty = self.current_difficulty
        
        # Decrease difficulty
        self.current_difficulty = max(
            0.1,
            self.current_difficulty - self.config.difficulty_increment
        )
        
        # Update scenarios
        self.current_scenarios = self._get_scenarios_for_difficulty(self.current_difficulty)
        
        # Reset stage tracking
        self.episodes_in_stage = 0
        
        # Log rollback
        recent_success_rate = np.mean(self.success_history[-self.config.success_window:])
        self.logger.warning(
            f"Rolled back difficulty from {old_difficulty:.1f} to {self.current_difficulty:.1f} "
            f"- Success rate: {recent_success_rate:.3f}"
        )
        
        # Save state
        self._save_curriculum_state()
    
    def _log_progress(self):
        """Log current progress"""
        if len(self.success_history) >= self.config.success_window:
            recent_success_rate = np.mean(self.success_history[-self.config.success_window:])
        else:
            recent_success_rate = np.mean(self.success_history) if self.success_history else 0.0
        
        threshold = self._get_current_threshold()
        
        self.logger.info(
            f"Stage {self.current_stage}, Episode {self.episodes_in_stage}, "
            f"Difficulty: {self.current_difficulty:.1f}, "
            f"Success Rate: {recent_success_rate:.3f} (threshold: {threshold:.3f}), "
            f"Scenarios: {len(self.current_scenarios)}"
        )
    
    def _save_curriculum_state(self):
        """Save curriculum state to file"""
        state = {
            'current_difficulty': self.current_difficulty,
            'current_stage': self.current_stage,
            'episodes_in_stage': self.episodes_in_stage,
            'success_history': self.success_history[-1000:],  # Keep last 1000
            'stage_history': self.stage_history,
            'current_scenarios': [s.name for s in self.current_scenarios]
        }
        
        state_file = self.log_dir / "curriculum_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_curriculum_state(self, state_file: str):
        """Load curriculum state from file"""
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            self.current_difficulty = state['current_difficulty']
            self.current_stage = state['current_stage']
            self.episodes_in_stage = state['episodes_in_stage']
            self.success_history = state['success_history']
            self.stage_history = state['stage_history']
            
            # Update scenarios
            self.current_scenarios = self._get_scenarios_for_difficulty(self.current_difficulty)
            
            self.logger.info(f"Loaded curriculum state from {state_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load curriculum state: {e}")
    
    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get current curriculum statistics"""
        if len(self.success_history) >= self.config.success_window:
            recent_success_rate = np.mean(self.success_history[-self.config.success_window:])
        else:
            recent_success_rate = np.mean(self.success_history) if self.success_history else 0.0
        
        return {
            'current_difficulty': self.current_difficulty,
            'current_stage': self.current_stage,
            'episodes_in_stage': self.episodes_in_stage,
            'recent_success_rate': recent_success_rate,
            'threshold': self._get_current_threshold(),
            'total_episodes': len(self.success_history),
            'num_scenarios': len(self.current_scenarios),
            'scenario_names': [s.name for s in self.current_scenarios]
        }
    
    def is_curriculum_complete(self) -> bool:
        """Check if curriculum learning is complete"""
        return bool(self.current_difficulty >= self.config.max_difficulty and
                   len(self.success_history) >= self.config.success_window and
                   np.mean(self.success_history[-self.config.success_window:]) >= self.config.min_threshold)


def create_adaptive_curriculum(scenario_generator: ScenarioGenerator,
                             initial_threshold: float = 0.8,
                             log_dir: str = "./curriculum_logs") -> CurriculumManager:
    """Create an adaptive curriculum learning manager"""
    
    config = CurriculumConfig(
        initial_success_threshold=initial_threshold,
        min_episodes_per_stage=500,
        max_episodes_per_stage=2000,
        success_window=100,
        difficulty_increment=0.1,
        adaptive_thresholds=True,
        threshold_decay=0.95,
        min_threshold=0.6,
        enable_rollback=True,
        rollback_threshold=0.4,
        rollback_episodes=200
    )
    
    return CurriculumManager(config, scenario_generator, log_dir)


def evaluate_scenario_performance(results: List[Dict[str, Any]], 
                                scenario: TrainingScenario) -> bool:
    """Evaluate if performance on a scenario is successful"""
    if not results:
        return False
    
    # Extract key metrics
    angles = [r.get('final_angle', float('inf')) for r in results]
    rates = [r.get('final_rate', float('inf')) for r in results]
    settling_times = [r.get('settling_time', float('inf')) for r in results]
    
    # Check success criteria
    angle_success = np.mean([abs(a) <= scenario.max_stable_angle for a in angles])
    rate_success = np.mean([abs(r) <= scenario.max_stable_rate for r in rates])
    settling_success = np.mean([t <= scenario.settling_time for t in settling_times])
    
    # Overall success (all criteria must be met for majority of episodes)
    overall_success = bool(angle_success >= 0.8 and 
                          rate_success >= 0.8 and 
                          settling_success >= 0.7)
    
    return overall_success