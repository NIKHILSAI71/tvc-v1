"""
Comprehensive Training Scenarios for Real-World TVC Systems

This module defines various training scenarios that cover all aspects of real-world
TVC rocket operations including launch, flight, landing, and emergency scenarios.

Author: Enhanced by GitHub Copilot (God Mode)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import random


class ScenarioType(Enum):
    """Types of training scenarios"""
    HOVERING = "hovering"
    LAUNCH = "launch"
    ASCENT = "ascent"
    LANDING = "landing"
    DISTURBANCE_REJECTION = "disturbance_rejection"
    ENGINE_OUT = "engine_out"
    WIND_REJECTION = "wind_rejection"
    FUEL_DEPLETION = "fuel_depletion"
    HIGH_DYNAMICS = "high_dynamics"
    PRECISION_LANDING = "precision_landing"
    EMERGENCY_RECOVERY = "emergency_recovery"


@dataclass
class TrainingScenario:
    """Definition of a training scenario"""
    name: str
    scenario_type: ScenarioType
    description: str
    
    # Initial conditions
    initial_angle_range: Tuple[float, float]  # min, max in radians
    initial_rate_range: Tuple[float, float]   # min, max in rad/s
    initial_thrust_range: Tuple[float, float] # min, max as fraction of nominal
    
    # Environment conditions
    wind_speed_range: Tuple[float, float]     # m/s
    turbulence_intensity: float               # 0-1
    gravity_variation: float                  # fraction of nominal
    
    # Simulation parameters
    episode_duration: float                   # seconds
    difficulty_weight: float                  # 0-1, for curriculum learning
    
    # Special conditions
    engine_failure_prob: float = 0.0         # probability of engine failure
    thrust_variation: float = 0.0            # thrust variation std
    mass_variation: float = 0.0              # mass variation for fuel consumption
    
    # Success criteria
    max_stable_angle: float = 0.1             # rad
    max_stable_rate: float = 0.2              # rad/s
    settling_time: float = 3.0                # seconds


class ScenarioGenerator:
    """Generates training scenarios for curriculum learning"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Define all scenarios
        self.scenarios = self._create_all_scenarios()
        
        # Sort by difficulty for curriculum learning
        self.scenarios.sort(key=lambda x: x.difficulty_weight)
    
    def _create_all_scenarios(self) -> List[TrainingScenario]:
        """Create all training scenarios"""
        scenarios = []
        
        # 1. HOVERING - Basic stabilization (Easy)
        scenarios.append(TrainingScenario(
            name="basic_hovering",
            scenario_type=ScenarioType.HOVERING,
            description="Basic hovering with small perturbations",
            initial_angle_range=(-0.05, 0.05),      # ±3 degrees
            initial_rate_range=(-0.1, 0.1),         # ±6 deg/s
            initial_thrust_range=(0.95, 1.05),      # ±5% thrust
            wind_speed_range=(0.0, 1.0),            # Light wind
            turbulence_intensity=0.05,
            gravity_variation=0.0,
            episode_duration=10.0,
            difficulty_weight=0.1
        ))
        
        # 2. MODERATE HOVERING (Medium-Easy)
        scenarios.append(TrainingScenario(
            name="moderate_hovering",
            scenario_type=ScenarioType.HOVERING,
            description="Hovering with moderate disturbances",
            initial_angle_range=(-0.15, 0.15),      # ±9 degrees
            initial_rate_range=(-0.3, 0.3),         # ±17 deg/s
            initial_thrust_range=(0.9, 1.1),        # ±10% thrust
            wind_speed_range=(0.0, 3.0),            # Moderate wind
            turbulence_intensity=0.1,
            gravity_variation=0.01,
            episode_duration=15.0,
            difficulty_weight=0.2
        ))
        
        # 3. DISTURBANCE REJECTION (Medium)
        scenarios.append(TrainingScenario(
            name="disturbance_rejection",
            scenario_type=ScenarioType.DISTURBANCE_REJECTION,
            description="Handle external disturbances and gusts",
            initial_angle_range=(-0.1, 0.1),        # ±6 degrees
            initial_rate_range=(-0.2, 0.2),         # ±11 deg/s
            initial_thrust_range=(0.9, 1.1),        # ±10% thrust
            wind_speed_range=(2.0, 8.0),            # Strong wind
            turbulence_intensity=0.2,
            gravity_variation=0.02,
            episode_duration=12.0,
            difficulty_weight=0.3,
            thrust_variation=0.05
        ))
        
        # 4. LAUNCH PHASE (Medium-Hard)
        scenarios.append(TrainingScenario(
            name="launch_phase",
            scenario_type=ScenarioType.LAUNCH,
            description="Initial launch phase with high dynamics",
            initial_angle_range=(-0.3, 0.3),        # ±17 degrees
            initial_rate_range=(-0.8, 0.8),         # ±46 deg/s
            initial_thrust_range=(1.2, 1.8),        # High thrust for launch
            wind_speed_range=(0.0, 5.0),            # Variable wind
            turbulence_intensity=0.15,
            gravity_variation=0.01,
            episode_duration=8.0,
            difficulty_weight=0.4,
            thrust_variation=0.03
        ))
        
        # 5. HIGH DYNAMICS (Hard)
        scenarios.append(TrainingScenario(
            name="high_dynamics",
            scenario_type=ScenarioType.HIGH_DYNAMICS,
            description="High angular rates and aggressive maneuvers",
            initial_angle_range=(-0.5, 0.5),        # ±29 degrees
            initial_rate_range=(-1.5, 1.5),         # ±86 deg/s
            initial_thrust_range=(0.8, 1.4),        # Variable thrust
            wind_speed_range=(0.0, 6.0),            # Variable wind
            turbulence_intensity=0.2,
            gravity_variation=0.02,
            episode_duration=6.0,
            difficulty_weight=0.5,
            thrust_variation=0.08
        ))
        
        # 6. WIND REJECTION (Hard)
        scenarios.append(TrainingScenario(
            name="wind_rejection",
            scenario_type=ScenarioType.WIND_REJECTION,
            description="Strong wind and turbulence conditions",
            initial_angle_range=(-0.2, 0.2),        # ±11 degrees
            initial_rate_range=(-0.4, 0.4),         # ±23 deg/s
            initial_thrust_range=(0.9, 1.2),        # Variable thrust
            wind_speed_range=(5.0, 15.0),           # Very strong wind
            turbulence_intensity=0.3,
            gravity_variation=0.02,
            episode_duration=10.0,
            difficulty_weight=0.6,
            thrust_variation=0.1
        ))
        
        # 7. LANDING PHASE (Very Hard)
        scenarios.append(TrainingScenario(
            name="landing_phase",
            scenario_type=ScenarioType.LANDING,
            description="Precision landing with fuel constraints",
            initial_angle_range=(-0.4, 0.4),        # ±23 degrees
            initial_rate_range=(-1.0, 1.0),         # ±57 deg/s
            initial_thrust_range=(0.3, 0.8),        # Low thrust for landing
            wind_speed_range=(0.0, 8.0),            # Variable wind
            turbulence_intensity=0.15,
            gravity_variation=0.01,
            episode_duration=15.0,
            difficulty_weight=0.7,
            mass_variation=0.3,                      # Fuel depletion
            thrust_variation=0.05
        ))
        
        # 8. ENGINE OUT (Very Hard)
        scenarios.append(TrainingScenario(
            name="engine_out",
            scenario_type=ScenarioType.ENGINE_OUT,
            description="Engine failure or reduced thrust scenarios",
            initial_angle_range=(-0.3, 0.3),        # ±17 degrees
            initial_rate_range=(-0.6, 0.6),         # ±34 deg/s
            initial_thrust_range=(0.2, 0.6),        # Severely reduced thrust
            wind_speed_range=(0.0, 4.0),            # Light to moderate wind
            turbulence_intensity=0.1,
            gravity_variation=0.01,
            episode_duration=12.0,
            difficulty_weight=0.8,
            engine_failure_prob=0.3,                # 30% chance of further failure
            thrust_variation=0.15
        ))
        
        # 9. PRECISION LANDING (Extreme)
        scenarios.append(TrainingScenario(
            name="precision_landing",
            scenario_type=ScenarioType.PRECISION_LANDING,
            description="High-precision landing with tight constraints",
            initial_angle_range=(-0.6, 0.6),        # ±34 degrees
            initial_rate_range=(-1.2, 1.2),         # ±69 deg/s
            initial_thrust_range=(0.2, 0.7),        # Low thrust
            wind_speed_range=(2.0, 10.0),           # Challenging wind
            turbulence_intensity=0.25,
            gravity_variation=0.02,
            episode_duration=20.0,
            difficulty_weight=0.9,
            mass_variation=0.4,                      # Significant fuel depletion
            thrust_variation=0.1,
            max_stable_angle=0.02,                   # Very tight precision
            max_stable_rate=0.05
        ))
        
        # 10. EMERGENCY RECOVERY (Extreme)
        scenarios.append(TrainingScenario(
            name="emergency_recovery",
            scenario_type=ScenarioType.EMERGENCY_RECOVERY,
            description="Emergency recovery from extreme conditions",
            initial_angle_range=(-1.0, 1.0),        # ±57 degrees
            initial_rate_range=(-2.0, 2.0),         # ±115 deg/s
            initial_thrust_range=(0.1, 1.5),        # Extreme thrust variation
            wind_speed_range=(0.0, 12.0),           # Extreme wind
            turbulence_intensity=0.4,
            gravity_variation=0.03,
            episode_duration=8.0,
            difficulty_weight=1.0,
            engine_failure_prob=0.2,                # 20% chance of failure
            mass_variation=0.2,
            thrust_variation=0.2,
            max_stable_angle=0.15,                   # Relaxed for emergency
            max_stable_rate=0.3
        ))
        
        return scenarios
    
    def get_scenario_by_name(self, name: str) -> Optional[TrainingScenario]:
        """Get scenario by name"""
        for scenario in self.scenarios:
            if scenario.name == name:
                return scenario
        return None
    
    def get_scenarios_by_type(self, scenario_type: ScenarioType) -> List[TrainingScenario]:
        """Get all scenarios of a specific type"""
        return [s for s in self.scenarios if s.scenario_type == scenario_type]
    
    def get_scenarios_by_difficulty(self, min_difficulty: float = 0.0, 
                                   max_difficulty: float = 1.0) -> List[TrainingScenario]:
        """Get scenarios within difficulty range"""
        return [s for s in self.scenarios 
                if min_difficulty <= s.difficulty_weight <= max_difficulty]
    
    def generate_initial_state(self, scenario: TrainingScenario) -> Dict[str, Any]:
        """Generate random initial state for a scenario"""
        initial_state = {}
        
        # Sample initial conditions
        initial_state['angle'] = np.random.uniform(*scenario.initial_angle_range)
        initial_state['angular_rate'] = np.random.uniform(*scenario.initial_rate_range)
        initial_state['thrust_fraction'] = np.random.uniform(*scenario.initial_thrust_range)
        
        # Sample environment conditions
        initial_state['wind_speed'] = np.random.uniform(*scenario.wind_speed_range)
        initial_state['wind_direction'] = np.random.uniform(0, 2*np.pi)
        initial_state['turbulence_intensity'] = scenario.turbulence_intensity
        initial_state['gravity_variation'] = np.random.uniform(
            -scenario.gravity_variation, scenario.gravity_variation)
        
        # Special conditions
        initial_state['engine_failure'] = (np.random.random() < scenario.engine_failure_prob)
        initial_state['thrust_variation'] = scenario.thrust_variation
        initial_state['mass_fraction'] = 1.0 - np.random.uniform(0, scenario.mass_variation)
        
        # Scenario metadata
        initial_state['scenario_name'] = scenario.name
        initial_state['scenario_type'] = scenario.scenario_type.value
        initial_state['episode_duration'] = scenario.episode_duration
        initial_state['difficulty'] = scenario.difficulty_weight
        
        return initial_state
    
    def create_curriculum_schedule(self, total_episodes: int) -> List[Tuple[int, str]]:
        """Create curriculum learning schedule"""
        schedule = []
        
        # Progressive difficulty introduction
        easy_scenarios = self.get_scenarios_by_difficulty(0.0, 0.3)
        medium_scenarios = self.get_scenarios_by_difficulty(0.3, 0.6)
        hard_scenarios = self.get_scenarios_by_difficulty(0.6, 0.8)
        extreme_scenarios = self.get_scenarios_by_difficulty(0.8, 1.0)
        
        # Phase 1: Easy scenarios (first 30%)
        phase1_episodes = int(total_episodes * 0.3)
        for i in range(phase1_episodes):
            scenario = random.choice(easy_scenarios)
            schedule.append((i, scenario.name))
        
        # Phase 2: Easy + Medium scenarios (next 30%)
        phase2_episodes = int(total_episodes * 0.3)
        mixed_scenarios = easy_scenarios + medium_scenarios
        for i in range(phase1_episodes, phase1_episodes + phase2_episodes):
            scenario = random.choice(mixed_scenarios)
            schedule.append((i, scenario.name))
        
        # Phase 3: Medium + Hard scenarios (next 25%)
        phase3_episodes = int(total_episodes * 0.25)
        mixed_scenarios = medium_scenarios + hard_scenarios
        for i in range(phase1_episodes + phase2_episodes, 
                      phase1_episodes + phase2_episodes + phase3_episodes):
            scenario = random.choice(mixed_scenarios)
            schedule.append((i, scenario.name))
        
        # Phase 4: All scenarios with focus on hard/extreme (final 15%)
        phase4_episodes = total_episodes - (phase1_episodes + phase2_episodes + phase3_episodes)
        all_scenarios = self.scenarios
        # Weight towards harder scenarios
        weights = [s.difficulty_weight + 0.2 for s in all_scenarios]  # Bias towards harder
        for i in range(phase1_episodes + phase2_episodes + phase3_episodes, total_episodes):
            scenario = np.random.choice(len(all_scenarios), p=np.array(weights)/np.sum(weights))
            schedule.append((i, all_scenarios[scenario].name))
        
        return schedule
    
    def get_evaluation_scenarios(self) -> List[TrainingScenario]:
        """Get comprehensive evaluation scenarios"""
        # Include representative scenarios from each difficulty level
        eval_scenarios = [
            self.get_scenario_by_name("basic_hovering"),
            self.get_scenario_by_name("moderate_hovering"),
            self.get_scenario_by_name("disturbance_rejection"),
            self.get_scenario_by_name("launch_phase"),
            self.get_scenario_by_name("high_dynamics"),
            self.get_scenario_by_name("wind_rejection"),
            self.get_scenario_by_name("landing_phase"),
            self.get_scenario_by_name("engine_out"),
            self.get_scenario_by_name("precision_landing"),
            self.get_scenario_by_name("emergency_recovery")
        ]
        return [s for s in eval_scenarios if s is not None]


def create_scenario_config(scenario_name: str) -> Dict[str, Any]:
    """Create configuration dictionary for a specific scenario"""
    generator = ScenarioGenerator()
    scenario = generator.get_scenario_by_name(scenario_name)
    
    if scenario is None:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    # Generate configuration
    config = {
        'experiment_name': f"scenario_{scenario_name}",
        'output_dir': f"./results/scenarios/{scenario_name}",
        'scenario': {
            'name': scenario.name,
            'type': scenario.scenario_type.value,
            'description': scenario.description,
            'difficulty': scenario.difficulty_weight,
            'duration': scenario.episode_duration
        },
        'plant': {
            'mass': 1.0 * (1.0 + np.random.uniform(-scenario.mass_variation, 0)),
            'nominal_thrust': 15.0 * np.random.uniform(*scenario.initial_thrust_range),
            'max_gimbal_angle': 0.524,  # ±30 degrees
            'gravity': 9.81 * (1.0 + np.random.uniform(-scenario.gravity_variation, 
                                                       scenario.gravity_variation))
        },
        'environment': {
            'wind_speed_range': scenario.wind_speed_range,
            'turbulence_intensity': scenario.turbulence_intensity,
            'engine_failure_prob': scenario.engine_failure_prob,
            'thrust_variation': scenario.thrust_variation
        }
    }
    
    return config