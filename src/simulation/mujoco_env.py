"""
Enhanced MuJoCo Simulation Environment for Real-Life TVC System

This module provides a comprehensive MuJoCo-based simulation environment
that integrates with YAML configuration files and implements realistic
physics for rocket thrust vector control systems.

Features:
- YAML configuration integration for rocket dimensions and parameters
- Realistic physics with environmental factors (wind, gravity variations)
- Detailed rocket geometry based on actual specifications
- Advanced actuator dynamics and sensor modeling
- Real-life constraints and safety limits

Author: Enhanced by GitHub Copilot (God Mode)
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Tuple, Any, List, Union, TYPE_CHECKING
import os
import tempfile
import yaml
from pathlib import Path
from dataclasses import dataclass

# Import from the main source directory
import sys
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.dynamics import TVCParameters
from src.control import MPCController, CLFCBFQPFilter
from src.utils.config import ExperimentConfig


@dataclass
class RealLifeSimParams:
    """Enhanced parameters for realistic MuJoCo simulation"""
    # Simulation parameters
    timestep: float = 0.0005  # Higher precision (0.5ms)
    control_timestep: float = 0.01  # 100Hz control loop
    
    # Rendering
    render_mode: str = "rgb_array"
    render_width: int = 1280
    render_height: int = 720
    
    # Kinematics mode
    planar_2d: bool = True  # Use planar base (x,z, pitch) when True; else full 3D free base
    tvc_only_actions: bool = True  # If True, actions are gimbal commands only; throttle fixed
    
    # Environment bounds
    max_angle: float = np.pi/3  # ±60 degrees
    max_rate: float = 4*np.pi   # ±4π rad/s
    max_episode_steps: int = 2000  # 20 seconds at 100Hz
    
    # Environmental conditions
    wind_speed_mean: float = 0.0     # m/s
    wind_speed_std: float = 2.0      # m/s
    wind_direction_std: float = 0.5  # rad
    gravity_variation: float = 0.01  # ±1% gravity variation
    air_density_sea_level: float = 1.225       # kg/m³ (sea level)
    temperature_kelvin: float = 288.15         # Standard temperature (15°C)
    pressure_sea_level: float = 101325.0       # Pa (sea level)
    
    # Atmospheric model parameters
    enable_altitude_effects: bool = True        # Air density/pressure vary with altitude
    temperature_lapse_rate: float = -0.0065    # K/m (standard atmosphere)
    scale_height: float = 8400.0               # m (atmospheric scale height)
    
    # Ground effects
    enable_ground_effect: bool = True           # Ground effect for landing
    ground_effect_height: float = 5.0          # m (height where ground effect starts)
    ground_effect_strength: float = 0.15       # Multiplicative factor
    
    # Advanced wind modeling
    enable_turbulence: bool = True              # Realistic turbulence
    turbulence_intensity: float = 0.1          # Turbulence strength (0-1)
    wind_shear_rate: float = 0.1               # Wind change with altitude (m/s per m)
    gust_frequency: float = 0.2                # Hz (gust frequency)
    gust_intensity: float = 2.0                # m/s (gust strength)
    
    # Engine and fuel modeling
    enable_fuel_consumption: bool = True        # Realistic fuel usage
    initial_fuel_mass_fraction: float = 0.7    # Fuel as fraction of total mass
    specific_impulse: float = 280.0            # seconds (engine efficiency)
    engine_response_time: float = 0.05         # seconds (throttle response)
    min_throttle_level: float = 0.3            # Minimum engine throttle
    
    # Performance variations
    enable_engine_degradation: bool = False     # Engine performance degrades over time
    thrust_variation_std: float = 0.02         # ±2% thrust variation
    temperature_thrust_factor: float = 0.001   # Thrust change per degree K
    
    # Sensor parameters
    imu_noise_std: float = 0.001     # rad
    imu_bias_drift: float = 0.0001   # rad/s per second
    gyro_noise_std: float = 0.002    # rad/s
    accelerometer_noise_std: float = 0.1  # m/s²
    barometer_noise_std: float = 1.0      # Pa (pressure sensor noise)
    gps_noise_std: float = 0.1            # m (GPS position noise)
    gps_dropout_rate: float = 0.01        # Probability of GPS signal loss
    
    # Actuator dynamics
    actuator_delay_steps: int = 2    # 20ms delay
    actuator_bandwidth: float = 20.0 # Hz
    actuator_noise_std: float = 0.001 # rad
    actuator_friction: float = 0.02   # Static friction in actuators
    actuator_backlash: float = 0.001  # rad (mechanical backlash)
    
    # Derived actuator time constant from bandwidth (τ ≈ 1/(2πf))
    @property
    def gimbal_time_constant(self) -> float:
        return 1.0 / max(1e-6, (2.0 * np.pi * self.actuator_bandwidth))
    
    # Optional engine (throttle) lag time constant (separate from response time shaping)
    thrust_time_constant: float = 0.1
    
    # Realistic physics
    enable_aerodynamics: bool = True
    enable_propellant_slosh: bool = True
    enable_structural_flexibility: bool = False  # Advanced feature
    
    # Advanced aerodynamics
    drag_coefficient: float = 0.5             # Rocket drag coefficient
    reference_area: float = 0.008            # m² (cross-sectional area)
    lift_coefficient: float = 0.1            # Side force due to angle of attack
    magnus_coefficient: float = 0.05         # Magnus effect for spinning rockets


def create_realistic_rocket_mjcf(config: ExperimentConfig, sim_params: RealLifeSimParams) -> str:
    """
    Create realistic MuJoCo XML model based on YAML configuration
    
    Args:
        config: Experiment configuration from YAML
        sim_params: Simulation parameters
        
    Returns:
        XML string defining the realistic rocket model
    """
    
    params = config.plant
    
    # Calculate realistic dimensions based on mass and typical rocket proportions
    # Assuming cylindrical rocket with reasonable proportions
    total_mass = params.mass
    length = params.length
    
    # Estimate rocket diameter from mass (typical density ~300-500 kg/m³ for rockets)
    estimated_volume = total_mass / 400.0  # kg/m³
    radius = np.sqrt(estimated_volume / (np.pi * length)) * 0.5  # Conservative radius
    radius = max(0.02, min(0.2, radius))  # Clamp between 2cm and 20cm
    
    # Nozzle dimensions
    nozzle_length = length * 0.15  # 15% of total length
    nozzle_radius = radius * 0.4   # 40% of body radius
    
    # Engine gimbal mount dimensions
    gimbal_length = nozzle_length * 0.3
    gimbal_radius = nozzle_radius * 0.8
    
    # Material properties
    body_density = total_mass / (np.pi * radius**2 * length)
    
    # Root joint configuration based on 2D/3D selection
    if sim_params.planar_2d:
        root_joint_xml = (
            '                <joint name="slide_x" type="slide" axis="1 0 0"/>'
            '\n                <joint name="slide_z" type="slide" axis="0 0 1"/>'
            '\n                <joint name="pitch"   type="hinge" axis="0 1 0" limited="true" range="-1.0 1.0"/>'
        )
    else:
        root_joint_xml = '                <freejoint/>'

    # Include yaw joint only in 3D
    yaw_joint_xml = ""
    yaw_actuator_xml = ""
    if not sim_params.planar_2d:
        yaw_joint_xml = (
            f"                    <joint name=\"gimbal_yaw\" type=\"hinge\" axis=\"1 0 0\" \n"
            f"                           range=\"{-params.max_gimbal_angle*0.5} {params.max_gimbal_angle*0.5}\"\n"
            f"                           damping=\"0.5\" frictionloss=\"0.01\"/>\n"
        )
        yaw_actuator_xml = (
            f"            <motor name=\"gimbal_yaw_motor\" joint=\"gimbal_yaw\" \n"
            f"                   gear=\"{params.nominal_thrust * 0.1}\"\n"
            f"                   ctrllimited=\"true\" ctrlrange=\"{-params.max_gimbal_angle*0.5} {params.max_gimbal_angle*0.5}\"\n"
            f"                   forcelimited=\"true\" forcerange=\"-100 100\"/>\n"
        )

    mjcf = f"""
    <mujoco model="Realistic_Rocket_TVC">
        <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
        
        <option timestep="{sim_params.timestep}" iterations="100" solver="PGS" jacobian="sparse">
            <flag gravity="enable" contact="enable" override="enable"/>
        </option>
        
        <size nconmax="100" njmax="1000"/>
        
        <default>
            <joint limited="true" damping="0.05" frictionloss="0.01"/>
            <geom friction="0.9 0.01 0.001" solref="0.02 1" solimp="0.9 0.95 0.001"/>
            <motor gear="1" ctrllimited="true" forcelimited="true"/>
        </default>
        
        <asset>
            <!-- Textures for realistic appearance -->
            <texture name="rocket_body" type="2d" builtin="checker" width="256" height="256"
                     rgb1="0.8 0.1 0.1" rgb2="0.9 0.2 0.2"/>
            <texture name="nozzle" type="2d" builtin="gradient" width="256" height="256"
                     rgb1="0.3 0.3 0.3" rgb2="0.6 0.6 0.6"/>
            <texture name="ground" type="2d" builtin="checker" width="512" height="512"
                     rgb1="0.6 0.6 0.4" rgb2="0.7 0.7 0.5"/>
            
            <material name="rocket_mat" texture="rocket_body" specular="0.3" shininess="0.5"/>
            <material name="nozzle_mat" texture="nozzle" specular="0.8" shininess="0.9"/>
            <material name="ground_mat" texture="ground" specular="0.1" shininess="0.1"/>
        </asset>
        
        <worldbody>
            <!-- Ground plane with realistic material -->
            <light pos="10 10 20" dir="-1 -1 -2" diffuse="0.8 0.8 0.8"/>
            <light pos="-10 10 20" dir="1 -1 -2" diffuse="0.6 0.6 0.8"/>
            
            <geom name="ground" type="plane" size="50 50 0.1" material="ground_mat"
                  friction="1.0 0.05 0.001"/>
            
            <!-- Main rocket body -->
            <body name="rocket" pos="0 0 {length/2 + 0.1}" quat="1 0 0 0">
{root_joint_xml}
                <!-- Main body structure -->
                <geom name="body_main" type="cylinder" size="{radius} {length/2}" 
                      material="rocket_mat" mass="{total_mass * 0.7}" density="{body_density}"/>
                
                <!-- Nose cone -->
                <geom name="nose_cone" type="capsule" size="{radius*0.8} {length*0.1}" 
                      pos="0 0 {length/2 + length*0.05}" material="rocket_mat" 
                      mass="{total_mass * 0.05}"/>
                
                <!-- Fins for aerodynamic stability -->
                <geom name="fin1" type="box" size="{radius*0.3} 0.01 {length*0.2}" 
                      pos="{radius*1.2} 0 -{length*0.3}" material="rocket_mat"
                      mass="{total_mass * 0.02}"/>
                <geom name="fin2" type="box" size="{radius*0.3} 0.01 {length*0.2}" 
                      pos="-{radius*1.2} 0 -{length*0.3}" material="rocket_mat"
                      mass="{total_mass * 0.02}"/>
                <geom name="fin3" type="box" size="0.01 {radius*0.3} {length*0.2}" 
                      pos="0 {radius*1.2} -{length*0.3}" material="rocket_mat"
                      mass="{total_mass * 0.02}"/>
                <geom name="fin4" type="box" size="0.01 {radius*0.3} {length*0.2}" 
                      pos="0 -{radius*1.2} -{length*0.3}" material="rocket_mat"
                      mass="{total_mass * 0.02}"/>
                
                <!-- Inertial properties (realistic for rocket) -->
                <inertial pos="0 0 0" mass="{total_mass}" 
                         diaginertia="{params.moment_of_inertia} {params.moment_of_inertia} {params.moment_of_inertia/5}"/>
                
                <!-- Engine gimbal system -->
                <body name="gimbal_mount" pos="0 0 -{length/2}">
                    <!-- Gimbal joint with realistic limits and dynamics -->
                    <joint name="gimbal_pitch" type="hinge" axis="0 1 0" 
                           range="{-params.max_gimbal_angle} {params.max_gimbal_angle}"
                           damping="0.5" frictionloss="0.01"/>
                    
{yaw_joint_xml}                    
                    
                    <!-- Gimbal mount structure -->
                    <geom name="gimbal_structure" type="cylinder" size="{gimbal_radius} {gimbal_length/2}" 
                          material="nozzle_mat" mass="{total_mass * 0.05}"/>
                    
                    <!-- Engine nozzle -->
                    <body name="nozzle" pos="0 0 -{gimbal_length/2}">
                        <geom name="nozzle_outer" type="cylinder" size="{nozzle_radius} {nozzle_length/2}" 
                              material="nozzle_mat" mass="{total_mass * 0.08}"/>
                        <geom name="nozzle_inner" type="cylinder" size="{nozzle_radius*0.7} {nozzle_length/2}" 
                              pos="0 0 0" rgba="0.1 0.1 0.1 1" mass="{total_mass * 0.02}"/>
                        
                        <!-- Thrust application point -->
                        <site name="thrust_point" pos="0 0 -{nozzle_length/2}" size="0.005"/>
                        
                        <!-- Exhaust visualization point -->
                        <site name="exhaust_point" pos="0 0 -{nozzle_length}" size="0.01" rgba="1 0.5 0 0.5"/>
                    </body>
                </body>
                
                <!-- Sensor mounting points -->
                <site name="imu_sensor" pos="0 0 {length*0.1}" size="0.01"/>
                <site name="gps_sensor" pos="0 0 {length*0.4}" size="0.01"/>
                
                <!-- Additional masses for fuel tanks (simplified) -->
                <geom name="fuel_tank" type="cylinder" size="{radius*0.8} {length*0.3}" 
                      pos="0 0 0" rgba="0.3 0.3 0.8 0.3" mass="{total_mass * 0.06}"/>
            </body>
            
            <!-- Landing pad target -->
            <geom name="landing_target" type="cylinder" size="2.0 0.01" pos="0 0 0.01" 
                  rgba="0 1 0 0.8"/>
            <geom name="landing_target_center" type="cylinder" size="0.5 0.02" pos="0 0 0.02" 
                  rgba="1 0 0 0.8"/>
        </worldbody>
        
        <actuator>
            <!-- Realistic gimbal actuators with proper dynamics -->
            <motor name="gimbal_pitch_motor" joint="gimbal_pitch" 
                   gear="{params.nominal_thrust * 0.1}" 
                   ctrllimited="true" ctrlrange="{-params.max_gimbal_angle} {params.max_gimbal_angle}"
                   forcelimited="true" forcerange="-200 200"/>
{yaw_actuator_xml}            
        </actuator>
        
        <sensor>
            <!-- Realistic sensor suite -->
            <accelerometer name="imu_accel" site="imu_sensor"/>
            <gyro name="imu_gyro" site="imu_sensor"/>
            <magnetometer name="imu_mag" site="imu_sensor"/>
            <framequat name="attitude" objtype="site" objname="imu_sensor"/>
            <framepos name="position" objtype="site" objname="gps_sensor"/>
            
            <!-- Joint sensors -->
            <jointpos name="gimbal_pitch_pos" joint="gimbal_pitch"/>
            <jointvel name="gimbal_pitch_vel" joint="gimbal_pitch"/>
            <jointpos name="gimbal_yaw_pos" joint="gimbal_yaw"/>
            <jointvel name="gimbal_yaw_vel" joint="gimbal_yaw"/>
        </sensor>
        
        <visual>
            <map force="0.1" zfar="100"/>
            <rgba haze="0.1 0.15 0.25 1"/>
            <quality shadowsize="4096"/>
            <global offwidth="{sim_params.render_width}" offheight="{sim_params.render_height}"/>
        </visual>
    </mujoco>
    """
    
    return mjcf


class RealisticTVCMuJoCoEnv(gym.Env):
    """
    Realistic MuJoCo-based TVC simulation environment with YAML configuration
    
    Provides comprehensive physics simulation with:
    - YAML configuration integration
    - Realistic rocket geometry and mass properties
    - Environmental factors (wind, gravity variations)
    - Advanced sensor modeling with noise and bias
    - Detailed actuator dynamics
    - Aerodynamic effects
    """
    
    def __init__(self,
                 config_path: Optional[Union[Path, str]] = None,
                 config_name: str = "default",
                 sim_params: Optional[RealLifeSimParams] = None,
                 use_safety_filter: bool = False,
                 render_mode: Optional[str] = None):
        
        super().__init__()
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / f"{config_name}.yaml"
        else:
            if isinstance(config_path, str):
                config_path = Path(config_path)
        
        self.config = self._load_config(config_path)
        self.sim_params = sim_params or RealLifeSimParams()
        self.use_safety_filter = use_safety_filter
        
        # Override render mode if specified
        if render_mode is not None:
            self.sim_params.render_mode = render_mode
        
        # Create realistic MuJoCo model
        self._create_mujoco_model()
        
        # Initialize safety filter if requested
        if use_safety_filter:
            from src.dynamics import TVCPlant
            plant = TVCPlant(self.config.plant)
            baseline = MPCController(plant)
            self.safety_filter = CLFCBFQPFilter(plant, baseline)
        else:
            self.safety_filter = None
        
        # Define observation and action spaces
        self._define_spaces()
        
        # Environment state
        self.step_count = 0
        self.episode_reward = 0.0
        self.total_thrust_used = 0.0
        
        # Actuator internal states (for first-order lag + rate limits)
        self._gimbal_pitch_state = 0.0
        self._gimbal_yaw_state = 0.0
        self._thrust_state = 0.0
        
        # Control delay buffers
        self.pitch_control_buffer = [0.0] * (self.sim_params.actuator_delay_steps + 1)
        self.yaw_control_buffer = [0.0] * (self.sim_params.actuator_delay_steps + 1)
        
        # Sensor state
        self.sensor_bias = np.zeros(3)  # IMU bias drift
        self.wind_state = np.array([0.0, 0.0])  # Wind velocity [x, z]
        
        # New realistic state variables
        self.current_fuel_mass = self.config.plant.mass * self.sim_params.initial_fuel_mass_fraction
        self.atmospheric_state = {
            'temperature': self.sim_params.temperature_kelvin,
            'pressure': self.sim_params.pressure_sea_level,
            'air_density': self.sim_params.air_density_sea_level
        }
        self.engine_state = {
            'throttle_response_filter': 0.0,
            'temperature': self.sim_params.temperature_kelvin,
            'performance_factor': 1.0
        }
        self.turbulence_state = {
            'intensity': 0.0,
            'direction': 0.0,
            'phase': 0.0
        }
        self.gps_available = True
        
        # Rendering
        self.renderer = None
        
        print(f"Realistic TVC Environment initialized with config: {config_name}")
        print(f"Rocket mass: {self.config.plant.mass}kg, length: {self.config.plant.length}m")
        
    def _load_config(self, config_path: Union[Path, str]) -> ExperimentConfig:
        """Load configuration from YAML file"""
        try:
            if isinstance(config_path, str):
                config_path = Path(config_path)
            with open(config_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            # Extract plant parameters
            plant_data = yaml_data.get('plant', {})
            plant = TVCParameters(
                mass=plant_data.get('mass', 1.0),
                moment_of_inertia=plant_data.get('moment_of_inertia', 0.1),
                length=plant_data.get('length', 1.0),
                nominal_thrust=plant_data.get('nominal_thrust', 15.0),
                max_gimbal_angle=plant_data.get('max_gimbal_angle', np.pi/6),
                max_angle=plant_data.get('max_angle', np.pi/4),
                max_angular_rate=plant_data.get('max_angular_rate', 2*np.pi)
            )
            
            # Create simplified config (we mainly need plant parameters)
            config = ExperimentConfig(
                plant=plant,
                mpc=None,  # Will be set/used by safety filter if needed
                safety=None,
                evolution=None,
                ppo=None,
                simulation=None,
                experiment_name=yaml_data.get('experiment_name', 'realistic_sim'),
                use_safety_filter=yaml_data.get('use_safety_filter', True)
            )
            
            return config
            
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            print("Using default parameters...")
            return ExperimentConfig(
                plant=TVCParameters(),
                mpc=None, safety=None, evolution=None, ppo=None, simulation=None
            )
    
    def _create_mujoco_model(self):
        """Create and load realistic MuJoCo model"""
        # Generate MJCF
        mjcf_xml = create_realistic_rocket_mjcf(self.config, self.sim_params)
        
        # Create temporary file for the model
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(mjcf_xml)
            self.model_path = f.name
        
        try:
            # Load model
            self.model = mujoco.MjModel.from_xml_string(mjcf_xml)  # type: ignore[attr-defined]
            self.data = mujoco.MjData(self.model)  # type: ignore[attr-defined]
            
            # Set simulation parameters
            self.model.opt.timestep = self.sim_params.timestep
            
            # Find important indices
            self._find_model_indices()
            
        except Exception as e:
            print(f"Error creating MuJoCo model: {e}")
            raise
    
    def _find_model_indices(self):
        """Find important body, joint, and sensor indices"""
        try:
            self.rocket_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rocket")  # type: ignore[attr-defined]
            self.gimbal_pitch_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gimbal_pitch")  # type: ignore[attr-defined]
            self.gimbal_yaw_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gimbal_yaw")  # type: ignore[attr-defined]
            # Planar base joints (may not exist in 3D mode)
            try:
                self.slide_x_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slide_x")  # type: ignore[attr-defined]
            except Exception:
                self.slide_x_joint_id = -1
            try:
                self.slide_z_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slide_z")  # type: ignore[attr-defined]
            except Exception:
                self.slide_z_joint_id = -1
            try:
                self.pitch_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pitch")  # type: ignore[attr-defined]
            except Exception:
                self.pitch_joint_id = -1
            self.gimbal_pitch_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gimbal_pitch_motor")  # type: ignore[attr-defined]
            self.gimbal_yaw_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gimbal_yaw_motor")  # type: ignore[attr-defined]
            self.thrust_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "thrust_point")  # type: ignore[attr-defined]
            self.imu_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu_sensor")  # type: ignore[attr-defined]
            
        except Exception as e:
            print(f"Warning: Could not find all model elements: {e}")
            # Set defaults if not found
            self.rocket_body_id = 1 if self.model.nbody > 1 else 0
            self.gimbal_pitch_joint_id = 0 if self.model.njnt > 0 else -1
            self.gimbal_yaw_joint_id = 1 if self.model.njnt > 1 else -1
            self.gimbal_pitch_actuator_id = 0 if self.model.nu > 0 else -1
            self.gimbal_yaw_actuator_id = 1 if self.model.nu > 1 else -1
            self.thrust_site_id = 0 if self.model.nsite > 0 else -1
            self.imu_site_id = 0 if self.model.nsite > 0 else -1
            self.slide_x_joint_id = -1
            self.slide_z_joint_id = -1
            self.pitch_joint_id = -1
    
    def _define_spaces(self):
        """Define observation and action spaces for realistic environment"""
        # Enhanced observation space: [position(3), orientation(4), velocity(3), angular_velocity(3), gimbal_angles(2)]
        # Total: 15 dimensional observation
        obs_high = np.array([
            100.0, 100.0, 200.0,  # Position bounds [x, y, z] in meters
            1.0, 1.0, 1.0, 1.0,   # Quaternion orientation (normalized)
            50.0, 50.0, 50.0,     # Linear velocity bounds [m/s]
            self.sim_params.max_rate, self.sim_params.max_rate, self.sim_params.max_rate,  # Angular velocity [rad/s]
            self.config.plant.max_gimbal_angle, self.config.plant.max_gimbal_angle  # Gimbal angles [rad]
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32
        )
        
        # Actions: either TVC-only (gimbal pitch[, yaw]) or thrust+gimbals
        if self.sim_params.tvc_only_actions:
            if self.sim_params.planar_2d:
                low = np.array([-self.config.plant.max_gimbal_angle], dtype=np.float32)
                high = np.array([self.config.plant.max_gimbal_angle], dtype=np.float32)
            else:
                low = np.array([-self.config.plant.max_gimbal_angle, -self.config.plant.max_gimbal_angle], dtype=np.float32)
                high = np.array([self.config.plant.max_gimbal_angle, self.config.plant.max_gimbal_angle], dtype=np.float32)
        else:
            if self.sim_params.planar_2d:
                low = np.array([0.0, -self.config.plant.max_gimbal_angle], dtype=np.float32)
                high = np.array([self.config.plant.nominal_thrust * 1.5, self.config.plant.max_gimbal_angle], dtype=np.float32)
            else:
                low = np.array([0.0, -self.config.plant.max_gimbal_angle, -self.config.plant.max_gimbal_angle], dtype=np.float32)
                high = np.array([self.config.plant.nominal_thrust * 1.5, self.config.plant.max_gimbal_angle, self.config.plant.max_gimbal_angle], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state with realistic conditions"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)  # type: ignore[attr-defined]
        
        # Set realistic initial conditions
        initial_height = np.random.uniform(10.0, 50.0)  # 10-50m altitude
        initial_x_pos = np.random.uniform(-5.0, 5.0)    # ±5m lateral offset
        initial_y_pos = np.random.uniform(-5.0, 5.0)    # ±5m lateral offset
        
        # Set initial position/velocity based on planar or free base
        if self.rocket_body_id >= 0:
            if self.sim_params.planar_2d and self.slide_x_joint_id >= 0 and self.slide_z_joint_id >= 0 and self.pitch_joint_id >= 0:
                # Addresses for qpos and qvel
                qpos_x = self.model.jnt_qposadr[self.slide_x_joint_id]
                qpos_z = self.model.jnt_qposadr[self.slide_z_joint_id]
                qpos_pitch = self.model.jnt_qposadr[self.pitch_joint_id]
                dof_x = self.model.jnt_dofadr[self.slide_x_joint_id]
                dof_z = self.model.jnt_dofadr[self.slide_z_joint_id]
                dof_pitch = self.model.jnt_dofadr[self.pitch_joint_id]

                # Positions
                self.data.qpos[qpos_x] = initial_x_pos
                self.data.qpos[qpos_z] = initial_height
                initial_angle = np.random.uniform(-0.1, 0.1)
                self.data.qpos[qpos_pitch] = initial_angle

                # Velocities
                self.data.qvel[dof_x] = np.random.uniform(-2.0, 2.0)
                self.data.qvel[dof_z] = np.random.uniform(-5.0, -1.0)
                self.data.qvel[dof_pitch] = np.random.uniform(-0.2, 0.2)
            else:
                # Assume a freejoint at rocket root
                jnt_id = self.model.body_jntadr[self.rocket_body_id]
                if jnt_id >= 0:
                    qposadr = self.model.jnt_qposadr[jnt_id]
                    dofadr = self.model.jnt_dofadr[jnt_id]
                    # Set position and quaternion
                    self.data.qpos[qposadr:qposadr+3] = [initial_x_pos, initial_y_pos, initial_height]
                    initial_angle = np.random.uniform(-0.1, 0.1)
                    quat = np.array([np.cos(initial_angle/2), 0, np.sin(initial_angle/2), 0])
                    quat = quat / np.linalg.norm(quat)
                    self.data.qpos[qposadr+3:qposadr+7] = quat
                    # Linear and angular velocities
                    self.data.qvel[dofadr:dofadr+3] = [
                        np.random.uniform(-2.0, 2.0),
                        np.random.uniform(-2.0, 2.0),
                        np.random.uniform(-5.0, -1.0)
                    ]
                    self.data.qvel[dofadr+3:dofadr+6] = [
                        np.random.uniform(-0.2, 0.2),
                        np.random.uniform(-0.2, 0.2),
                        np.random.uniform(-0.2, 0.2)
                    ]
        
        # Reset environment state
        self.step_count = 0
        self.episode_reward = 0.0
        self.total_thrust_used = 0.0
        
        # Reset control buffers
        self.pitch_control_buffer = [0.0] * (self.sim_params.actuator_delay_steps + 1)
        self.yaw_control_buffer = [0.0] * (self.sim_params.actuator_delay_steps + 1)
        
        # Reset sensor bias
        self.sensor_bias = np.random.normal(0, self.sim_params.imu_bias_drift, 3)
        
        # Reset wind conditions
        self.wind_state = np.random.normal(
            [self.sim_params.wind_speed_mean, 0], 
            [self.sim_params.wind_speed_std, self.sim_params.wind_speed_std/2]
        )
        
        # Forward dynamics
        mujoco.mj_forward(self.model, self.data)  # type: ignore[attr-defined]
        
        return self._get_observation(), {"initial_position": [initial_x_pos, initial_y_pos, initial_height]}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step with realistic physics"""
        if self.sim_params.tvc_only_actions:
            thrust_magnitude = self.config.plant.nominal_thrust
            if self.sim_params.planar_2d:
                gimbal_pitch = float(action[0])
                gimbal_yaw = 0.0
            else:
                gimbal_pitch = float(action[0])
                gimbal_yaw = float(action[1]) if len(action) > 1 else 0.0
        else:
            thrust_magnitude = float(action[0])
            gimbal_pitch = float(action[1])
            gimbal_yaw = float(action[2]) if len(action) > 2 else 0.0
        
        # Apply safety filter if enabled
        if self.safety_filter is not None:
            current_obs = self._get_observation()
            # Extract relevant state for 2D safety filter (pitch angle and rate)
            pitch_angle = self._extract_pitch_angle(current_obs[3:7])  # From quaternion
            pitch_rate = current_obs[9]  # Angular velocity around pitch axis
            
            state_2d = np.array([pitch_angle, pitch_rate])
            filter_result = self.safety_filter.filter_control(gimbal_pitch, state_2d)
            gimbal_pitch_safe = filter_result['control']
            gimbal_yaw_safe = np.clip(gimbal_yaw, -self.config.plant.max_gimbal_angle, 
                                     self.config.plant.max_gimbal_angle)
            safety_intervention = filter_result['intervention']
        else:
            gimbal_pitch_safe = np.clip(gimbal_pitch, -self.config.plant.max_gimbal_angle, 
                                       self.config.plant.max_gimbal_angle)
            gimbal_yaw_safe = np.clip(gimbal_yaw, -self.config.plant.max_gimbal_angle, 
                                     self.config.plant.max_gimbal_angle)
            safety_intervention = False
        
        # Apply control delays
        self.pitch_control_buffer.append(gimbal_pitch_safe)
        self.yaw_control_buffer.append(gimbal_yaw_safe)
        delayed_pitch = self.pitch_control_buffer.pop(0)
        delayed_yaw = self.yaw_control_buffer.pop(0)
        
        # First-order actuator lag with rate limits
        def first_order_rate_limited(state, cmd, dt, rate_limit, tau):
            # Clamp command within angle limits
            cmd = np.clip(cmd, -self.config.plant.max_gimbal_angle, self.config.plant.max_gimbal_angle)
            # Desired derivative
            u_dot = (cmd - state) / max(1e-6, tau)
            u_dot = np.clip(u_dot, -rate_limit, rate_limit)
            return state + dt * u_dot

        # Control integration at control_timestep
        dtc = self.sim_params.control_timestep
        self._gimbal_pitch_state = first_order_rate_limited(
            self._gimbal_pitch_state, delayed_pitch, dtc, self.config.plant.max_gimbal_rate, self.sim_params.gimbal_time_constant
        )
        if not self.sim_params.planar_2d:
            self._gimbal_yaw_state = first_order_rate_limited(
                self._gimbal_yaw_state, delayed_yaw, dtc, self.config.plant.max_gimbal_rate, self.sim_params.gimbal_time_constant
            )
        else:
            self._gimbal_yaw_state = 0.0

        # Drive MuJoCo motors toward these states (position control proxy)
        if self.gimbal_pitch_actuator_id >= 0:
            self.data.ctrl[self.gimbal_pitch_actuator_id] = self._gimbal_pitch_state
        if self.gimbal_yaw_actuator_id >= 0 and not self.sim_params.planar_2d:
            self.data.ctrl[self.gimbal_yaw_actuator_id] = self._gimbal_yaw_state
        
        # Apply thrust force with realistic dynamics
        self._apply_thrust_and_environmental_forces(thrust_magnitude, delayed_pitch, delayed_yaw)
        
        # Step simulation multiple times for control frequency
        steps_per_control = int(self.sim_params.control_timestep / self.sim_params.timestep)
        for _ in range(steps_per_control):
            mujoco.mj_step(self.model, self.data)  # type: ignore[attr-defined]
        
        # Update sensor bias drift
        self.sensor_bias += np.random.normal(0, self.sim_params.imu_bias_drift, 3) * self.sim_params.control_timestep
        
        # Update wind conditions (slowly varying)
        wind_change = np.random.normal(0, 0.1, 2) * self.sim_params.control_timestep
        self.wind_state += wind_change
        self.wind_state = np.clip(self.wind_state, [-10, -5], [10, 5])  # Reasonable wind limits
        
        # Get new observation
        observation = self._get_observation()
        
        # Compute reward
        reward = self._compute_realistic_reward(
            observation, thrust_magnitude, self._gimbal_pitch_state, self._gimbal_yaw_state, safety_intervention
        )
        
        # Check termination
        self.step_count += 1
        terminated = self._is_terminated(observation)
        truncated = self.step_count >= self.sim_params.max_episode_steps
        
        self.episode_reward += reward
        self.total_thrust_used += thrust_magnitude * self.sim_params.control_timestep
        
        # Comprehensive info dictionary
        info = {
            'safety_intervention': safety_intervention,
            'thrust_command': thrust_magnitude,
            'gimbal_pitch_command': gimbal_pitch,
            'gimbal_yaw_command': gimbal_yaw,
            'gimbal_pitch_actual': delayed_pitch,
            'gimbal_yaw_actual': delayed_yaw,
            'wind_velocity': self.wind_state.copy(),
            'total_thrust_used': self.total_thrust_used,
            'episode_reward': self.episode_reward if terminated or truncated else None,
            'altitude': observation[2],
            'landing_error': np.linalg.norm(observation[:2])  # Distance from target
        }
        
        return observation, reward, terminated, truncated, info
    
    def _update_atmospheric_conditions(self, altitude: float):
        """Update atmospheric conditions based on altitude using standard atmosphere model"""
        if not self.sim_params.enable_altitude_effects:
            return
        
        # Standard atmosphere model
        altitude = max(0.0, altitude)  # Don't go below sea level
        
        # Temperature variation with altitude
        temperature = (self.sim_params.temperature_kelvin + 
                      self.sim_params.temperature_lapse_rate * altitude)
        temperature = max(200.0, temperature)  # Minimum temperature limit
        
        # Pressure variation (barometric formula)
        pressure = (self.sim_params.pressure_sea_level * 
                   np.exp(-altitude / self.sim_params.scale_height))
        
        # Air density (ideal gas law)
        air_density = (pressure / (287.0 * temperature))  # R = 287 J/(kg*K) for air
        
        self.atmospheric_state.update({
            'temperature': temperature,
            'pressure': pressure,
            'air_density': air_density
        })
    
    def _update_turbulence_model(self, dt: float):
        """Update realistic turbulence model"""
        if not self.sim_params.enable_turbulence:
            return
        
        # Update turbulence phase for oscillatory behavior
        self.turbulence_state['phase'] += 2 * np.pi * self.sim_params.gust_frequency * dt
        
        # Dryden turbulence model (simplified)
        gust_component = (self.sim_params.gust_intensity * 
                         np.sin(self.turbulence_state['phase']) * 
                         np.random.normal(0, 1))
        
        # Random walk for turbulence intensity
        intensity_change = np.random.normal(0, 0.01) * dt
        self.turbulence_state['intensity'] += intensity_change
        self.turbulence_state['intensity'] = np.clip(
            self.turbulence_state['intensity'], 0, 1
        )
        
        # Update turbulence direction
        direction_change = np.random.normal(0, 0.1) * dt
        self.turbulence_state['direction'] += direction_change
        
        # Apply turbulence to wind
        turbulence_magnitude = (self.sim_params.turbulence_intensity * 
                               self.turbulence_state['intensity'] * 
                               (1 + 0.5 * gust_component))
        
        turbulence_wind = turbulence_magnitude * np.array([
            np.cos(self.turbulence_state['direction']),
            np.sin(self.turbulence_state['direction'])
        ])
        
        # Add to base wind
        self.wind_state += turbulence_wind * dt
    
    def _calculate_ground_effect(self, altitude: float, velocity: np.ndarray) -> np.ndarray:
        """Calculate ground effect forces for landing"""
        if not self.sim_params.enable_ground_effect or altitude > self.sim_params.ground_effect_height:
            return np.zeros(3)
        
        # Ground effect increases lift and reduces drag near the ground
        ground_factor = 1.0 - (altitude / self.sim_params.ground_effect_height)
        ground_factor = max(0.0, ground_factor)
        
        # Additional upward force due to ground effect
        velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude > 0.1:
            # Ground effect creates additional lift
            lift_force = (self.sim_params.ground_effect_strength * 
                         ground_factor * 
                         self.atmospheric_state['air_density'] * 
                         velocity_magnitude**2 * 
                         self.sim_params.reference_area)
            
            return np.array([0, 0, lift_force])
        
        return np.zeros(3)
    
    def _update_fuel_consumption(self, thrust_magnitude: float, dt: float):
        """Update fuel consumption based on thrust usage"""
        if not self.sim_params.enable_fuel_consumption:
            return
        
        # Rocket equation: fuel consumption rate = thrust / (g * Isp)
        g0 = 9.81  # Standard gravity
        fuel_flow_rate = thrust_magnitude / (g0 * self.sim_params.specific_impulse)
        
        # Update fuel mass
        fuel_consumed = fuel_flow_rate * dt
        self.current_fuel_mass = max(0.0, self.current_fuel_mass - fuel_consumed)
        
        # Update total rocket mass in MuJoCo model
        dry_mass = self.config.plant.mass * (1 - self.sim_params.initial_fuel_mass_fraction)
        new_total_mass = dry_mass + self.current_fuel_mass
        
        # Update mass in simulation (simplified - affects whole rocket body)
        if self.rocket_body_id >= 0 and hasattr(self.model, 'body_mass'):
            if self.rocket_body_id < len(self.model.body_mass):
                mass_ratio = new_total_mass / self.config.plant.mass
                self.model.body_mass[self.rocket_body_id] *= mass_ratio
    
    def _calculate_enhanced_aerodynamics(self, position: np.ndarray, velocity: np.ndarray, 
                                       angular_velocity: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Calculate comprehensive aerodynamic forces"""
        if not self.sim_params.enable_aerodynamics:
            return np.zeros(3)
        
        altitude = position[2]
        self._update_atmospheric_conditions(altitude)
        
        # Wind with altitude effects and turbulence
        self._update_turbulence_model(self.sim_params.control_timestep)
        
        # Wind shear - wind speed increases with altitude
        wind_altitude_factor = 1.0 + self.sim_params.wind_shear_rate * altitude
        current_wind_3d = np.array([self.wind_state[0] * wind_altitude_factor, 0, 
                                   self.wind_state[1] * wind_altitude_factor])
        
        # Relative wind velocity
        relative_wind = current_wind_3d - velocity
        wind_speed = np.linalg.norm(relative_wind)
        
        if wind_speed < 0.1:
            return np.zeros(3)
        
        wind_direction = relative_wind / wind_speed
        air_density = self.atmospheric_state['air_density']
        
        # Basic drag force
        drag_magnitude = (0.5 * air_density * self.sim_params.drag_coefficient * 
                         self.sim_params.reference_area * wind_speed**2)
        drag_force = drag_magnitude * wind_direction
        
        # Angle of attack effects (simplified)
        rocket_forward = np.array([0, 0, 1])  # Rocket points up in local frame
        angle_of_attack = np.arccos(np.clip(np.dot(rocket_forward, -wind_direction), -1, 1))
        
        # Side force due to angle of attack
        if angle_of_attack > 0.1:  # Only if significant angle
            side_direction = np.cross(rocket_forward, wind_direction)
            if np.linalg.norm(side_direction) > 0:
                side_direction /= np.linalg.norm(side_direction)
                side_force_magnitude = (0.5 * air_density * self.sim_params.lift_coefficient * 
                                       self.sim_params.reference_area * wind_speed**2 * 
                                       np.sin(angle_of_attack))
                side_force = side_force_magnitude * side_direction
                drag_force += side_force
        
        # Magnus effect for spinning rockets
        spin_magnitude = np.linalg.norm(angular_velocity)
        if spin_magnitude > 0.1:
            magnus_direction = np.cross(angular_velocity, relative_wind)
            if np.linalg.norm(magnus_direction) > 0:
                magnus_direction /= np.linalg.norm(magnus_direction)
                magnus_magnitude = (0.5 * air_density * self.sim_params.magnus_coefficient * 
                                   self.sim_params.reference_area * wind_speed * spin_magnitude)
                magnus_force = magnus_magnitude * magnus_direction
                drag_force += magnus_force
        
        # Ground effect
        ground_effect_force = self._calculate_ground_effect(altitude, velocity)
        
        return drag_force + ground_effect_force
    
    def _update_engine_dynamics(self, thrust_command: float, dt: float) -> float:
        """Update realistic engine dynamics with throttle response and performance variations"""
        # Engine response time (first-order lag)
        response_time_constant = self.sim_params.engine_response_time
        alpha = dt / (response_time_constant + dt)
        
        self.engine_state['throttle_response_filter'] += alpha * (
            thrust_command - self.engine_state['throttle_response_filter']
        )
        
        # Apply minimum throttle constraint
        if self.engine_state['throttle_response_filter'] > 0:
            min_thrust = self.config.plant.nominal_thrust * self.sim_params.min_throttle_level
            actual_thrust = max(min_thrust, self.engine_state['throttle_response_filter'])
        else:
            actual_thrust = 0.0
        
        # Performance variations due to temperature
        temp_factor = 1.0 + (self.atmospheric_state['temperature'] - 
                            self.sim_params.temperature_kelvin) * self.sim_params.temperature_thrust_factor
        
        # Random thrust variations
        thrust_noise = np.random.normal(1.0, self.sim_params.thrust_variation_std)
        
        # Final thrust with all factors
        final_thrust = actual_thrust * temp_factor * thrust_noise * self.engine_state['performance_factor']
        
        return max(0.0, final_thrust)

    def _apply_thrust_and_environmental_forces(self, thrust_magnitude: float, gimbal_pitch: float, gimbal_yaw: float):
        """Apply thrust forces and environmental effects"""
        # Clear previously applied external forces on rocket to avoid accumulation
        if self.rocket_body_id >= 0 and self.rocket_body_id < self.data.xfrc_applied.shape[0]:
            self.data.xfrc_applied[self.rocket_body_id, :] = 0.0

        # Clamp thrust and apply throttle lag
        thrust_cmd = np.clip(thrust_magnitude, 0, self.config.plant.nominal_thrust * 1.5)
        tau_T = max(1e-6, self.sim_params.thrust_time_constant)
        self._thrust_state += (self.sim_params.control_timestep / tau_T) * (thrust_cmd - self._thrust_state)
        thrust_magnitude = max(0.0, self._thrust_state)
        
        # Get current gimbal angles from simulation
        if self.gimbal_pitch_joint_id >= 0:
            actual_pitch = self.data.qpos[self.gimbal_pitch_joint_id]
        else:
            actual_pitch = gimbal_pitch
            
        if self.gimbal_yaw_joint_id >= 0:
            actual_yaw = self.data.qpos[self.gimbal_yaw_joint_id]
        else:
            actual_yaw = gimbal_yaw
        
        # Compute nozzle axis in world frame from site xmat if available
        thrust_direction = np.array([0.0, 0.0, 1.0])
        if self.thrust_site_id >= 0 and self.thrust_site_id < self.data.site_xmat.shape[0]:
            # site_xmat is 3x3 row-major per site
            R_ws = self.data.site_xmat[self.thrust_site_id].reshape(3, 3)
            # Nozzle axis assumed local -Z (pointing down). If model uses +Z up, invert accordingly
            thrust_direction = R_ws @ np.array([0.0, 0.0, -1.0])
            n = np.linalg.norm(thrust_direction)
            if n > 1e-9:
                thrust_direction = thrust_direction / n
        
        # Apply thrust force at thrust point
        thrust_force = thrust_magnitude * thrust_direction
        
        if self.thrust_site_id >= 0 and self.rocket_body_id >= 0:
            # Apply force at the nozzle site with equivalent wrench: F at point r produces torque τ = r × F
            thrust_point = self.data.site_xpos[self.thrust_site_id]
            body_com = self.data.xipos[self.rocket_body_id]
            r = thrust_point - body_com
            torque = np.cross(r, thrust_force)
            self.data.xfrc_applied[self.rocket_body_id, 0:3] += thrust_force
            self.data.xfrc_applied[self.rocket_body_id, 3:6] += torque
        
        # Apply wind forces (aerodynamic drag simplified)
        if self.sim_params.enable_aerodynamics and self.rocket_body_id >= 0:
            # Get rocket velocity
            # cvel has shape (nbody, 6): [linear_vel(3), angular_vel(3)]
            rocket_vel = self.data.cvel[self.rocket_body_id, 0:3]
            
            # Relative wind velocity
            wind_3d = np.array([self.wind_state[0], 0, self.wind_state[1]])
            relative_wind = wind_3d - rocket_vel
            
            # Simplified drag force: F_drag = 0.5 * rho * Cd * A * v^2
            drag_coeff = 0.5
            reference_area = np.pi * (0.05)**2  # Rough estimate
            air_density = self.atmospheric_state['air_density']
            
            wind_speed_sq = np.linalg.norm(relative_wind)**2
            if wind_speed_sq > 0.01:  # Avoid division by zero
                wind_direction = relative_wind / np.linalg.norm(relative_wind)
                drag_force = 0.5 * air_density * drag_coeff * reference_area * wind_speed_sq * wind_direction
                
                # Apply drag force using the same method as thrust
                if self.data.xfrc_applied.shape[0] > self.rocket_body_id:
                    self.data.xfrc_applied[self.rocket_body_id, 0:3] += drag_force
        
        # Apply gravity variation
        if self.sim_params.gravity_variation > 0:
            gravity_factor = 1.0 + np.random.normal(0, self.sim_params.gravity_variation)
            self.model.opt.gravity[2] = -9.81 * gravity_factor
    
    def _extract_pitch_angle(self, quaternion: np.ndarray) -> float:
        """Extract pitch angle from quaternion"""
        # Convert quaternion to rotation matrix and extract pitch
        w, x, y, z = quaternion
        pitch = np.arcsin(2 * (w * y - z * x))
        return pitch
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation with realistic sensor modeling"""
        obs = np.zeros(15, dtype=np.float32)
        
        if self.rocket_body_id >= 0 and self.rocket_body_id < self.model.nbody:
            # Position (with GPS-like noise)
            if self.rocket_body_id < len(self.data.xpos):
                position = self.data.xpos[self.rocket_body_id].copy()
                position += np.random.normal(0, 0.1, 3)  # 10cm GPS noise
                obs[0:3] = position
            
            # Orientation quaternion (with IMU noise)
            if self.rocket_body_id < len(self.data.xquat):
                quat = self.data.xquat[self.rocket_body_id].copy()
                # Add small rotation noise
                noise_angle = np.random.normal(0, self.sim_params.imu_noise_std)
                noise_axis = np.random.normal(0, 1, 3)
                if np.linalg.norm(noise_axis) > 0:
                    noise_axis /= np.linalg.norm(noise_axis)
                    noise_quat = np.array([np.cos(noise_angle/2), *(noise_axis * np.sin(noise_angle/2))])
                # Simplified - should properly compose quaternions
                obs[3:7] = quat
            
            # Linear velocity (with noise) - properly access cvel array
            velocity = np.zeros(3)
            if hasattr(self.data, 'cvel') and self.data.cvel.shape[0] > self.rocket_body_id:
                # cvel has shape (nbody, 6) where 6 = [linear_vel(3), angular_vel(3)]
                velocity = self.data.cvel[self.rocket_body_id, 0:3].copy()
            else:
                # Fallback: estimate velocity from position change
                if hasattr(self, '_last_position'):
                    dt = self.sim_params.timestep
                    velocity = (obs[0:3] - self._last_position) / dt
                self._last_position = obs[0:3].copy()
                
            # Add velocity noise
            velocity = velocity + np.random.normal(0, 0.05, 3)  # 5cm/s velocity noise
            obs[7:10] = velocity
            
            # Angular velocity (with gyro noise and bias) - properly access cvel array
            ang_velocity = np.zeros(3)
            if hasattr(self.data, 'cvel') and self.data.cvel.shape[0] > self.rocket_body_id:
                # Angular velocity is in columns 3:6 of cvel
                ang_velocity = self.data.cvel[self.rocket_body_id, 3:6].copy()
            
            ang_velocity = ang_velocity + self.sensor_bias + np.random.normal(0, self.sim_params.gyro_noise_std, 3)
            obs[10:13] = ang_velocity
        
        # Gimbal angles (with actuator feedback noise)
        if self.gimbal_pitch_joint_id >= 0 and self.gimbal_pitch_joint_id < len(self.data.qpos):
            obs[13] = self.data.qpos[self.gimbal_pitch_joint_id] + np.random.normal(0, self.sim_params.actuator_noise_std)
        
        if self.gimbal_yaw_joint_id >= 0 and self.gimbal_yaw_joint_id < len(self.data.qpos):
            obs[14] = self.data.qpos[self.gimbal_yaw_joint_id] + np.random.normal(0, self.sim_params.actuator_noise_std)
        
        return obs.astype(np.float32)
    
    def _compute_realistic_reward(self, observation: np.ndarray, thrust: float, pitch: float, yaw: float, safety_intervention: bool) -> float:
        """Compute reward with realistic objectives"""
        position = observation[0:3]
        velocity = observation[7:10]
        ang_velocity = observation[10:13]
        
        # Target is landing at origin
        target_position = np.array([0.0, 0.0, 0.0])
        
        # Distance to target
        distance_error = np.linalg.norm(position - target_position)
        
        # Velocity magnitude (should be small for landing)
        velocity_magnitude = np.linalg.norm(velocity)
        
        # Angular velocity magnitude (should be small for stability)
        angular_velocity_magnitude = np.linalg.norm(ang_velocity)
        
        # Altitude component (positive reward for controlled descent)
        altitude = position[2]
        altitude_reward = 0.0
        if altitude > 0:
            # Reward for maintaining reasonable altitude
            altitude_reward = 0.1 if 1.0 < altitude < 20.0 else -0.1
        
        # Landing detection
        landing_bonus = 0.0
        if altitude < 0.5 and distance_error < 2.0 and velocity_magnitude < 1.0:
            landing_bonus = 10.0  # Big reward for successful landing
        
        # Fuel efficiency (penalize excessive thrust)
        fuel_penalty = -0.001 * thrust
        
        # Control effort penalty
        control_penalty = -0.01 * (abs(pitch) + abs(yaw))
        
        # Safety penalty
        safety_penalty = -1.0 if safety_intervention else 0.0
        
        # Stability reward (small angles and velocities)
        stability_reward = 0.1 * np.exp(-distance_error - 0.5 * velocity_magnitude - angular_velocity_magnitude)
        
        total_reward = (
            -0.1 * distance_error +           # Main tracking objective
            -0.05 * velocity_magnitude +      # Soft landing objective
            -0.02 * angular_velocity_magnitude + # Stability objective
            altitude_reward +                 # Altitude maintenance
            landing_bonus +                   # Landing achievement
            fuel_penalty +                    # Fuel efficiency
            control_penalty +                 # Control smoothness
            safety_penalty +                  # Safety compliance
            stability_reward                  # Overall stability
        )
        
        return total_reward
    
    def _is_terminated(self, observation: np.ndarray) -> bool:
        """Check termination conditions"""
        position = observation[0:3]
        velocity = observation[7:10]
        quaternion = observation[3:7]
        ang_velocity = observation[10:13]
        
        # Extract attitude angles from quaternion
        pitch = self._extract_pitch_angle(quaternion)
        
        # Safety limits
        altitude_limit = position[2] < -1.0  # Below ground
        distance_limit = np.linalg.norm(position[:2]) > 100.0  # Too far laterally
        angle_limit = abs(pitch) > self.sim_params.max_angle  # Excessive tilt
        rate_limit = np.linalg.norm(ang_velocity) > self.sim_params.max_rate  # Excessive rotation
        velocity_limit = np.linalg.norm(velocity) > 50.0  # Excessive velocity
        
        # Successful landing condition
        successful_landing = (
            position[2] < 1.0 and  # Near ground
            np.linalg.norm(position[:2]) < 2.0 and  # Near target
            np.linalg.norm(velocity) < 2.0 and  # Low velocity
            abs(pitch) < 0.2  # Upright
        )
        
        return bool(altitude_limit or distance_limit or angle_limit or rate_limit or velocity_limit or successful_landing)
    
    def render(self, mode: Optional[str] = None):
        """Render the environment with enhanced visuals"""
        if mode is None:
            mode = self.sim_params.render_mode
        
        if mode == "human":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, 
                                              height=self.sim_params.render_height,
                                              width=self.sim_params.render_width)
            
            self.renderer.update_scene(self.data)
            image = self.renderer.render()
            
            # Display using matplotlib
            import matplotlib.pyplot as plt
            plt.imshow(image)
            plt.axis('off')
            plt.title(f"Realistic TVC Simulation - Step {self.step_count}")
            plt.show()
            
        elif mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model,
                                              height=self.sim_params.render_height, 
                                              width=self.sim_params.render_width)
            
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        
        return None
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'model_path') and os.path.exists(self.model_path):
            os.unlink(self.model_path)
        
        if self.renderer is not None:
            self.renderer.close()
    
    def get_info(self) -> Dict[str, Any]:
        """Get detailed simulation information"""
        obs = self._get_observation()
        return {
            'config_name': self.config.experiment_name,
            'rocket_mass': self.config.plant.mass,
            'rocket_length': self.config.plant.length,
            'current_position': obs[0:3].tolist(),
            'current_velocity': obs[7:10].tolist(),
            'wind_conditions': self.wind_state.tolist(),
            'total_thrust_used': self.total_thrust_used,
            'step_count': self.step_count
        }


def test_realistic_env():
    """Test the realistic MuJoCo environment with different configurations"""
    print("Testing Realistic MuJoCo TVC Environment")
    print("=" * 50)
    
    # Test with different rocket configurations
    configs = ["default", "small_rocket", "large_rocket"]
    
    for config_name in configs:
        print(f"\nTesting with {config_name} configuration:")
        try:
            env = RealisticTVCMuJoCoEnv(config_name=config_name, render_mode=None)
            
            print(f"  Observation space: {env.observation_space.shape}")
            print(f"  Action space: {env.action_space.shape}")
            print(f"  Rocket mass: {env.config.plant.mass}kg")
            print(f"  Rocket length: {env.config.plant.length}m")
            
            # Test reset
            obs, info = env.reset(seed=42)
            print(f"  Initial position: {obs[0:3]}")
            print(f"  Initial altitude: {obs[2]:.2f}m")
            
            # Test a few steps
            for i in range(5):
                # Reasonable action respecting interface
                if env.sim_params.tvc_only_actions:
                    if env.sim_params.planar_2d:
                        action = np.array([0.05], dtype=np.float32)  # small pitch gimbal
                    else:
                        action = np.array([0.05, 0.02], dtype=np.float32)  # pitch, yaw
                else:
                    if env.sim_params.planar_2d:
                        action = np.array([env.config.plant.nominal_thrust * 0.8, 0.05], dtype=np.float32)
                    else:
                        action = np.array([env.config.plant.nominal_thrust * 0.8, 0.05, 0.02], dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                
                if i == 0:
                    print(f"  Step 1 reward: {reward:.3f}")
                    print(f"  Landing error: {info['landing_error']:.2f}m")
            
            env.close()
            print(f"  ✓ {config_name} test completed successfully")
            
        except Exception as e:
            print(f"  ✗ {config_name} test failed: {e}")
    
    print("\nRealistic environment testing completed!")


if __name__ == "__main__":
    test_realistic_env()