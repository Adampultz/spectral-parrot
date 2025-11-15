# config.py
"""
Configuration management for Spectral Parrot training.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import os


@dataclass
class TrainingConfig:
    """Configuration for motor control training with PPO."""
    
    # ========================================
    # Environment Configuration
    # ========================================
    num_motors: int = 8
    early_stopping_threshold: float = 0.5 # The spectral loss threshold below which an episode will be terminated
    max_steps_without_improvement: int = 800 # If no improvement has happened, the episode will terminate
    
    # ========================================
    # Motor Control
    # ========================================
    use_motors: bool = True
    manual_calibration: bool = False # Set to True if you wish to calibrate the motors prior to training. See the README file on motor positioning
    motor_speed: int = 200 
    # motor_steps: int = 25 # How many steps a motor will move per learning step
    use_variable_step_sizes = False
    available_step_sizes = [25, 50, 75, 100]
    step_size_logits_bias: Optional[List[float]] = None
    step_wait_time: float = 0.5 # The time in seconds between the termination of all motors and the next step. 
    reset_wait_time: float = 2 # The time in seconds to allow the strings to settle after an episode reset (which includes motor recalibration)
    reset_calibration: int = 1  # 0: center, 1: random, 2: skip
    
    # Motor position limits (per motor)
    # Maximum steps in each direction from the center. cw should correspond to when the tuner hits its mechanical limit, while ccw is set as the limit 
    # beyond which strings get too slack to vibrate.
    max_ccw_steps: List[int] = field(default_factory=lambda: [4000, 4000, 4000, 4000, 4500, 4500, 4500, 4500])
    max_cw_steps: List[int] = field(default_factory=lambda: [4500, 4500, 4500, 4500, 4500, 4500, 4500, 4500])
    danger_zone_ratio: float = 0.1               # Start penalties at 10% from limit
    critical_zone_ratio: float = 0.02            # Severe penalties at 2% from limit
    ccw_safety_margin: int = 200                 # Safety margin before CCW limit
    limit_penalty: float = 0.0                  # Penalty for hitting limit

    min_step_size: int = 10
    max_step_size: int = 100
    step_size_bias: float = 1.2

    # ========================================
    # PPO Hyperparameters
    # ========================================
    total_timesteps: int = 100000
    max_ep_length: int = 1024
    update_interval: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    
    # Learning rates
    lr_actor: float = 1e-4
    lr_critic: float = 0.0003
    
    # PPO algorithm parameters
    gamma: float = 0.98 # Down from 0.995
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    entropy_coef: float = 0.01

    initial_temperature: float = 2.0             # Initial exploration temperature
    temperature_decay: float = 0.998             # Temperature decay rate per update
    min_temperature: float = 0.3                 # Minimum exploration temperature
    max_grad_norm: float = 0.5                   # Gradient clipping threshold

    use_learning_rate_schedule: bool = False     # Use cosine annealing LR schedule
    lr_schedule_warmup_steps: int = 1000         # Warmup steps for LR schedule
    lr_schedule_total_steps: int = 100000        # Total steps for LR schedule
    gradient_accumulation_steps: int = 1         # Accumulate gradients over N steps
    use_gated_continuous: bool = True            # Use continuous motor step sizes
    
    # ========================================
    # Learning Rate Schedule (NEW)
    # ========================================
    lr_warmup_episodes: int = 20                 # Episodes before LR reduction
    lr_reduced_actor: float = 3e-4               # Actor LR after warmup
    lr_reduced_critic: float = 3e-4              # Critic LR after warmup
    
    # ========================================
    # Network Architecture
    # ========================================
    hidden_size: int = 64
    hold_bias: float = 1.0 # The higher the value, the more prone the network is to turn fewer motors
    
    # ========================================
    # Spectral Loss Normalization
    # ========================================
    use_normalized_loss: bool = True                  # Use volume-invariant normalized loss.
    min_signal_threshold: float = 0.05               # Minimum signal level (below = penalty)
    weak_signal_penalty: float = 0.0                 # Penalty when signal too weak
    normalization_method: str = "l2"                  # "l2" (Frobenius) or "cosine"

    # ========================================
    # Reward Configuration
    # ========================================
    reward_scale: float = 1.0

    # ========================================
    # Reward Shaping (NEW)
    # ========================================
    target_loss: float = 7.0                    # Baseline loss for reward calculation
    initial_movement_penalty: float = 0.0        # Early training movement penalty
    final_movement_penalty: float = 0.0          # Late training movement penalty
    penalty_decay_episodes: int = 50             # Episodes to decay penalty over
    
    # ========================================
    # Stagnation Detection (NEW)
    # ========================================
    stagnation_threshold: float = 0.5            # Loss std dev threshold for stagnation
    stagnation_window: int = 20                  # Window size for detecting stagnation
    
    # ========================================
    # Loss Averaging (NEW)
    # ========================================
    averaging_window_factor: float = 0.6         # Multiplier for loss averaging window
    loss_history_buffer_size: int = 100          # Size of loss history buffer

    # ========================================
    # Reward Function Components (NEW)
    # ========================================
    use_improvement_bonus: bool = True           # Use improvement over previous loss
    use_consistency_bonus: bool = False          # Use consistent improvement bonus
    use_breakthrough_bonus: bool = True          # Use breakthrough to new best
    use_movement_penalty: bool = False           # Penalize excessive movements
    use_stagnation_penalty: bool = False         # Penalize lack of progress
    use_efficiency_bonus: bool = False           # Bonus for fewer motors
    use_proximity_bonus: bool = False            # Bonus near target
    
    # Reward component weights
    improvement_bonus_weight: float = 1.0        # Weight for improvement bonus
    consistency_bonus_weight: float = 5.0        # Weight for consistency bonus
    breakthrough_bonus_weight: float = 2.0      # Weight for breakthrough bonus
    movement_penalty_weight: float = 1.0         # Weight for movement penalty
    efficiency_bonus_weight: float = 0.5         # Weight for efficiency bonus
    
    # Proximity bonus thresholds
    proximity_threshold_close: float = 0.5       # "Getting close" threshold
    proximity_threshold_very_close: float = 0.1  # "Very close" threshold
    proximity_bonus_close: float = 10.0          # Bonus for getting close
    proximity_bonus_very_close: float = 50.0     # Bonus for very close
    
    # ========================================
    # Episode Behavior
    # ========================================
    episode_steps_before_breakthrough: int = 30  # Min steps before breakthrough bonus
    motors_for_movement_penalty: int = 4         # Penalize if more than this many move
    observation_space_loss_max: float = 100.0    # Max loss for observation space
    observation_space_loss_min: float = 0.0      # Min loss for observation space

    # ========================================
    # Advanced PPO Parameters
    # ========================================
    value_coef: float = 0.5                      # Value function loss coefficient
    normalize_advantages: bool = True            # Normalize advantages during training
    normalize_returns: bool = False              # Normalize returns (experimental)
    use_gae: bool = True                         # Use Generalized Advantage Estimation
    ppo_update_frequency: int = 1                # Episodes between PPO updates
    
    # ========================================
    # Action Distribution
    # ========================================
    action_hold_initial_bias: float = 2.0        # Initial bias toward HOLD action
    action_exploration_decay: bool = True        # Decay exploration over time
    
    # ========================================
    # Episode Termination
    # ========================================
    use_early_stopping: bool = True              # Enable early stopping on good loss
    use_truncation: bool = True                  # Enable truncation on stagnation
    min_episode_steps: int = 1                   # Minimum steps per episode
    reward_threshold_for_early_stop: float = None  # Stop episode if reward exceeds this

    # ========================================
    # StallGuard Configuration
    # ========================================
    stallguard_threshold: int = 150              # Global StallGuard threshold (0-255)
    stallguard_warnings_before_stop: int = 5    # Number of warnings before emergency stop
    use_per_motor_stallguard: bool = False       # Use different thresholds per motor
    per_motor_stallguard: List[int] = field(default_factory=lambda: [150]*8)  # Per-motor SG thresholds
    
    # ========================================
    # Motor Movement Parameters
    # ========================================
    motor_acceleration: int = 400                # Motor acceleration (steps/s²)
    motor_microsteps: int = 256                  # Microstepping (1, 2, 4, 8, 16, 32, 64, 128, 256)
    motor_current_ma: int = 1200                 # Motor current in mA (RMS)
    enable_spreadcycle: bool = False             # false=StealthChop, true=SpreadCycle
    
    # ========================================
    # Motor Safety
    # ========================================
    enable_stallguard: bool = True               # Enable StallGuard detection
    motor_timeout_seconds: float = 60.0          # Max time for any single movement
    auto_disable_on_stall: bool = False          # Disable driver on repeated stalls
    stall_count_before_disable: int = 5          # Stalls before auto-disable

    # ========================================
    # STFT and Audio Processing
    # ========================================
    stft_scales: List[int] = field(default_factory=lambda: [512, 1024, 2048, 4096])  # STFT window sizes
    stft_window_type: str = "hann"               # Window function: hann, hamming, blackman
    stft_hop_divisor: int = 4                    # Hop length = window_size / hop_divisor
    use_pyfftw: bool = True                      # Use pyFFTW for faster FFT
    fft_threads: int = 4                         # Number of FFT threads
    
    # ========================================
    # Network Architecture Details (NEW)
    # ========================================
    use_layernorm: bool = True                   # Use layer normalization in networks
    dropout_rate: float = 0.1                    # Dropout rate for regularization
    actor_hidden_layers: List[int] = field(default_factory=lambda: [64, 64])  # Actor hidden layer sizes
    critic_hidden_layers: List[int] = field(default_factory=lambda: [64, 64]) # Critic hidden layer sizes
    use_shared_features: bool = False            # Share feature extraction between actor/critic
    activation_function: str = "relu"            # Activation: relu, tanh, elu

    # ========================================
    # Timeouts and Stability (NEW)
    # ========================================
    motor_completion_timeout: float = 30.0       # Motor completion wait timeout (seconds)
    stabilization_time: float = 2.0              # Wait after motor movement (seconds)
    loss_clip_max: float = 50.0                  # Maximum loss value clipping
    outlier_rejection_threshold: float = 3.0

     # ========================================
    # Memory Management (NEW)
    # ========================================
    gc_interval: int = 10                        # Episodes between garbage collection
    history_maxlen: int = 10000                  # Max episodes to keep in history
    spectral_data_max_age: float = 1.0          # Max age for spectral data (seconds)
    
    # ========================================
    # Experiment Metadata (NEW)
    # ========================================
    experiment_name: str = "default"             # Name for this experiment
    experiment_tags: List[str] = field(default_factory=list)  # Tags for organization
    experiment_notes: str = ""                   # Free-form notes about this run
    random_seed: Optional[int] = None            # Random seed for reproducibility

     # ========================================
    # Serial Communication
    # ========================================
    port1: str = "/dev/cu.usbserial-0001"
    port2: str = "/dev/cu.usbserial-1"
    baudrate: int = 115200
    
    # ========================================
    # Audio Configuration
    # ========================================
    input_device: Optional[int] = None
    sample_rate: int = 48000
    channels: int = 2
    buffer_size: int = 4096
    
    # ========================================
    # Per-Motor Configuration
    # ========================================
    enable_per_motor_settings: bool = False      # Use per-motor speed/step overrides
    per_motor_speeds: List[int] = field(default_factory=lambda: [200]*8)      # Speed per motor
    per_motor_steps: List[int] = field(default_factory=lambda: [50]*8)        # Steps per motor
    per_motor_enabled: List[bool] = field(default_factory=lambda: [True]*8)   # Enable/disable motors
    
    # ========================================
    # Logging Control
    # ========================================
    log_level: str = "INFO"                      # Logging level: DEBUG, INFO, WARNING, ERROR
    log_motor_details: bool = True              # Log detailed motor movements
    log_reward_breakdown: bool = True           # Log reward component breakdown
    log_action_distribution: bool = False        # Log action distribution stats
    log_loss_processor_stats: bool = True       # Log loss processor performance
    
    # Logging frequencies (episodes)
    log_episode_summary_freq: int = 1            # Log episode summary every N episodes
    log_position_status_freq: int = 50           # Log motor positions every N episodes
    log_performance_stats_freq: int = 100        # Log performance stats every N episodes
    
    # ========================================
    # Visualization (NEW)
    # ========================================
    plot_save_frequency: int = 1
    plot_smoothing_window: int = 10              # Smoothing window for plots
    plot_dpi: int = 150                          # DPI for saved plots
    plot_figsize_width: float = 10.0             # Plot width in inches
    plot_figsize_height: float = 10.0            # Plot height in inches
    save_plots: bool = True                      # Save plots to disk
    
    # ========================================
    # Checkpointing Control (NEW)
    # ========================================
    save_best_only: bool = False                 # Only save when best reward improves
    keep_all_checkpoints: bool = True            # Keep all checkpoints vs only latest
    max_checkpoints_to_keep: int = 5             # Max checkpoints if not keeping all
    save_optimizer_state: bool = True            # Include optimizer state in checkpoints

    # ========================================
    # Logging and Checkpointing
    # ========================================
    save_interval: int = 1
    plot_frequency: int = 1
    log_frequency: int = 1
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    log_dir: str = "./logs"                      # Base directory for logs
    log_to_master: bool = False                   # Also log to master.log
    log_filename_format: str = "{timestamp}_{experiment}_ep{episode}.log"  # Format for log files

    # ========================================
    # Training data management
    # ========================================
    training_audio_rotation_interval: int = 5  # Change audio every N episodes (0 = disabled)
    training_audio_folder: str = "/Users/adammac2023/Documents/Musik-business/Projects/Spectral Parrot/Audio/Training Audio/Nov_13th_mono"    
    # ========================================
    # Helper Methods
    # ========================================
    
    def to_dict(self):
        """Convert config to dictionary for saving."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: str):
        """
        Save configuration to JSON file.
        
        Args:
            path: Path to save the config file
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        print(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """
        Load configuration from JSON file.
        
        Args:
            path: Path to the config file
            
        Returns:
            TrainingConfig: Loaded configuration
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        print(f"Configuration loaded from {path}")
        return cls.from_dict(config_dict)
    
    @classmethod
    def load_or_default(cls, path: str):
        """
        Load config from file if it exists, otherwise return default.
        
        Args:
            path: Path to the config file
            
        Returns:
            TrainingConfig: Loaded or default configuration
        """
        if os.path.exists(path):
            return cls.load(path)
        else:
            print(f"Config file not found at {path}, using defaults")
            return cls()
    
    def get_num_actions_per_motor(self) -> int:
        """Calculate number of discrete actions per motor based on configuration."""
        if self.use_variable_step_sizes:
            # 2N + 1 actions: N for CCW, 1 for HOLD, N for CW
            return 2 * len(self.available_step_sizes) + 1
        else:
            # Legacy: 3 actions (CCW, HOLD, CW)
            return 3

    def get_hold_action_index(self) -> int:
        """Get the action index for HOLD."""
        return len(self.available_step_sizes)
        
    def print_summary(self):
        """Print a summary of the current configuration."""
        print("\n" + "="*60)
        print("TRAINING CONFIGURATION")
        print("="*60)
        
        print("\n[Environment]")
        print(f"  Motors: {self.num_motors}")
        print(f"  Early stopping threshold: {self.early_stopping_threshold}")
        print(f"  Max steps without improvement: {self.max_steps_without_improvement}")
        
        print("\n[Motor Control]")
        print(f"  Use motors: {self.use_motors}")
        print(f"  Speed: {self.motor_speed}, Reset speed: {self.motor_speed}")
        print(f"  Steps per action: {self.per_motor_steps}")
        print(f"  Step wait time: {self.step_wait_time}s")
        print(f"  Reset calibration mode: {self.reset_calibration}")
        
        print("\n[PPO Hyperparameters]")
        print(f"  Total timesteps: {self.total_timesteps}")
        print(f"  Learning rates: actor={self.lr_actor}, critic={self.lr_critic}")
        print(f"  Gamma: {self.gamma}, Lambda: {self.gae_lambda}")
        print(f"  Clip param: {self.clip_param}, Entropy coef: {self.entropy_coef}")
        
        print("\n[Network]")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Hold bias: {self.hold_bias}")
        
        print("\n[Logging]")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print(f"  Results dir: {self.results_dir}")
        print(f"  Save interval: {self.save_interval} episodes")

        print("\n[Reward Shaping]")
        print(f"  Target loss: {self.target_loss}")
        print(f"  Movement penalty: {self.initial_movement_penalty} → {self.final_movement_penalty}")
        print(f"  Penalty decay: {self.penalty_decay_episodes} episodes")
        
        print("\n[Position Safety]")
        print(f"  Danger zone: {self.danger_zone_ratio*100}% from limit")
        print(f"  Critical zone: {self.critical_zone_ratio*100}% from limit")
        print(f"  CCW safety margin: {self.ccw_safety_margin} steps")
        
        print("\n[Stagnation Detection]")
        print(f"  Threshold: {self.stagnation_threshold}")
        print(f"  Window: {self.stagnation_window} steps")
        
        print("\n[PPO Exploration]")
        print(f"  Temperature: {self.initial_temperature} → {self.min_temperature}")
        print(f"  Decay rate: {self.temperature_decay}")
        print(f"  Gradient clipping: {self.max_grad_norm}")
        
        print("\n[Timeouts]")
        print(f"  Motor completion: {self.motor_completion_timeout}s")
        print(f"  Stabilization: {self.stabilization_time}s")
        print(f"  Loss clipping: {self.loss_clip_max}")

        print("\n[Memory Management]")
        print(f"  GC interval: {self.gc_interval} episodes")
        print(f"  History max length: {self.history_maxlen} episodes")
        print(f"  Spectral data max age: {self.spectral_data_max_age}s")
        
        print("\n[Learning Rate Schedule]")
        print(f"  Warmup episodes: {self.lr_warmup_episodes}")
        print(f"  After warmup: actor={self.lr_reduced_actor}, critic={self.lr_reduced_critic}")
        
        print("\n[Loss Processing]")
        print(f"  Averaging window factor: {self.averaging_window_factor}")
        print(f"  History buffer size: {self.loss_history_buffer_size}")

        print("\n[Reward Components]")
        enabled_components = []
        if self.use_improvement_bonus:
            enabled_components.append(f"improvement({self.improvement_bonus_weight}x)")
        if self.use_consistency_bonus:
            enabled_components.append(f"consistency({self.consistency_bonus_weight}x)")
        if self.use_breakthrough_bonus:
            enabled_components.append(f"breakthrough({self.breakthrough_bonus_weight}x)")
        if self.use_movement_penalty:
            enabled_components.append(f"movement_penalty({self.movement_penalty_weight}x)")
        if self.use_efficiency_bonus:
            enabled_components.append(f"efficiency({self.efficiency_bonus_weight}x)")
        if self.use_proximity_bonus:
            enabled_components.append("proximity")
        if self.use_stagnation_penalty:
            enabled_components.append("stagnation")
        
        print(f"  Enabled: {', '.join(enabled_components)}")
        
        if self.use_proximity_bonus:
            print(f"  Proximity thresholds: {self.proximity_threshold_very_close} / {self.proximity_threshold_close}")
        
        print("\n[Episode Behavior]")
        print(f"  Steps before breakthrough: {self.episode_steps_before_breakthrough}")
        print(f"  Movement penalty threshold: >{self.motors_for_movement_penalty} motors")
        print(f"  Observation space: [{self.observation_space_loss_min}, {self.observation_space_loss_max}]")

        print("\n[Advanced PPO]")
        print(f"  Value coefficient: {self.value_coef}")
        print(f"  Normalize advantages: {self.normalize_advantages}")
        print(f"  Use GAE: {self.use_gae}")
        print(f"  Update frequency: {self.ppo_update_frequency} episodes")
        
        print("\n[Action Distribution]")
        print(f"  Hold bias: {self.action_hold_initial_bias}")
        print(f"  Exploration decay: {self.action_exploration_decay}")
        
        print("\n[Episode Termination]")
        print(f"  Early stopping: {self.use_early_stopping}")
        print(f"  Truncation: {self.use_truncation}")
        print(f"  Min episode steps: {self.min_episode_steps}")
        if self.reward_threshold_for_early_stop:
            print(f"  Reward threshold: {self.reward_threshold_for_early_stop}")

        print("\n[StallGuard Configuration]")
        print(f"  Enabled: {self.enable_stallguard}")
        print(f"  Threshold: {self.stallguard_threshold} (0-255)")
        print(f"  Warnings before stop: {self.stallguard_warnings_before_stop}")
        if self.use_per_motor_stallguard:
            print(f"  Per-motor thresholds: {self.per_motor_stallguard}")
        
        print("\n[Motor Physical Parameters]")
        print(f"  Acceleration: {self.motor_acceleration} steps/s²")
        print(f"  Microsteps: {self.motor_microsteps}")
        print(f"  Current: {self.motor_current_ma} mA")
        print(f"  Mode: {'SpreadCycle' if self.enable_spreadcycle else 'StealthChop'}")
        
        print("\n[Motor Safety]")
        print(f"  Movement timeout: {self.motor_timeout_seconds}s")
        if self.auto_disable_on_stall:
            print(f"  Auto-disable after {self.stall_count_before_disable} stalls")
        
        print("\n[Logging]")
        print(f"  Level: {self.log_level}")
        print(f"  Motor details: {self.log_motor_details}")
        print(f"  Reward breakdown: {self.log_reward_breakdown}")
        print(f"  Action distribution: {self.log_action_distribution}")
        print(f"  Episode summary every: {self.log_episode_summary_freq} episodes")
        
        print("\n[Visualization]")
        print(f"  Plot save frequency: every {self.plot_save_frequency} episodes")
        print(f"  Plot smoothing: {self.plot_smoothing_window}")
        print(f"  Plot size: {self.plot_figsize_width}x{self.plot_figsize_height} @ {self.plot_dpi} DPI")
        print(f"  Save plots: {self.save_plots}")
        
        print("\n[Checkpointing]")
        print(f"  Save best only: {self.save_best_only}")
        print(f"  Keep all: {self.keep_all_checkpoints}")
        if not self.keep_all_checkpoints:
            print(f"  Max to keep: {self.max_checkpoints_to_keep}")

        print("\n[Experiment]")
        print(f"  Name: {self.experiment_name}")
        if self.experiment_tags:
            print(f"  Tags: {', '.join(self.experiment_tags)}")
        if self.experiment_notes:
            print(f"  Notes: {self.experiment_notes[:60]}...")
        if self.random_seed is not None:
            print(f"  Random seed: {self.random_seed}")
        
        print("\n[Audio Processing]")
        print(f"  STFT scales: {self.stft_scales}")
        print(f"  Window type: {self.stft_window_type}")
        print(f"  Hop divisor: {self.stft_hop_divisor}")
        print(f"  Use pyFFTW: {self.use_pyfftw}")
        print(f"  FFT threads: {self.fft_threads}")
        
        print("\n[Network Architecture]")
        print(f"  Layer normalization: {self.use_layernorm}")
        print(f"  Dropout rate: {self.dropout_rate}")
        print(f"  Actor layers: {self.actor_hidden_layers}")
        print(f"  Critic layers: {self.critic_hidden_layers}")
        print(f"  Activation: {self.activation_function}")
        
        if self.enable_per_motor_settings:
            print("\n[Per-Motor Settings]")
            print(f"  Custom speeds: {self.per_motor_speeds}")
            print(f"  Custom steps: {self.per_motor_steps}")
            disabled_motors = [i+1 for i, enabled in enumerate(self.per_motor_enabled) if not enabled]
            if disabled_motors:
                print(f"  Disabled motors: {disabled_motors}")
        
        print("\n[Advanced Training]")
        if self.use_learning_rate_schedule:
            print(f"  LR schedule: cosine annealing")
            print(f"  Warmup: {self.lr_schedule_warmup_steps} steps")
        if self.gradient_accumulation_steps > 1:
            print(f"  Gradient accumulation: {self.gradient_accumulation_steps} steps")
        
        print("="*60 + "\n")

# Convenience function for creating default config file
def create_default_config(path: str = "./config.json"):
    """
    Create a default configuration file.
    
    Args:
        path: Where to save the config file
    """
    config = TrainingConfig()
    config.save(path)
    print(f"Default configuration created at {path}")
    print("Edit this file to customize your training parameters.")


if __name__ == "__main__":
    # When run directly, create a default config file
    import sys
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "./config.json"
    
    create_default_config(path)