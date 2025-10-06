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
    early_stopping_threshold: float = 10.0
    max_steps_without_improvement: int = 300
    
    # ========================================
    # Motor Control
    # ========================================
    use_motors: bool = True
    manual_calibration: bool = False
    motor_speed: int = 200
    motor_reset_speed: int = 200
    motor_steps: int = 50
    step_wait_time: float = 1.0
    reset_wait_time: float = 0.3
    reset_calibration: int = 1  # 0: center, 1: random, 2: skip
    
    # Motor position limits (per motor)
    max_ccw_steps: List[int] = field(default_factory=lambda: [4000, 4000, 4000, 4000, 5000, 5000, 5000, 5000])
    max_cw_steps: List[int] = field(default_factory=lambda: [4500, 4500, 4500, 4500, 4500, 4500, 4500, 4500])
    limit_penalty: float = 0.0
    
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
    buffer_size: int = 1024
    
    # ========================================
    # PPO Hyperparameters
    # ========================================
    total_timesteps: int = 100000
    max_ep_length: int = 1024
    update_interval: int = 64
    batch_size: int = 32
    n_epochs: int = 10
    
    # Learning rates
    lr_actor: float = 5e-4
    lr_critic: float = 1e-5
    
    # PPO algorithm parameters
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    
    # ========================================
    # Network Architecture
    # ========================================
    hidden_size: int = 64
    hold_bias: float = 2.0
    
    # ========================================
    # Reward Configuration
    # ========================================
    reward_scale: float = 1.0

    # ========================================
    # Reward Shaping (NEW)
    # ========================================
    target_loss: float = 15.0                    # Baseline loss for reward calculation
    initial_movement_penalty: float = 1.0        # Early training movement penalty
    final_movement_penalty: float = 0.1          # Late training movement penalty
    penalty_decay_episodes: int = 50             # Episodes to decay penalty over
    
    # ========================================
    # Position Safety (NEW)
    # ========================================
    danger_zone_ratio: float = 0.1               # Start penalties at 10% from limit
    critical_zone_ratio: float = 0.02            # Severe penalties at 2% from limit
    ccw_safety_margin: int = 200                 # Safety margin before CCW limit
    
    # ========================================
    # Stagnation Detection (NEW)
    # ========================================
    stagnation_threshold: float = 0.5            # Loss std dev threshold for stagnation
    stagnation_window: int = 20                  # Window size for detecting stagnation
    
    # ========================================
    # PPO Exploration (NEW)
    # ========================================
    initial_temperature: float = 2.0             # Initial exploration temperature
    temperature_decay: float = 0.998             # Temperature decay rate per update
    min_temperature: float = 0.3                 # Minimum exploration temperature
    max_grad_norm: float = 0.5                   # Gradient clipping threshold
    
    # ========================================
    # Timeouts and Stability (NEW)
    # ========================================
    motor_completion_timeout: float = 30.0       # Motor completion wait timeout (seconds)
    stabilization_time: float = 2.0              # Wait after motor movement (seconds)
    loss_clip_max: float = 50.0                  # Maximum loss value clipping

     # ========================================
    # Memory Management (NEW)
    # ========================================
    gc_interval: int = 10                        # Episodes between garbage collection
    history_maxlen: int = 10000                  # Max episodes to keep in history
    spectral_data_max_age: float = 1.0          # Max age for spectral data (seconds)
    
    # ========================================
    # Learning Rate Schedule (NEW)
    # ========================================
    lr_warmup_episodes: int = 20                 # Episodes before LR reduction
    lr_reduced_actor: float = 3e-4               # Actor LR after warmup
    lr_reduced_critic: float = 3e-4              # Critic LR after warmup
    
    # ========================================
    # Loss Averaging (NEW)
    # ========================================
    averaging_window_factor: float = 1.0         # Multiplier for loss averaging window
    loss_history_buffer_size: int = 200          # Size of loss history buffer

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
    breakthrough_bonus_weight: float = 10.0      # Weight for breakthrough bonus
    movement_penalty_weight: float = 1.0         # Weight for movement penalty
    efficiency_bonus_weight: float = 0.5         # Weight for efficiency bonus
    
    # Proximity bonus thresholds
    proximity_threshold_close: float = 0.5       # "Getting close" threshold
    proximity_threshold_very_close: float = 0.1  # "Very close" threshold
    proximity_bonus_close: float = 10.0          # Bonus for getting close
    proximity_bonus_very_close: float = 50.0     # Bonus for very close
    
    # ========================================
    # Episode Behavior (NEW)
    # ========================================
    episode_steps_before_breakthrough: int = 30  # Min steps before breakthrough bonus
    motors_for_movement_penalty: int = 2         # Penalize if more than this many move
    observation_space_loss_max: float = 100.0    # Max loss for observation space
    observation_space_loss_min: float = 0.0      # Min loss for observation space

    # ========================================
    # Advanced PPO Parameters (NEW)
    # ========================================
    value_coef: float = 0.5                      # Value function loss coefficient
    normalize_advantages: bool = True            # Normalize advantages during training
    normalize_returns: bool = False              # Normalize returns (experimental)
    use_gae: bool = True                         # Use Generalized Advantage Estimation
    ppo_update_frequency: int = 1                # Episodes between PPO updates
    
    # ========================================
    # Action Distribution (NEW)
    # ========================================
    action_hold_initial_bias: float = 2.0        # Initial bias toward HOLD action
    action_exploration_decay: bool = True        # Decay exploration over time
    
    # ========================================
    # Episode Termination (NEW)
    # ========================================
    use_early_stopping: bool = True              # Enable early stopping on good loss
    use_truncation: bool = True                  # Enable truncation on stagnation
    min_episode_steps: int = 1                   # Minimum steps per episode
    reward_threshold_for_early_stop: float = None  # Stop episode if reward exceeds this
    
    # ========================================
    # Logging Control (NEW)
    # ========================================
    log_level: str = "INFO"                      # Logging level: DEBUG, INFO, WARNING, ERROR
    log_motor_details: bool = False              # Log detailed motor movements
    log_reward_breakdown: bool = False           # Log reward component breakdown
    log_action_distribution: bool = False        # Log action distribution stats
    log_loss_processor_stats: bool = False       # Log loss processor performance
    
    # Logging frequencies (episodes)
    log_episode_summary_freq: int = 1            # Log episode summary every N episodes
    log_position_status_freq: int = 50           # Log motor positions every N episodes
    log_performance_stats_freq: int = 100        # Log performance stats every N episodes
    
    # ========================================
    # Visualization (NEW)
    # ========================================
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
        print(f"  Speed: {self.motor_speed}, Reset speed: {self.motor_reset_speed}")
        print(f"  Steps per action: {self.motor_steps}")
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
        
        print("\n[Logging]")
        print(f"  Level: {self.log_level}")
        print(f"  Motor details: {self.log_motor_details}")
        print(f"  Reward breakdown: {self.log_reward_breakdown}")
        print(f"  Action distribution: {self.log_action_distribution}")
        print(f"  Episode summary every: {self.log_episode_summary_freq} episodes")
        
        print("\n[Visualization]")
        print(f"  Plot smoothing: {self.plot_smoothing_window}")
        print(f"  Plot size: {self.plot_figsize_width}x{self.plot_figsize_height} @ {self.plot_dpi} DPI")
        print(f"  Save plots: {self.save_plots}")
        
        print("\n[Checkpointing]")
        print(f"  Save best only: {self.save_best_only}")
        print(f"  Keep all: {self.keep_all_checkpoints}")
        if not self.keep_all_checkpoints:
            print(f"  Max to keep: {self.max_checkpoints_to_keep}")
        
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