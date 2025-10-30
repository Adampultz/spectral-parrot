# Enhanced main_motor_training.py with checkpoint resume functionality
from __future__ import annotations
import logging
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import signal
import json
# import pickle
import gc
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any
from pythonosc import udp_client
from datetime import datetime
from collections import deque
from config import TrainingConfig
from save_session_hyperparameters import save_session_hyperparameters


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("motor_training.log")
    ]
)
logger = logging.getLogger(__name__)

# Import simplified components
from stft_audio import EnhancedAudio
from simple_spectral_loss_processor import SimpleLossProcessor
from motor_environment import MotorEnvironment
from motor_ppo_agent import MotorPPOAgent
from Stepper_Control import DualESP32StepperController
from osc_handler import OSCHandler, setup_signal_handlers

# Global for signal handling
env = None
osc_handler = None

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    global env, osc_handler, loss_processor, audio
    
    logger.info("Received shutdown signal, cleaning up...")

        # Pause loss processor to suppress warnings during shutdown
    if 'loss_processor' in globals() and loss_processor is not None:  # <-- ADD THIS
        loss_processor.pause()
    
    # Stop audio before closing environment
    if 'audio' in globals() and audio is not None:  # <-- ADD THIS
        try:
            audio.stop()
        except Exception as e:
            logger.error(f"Error stopping audio: {e}")
            
    if env is not None:
        env.close()
    if 'osc_handler' in globals():
        try:
            logger.info("Cleaning up OSC handler...")
            osc_handler.cleanup()
        except Exception as e:
            logger.error(f"Error during OSC cleanup: {e}")
    sys.exit(0)

def setup_session_logging(config: TrainingConfig, training_state: TrainingState, resumed: bool = False) -> str:
    """
    Set up logging for this training session with a unique log file.
    
    Args:
        config: Training configuration
        training_state: Current training state (for episode number)
        resumed: Whether this is a resumed session
        
    Returns:
        str: Path to the log file for this session
    """
    # Create logs directory structure
    experiment_log_dir = os.path.join(config.log_dir, config.experiment_name)
    os.makedirs(experiment_log_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Generate log filename
    suffix = "_resumed" if resumed else ""
    log_filename = f"{timestamp}_ep{training_state.episode}{suffix}.log"
    session_log_path = os.path.join(experiment_log_dir, log_filename)
    
    # Remove existing handlers to reconfigure
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create new handlers
    handlers = [
        logging.StreamHandler(sys.stdout),  # Console output
        logging.FileHandler(session_log_path, mode='w')  # Session-specific log
    ]
    
    # Optionally add master log handler
    if config.log_to_master:
        master_log_path = os.path.join(config.log_dir, "master.log")
        handlers.append(logging.FileHandler(master_log_path, mode='a'))
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=handlers,
        force=True  # Force reconfiguration
    )
    
    # Log session start information
    logger.info("="*80)
    logger.info(f"NEW TRAINING SESSION: {config.experiment_name}")
    logger.info(f"Log file: {session_log_path}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Starting episode: {training_state.episode}")
    if resumed:
        logger.info("This is a RESUMED session")
    if config.experiment_tags:
        logger.info(f"Tags: {', '.join(config.experiment_tags)}")
    if config.experiment_notes:
        logger.info(f"Notes: {config.experiment_notes}")
    logger.info("="*80)
    
    return session_log_path
class TrainingState:
    """Class to manage training state for checkpointing."""
    
    def __init__(self, max_history=10000):
        self.episode = 0
        self.timesteps = 0
        self.best_reward = -float('inf')
        # Is it a problem that they grow forever?/
        # self.episode_rewards = []  
        # self.episode_lengths = []   
        # self.episode_losses = []   
        self.episode_rewards = deque(maxlen=max_history)
        self.episode_lengths = deque(maxlen=max_history)
        self.episode_losses = deque(maxlen=max_history)
        self.training_start_time = time.time()
        
    def to_dict(self):
        """Convert state to dictionary for saving."""
        return {
            'episode': self.episode,
            'timesteps': self.timesteps,
            'best_reward': self.best_reward,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_losses': self.episode_losses,
            'training_start_time': self.training_start_time
        }
    
    @classmethod
    def from_dict(cls, state_dict):
        """Create state from dictionary."""
        state = cls()
        state.episode = state_dict['episode']
        state.timesteps = state_dict['timesteps']
        state.best_reward = state_dict['best_reward']
        state.episode_rewards = state_dict['episode_rewards']
        state.episode_lengths = state_dict['episode_lengths']
        state.episode_losses = state_dict['episode_losses']
        state.training_start_time = state_dict['training_start_time']
        return state


def save_checkpoint(agent: MotorPPOAgent, 
                    training_state: TrainingState,
                    config: TrainingConfig,
                    checkpoint_path: str,
                    is_best: bool = False):
    """
    Save a comprehensive training checkpoint.
    
    Args:
        agent: The PPO agent
        training_state: Current training state
        config: Training configuration
        checkpoint_path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        # Agent state
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
        'temperature': agent.current_temperature,
        
        # Training state
        'training_state': training_state.to_dict(),
        
        # Configuration
        'config': config.to_dict(),
        
        # Metadata
        'timestamp': datetime.now().isoformat(),
        'is_best': is_best,
        'pytorch_version': torch.__version__,
        'checkpoint_version': '1.0',

        # Experiment metadata
        'experiment_name': config.experiment_name,
        'experiment_tags': config.experiment_tags,
        'experiment_notes': config.experiment_notes,
        'random_seed': config.random_seed,
    }
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Also save a separate JSON file with training metrics for easy inspection
    metrics_path = checkpoint_path.replace('.pt', '_metrics.json')
    metrics = {
        'episode': training_state.episode,
        'timesteps': training_state.timesteps,
        'best_reward': training_state.best_reward,
        'latest_rewards': list(training_state.episode_rewards)[-10:] if training_state.episode_rewards else [],
        'latest_losses': list(training_state.episode_losses)[-10:] if training_state.episode_losses else [],
        'timestamp': datetime.now().isoformat()
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")


def load_checkpoint(checkpoint_path: str, 
                   agent: Optional[MotorPPOAgent] = None,
                   device: str = 'cpu') -> tuple:
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        agent: Existing agent to load into (optional)
        device: Device to load onto
        
    Returns:
        (agent, training_state, config): Loaded components
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load configuration
    config = TrainingConfig.from_dict(checkpoint['config'])
    
    # Create or update agent
    if agent is None:
        agent = MotorPPOAgent(
            state_dim=2,
            num_motors=config.num_motors,
            num_actions_per_motor=config.get_num_actions_per_motor(),
            device=device,
            hidden_size=config.hidden_size,
            lr_actor=config.lr_actor,
            lr_critic=config.lr_critic,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_param=config.clip_param,
            entropy_coef=config.entropy_coef,
            batch_size=config.batch_size,
            hold_bias=config.hold_bias,
            step_size_logits_bias=config.step_size_logits_bias,
            max_grad_norm=config.max_grad_norm,
            initial_temperature=config.initial_temperature,
            temperature_decay=config.temperature_decay,
            min_temperature=config.min_temperature,
            value_coef=config.value_coef,               
            normalize_advantages=config.normalize_advantages, 
            normalize_returns=config.normalize_returns,       
            use_gae=config.use_gae,
            actor_hidden_layers=config.actor_hidden_layers,      
            critic_hidden_layers=config.critic_hidden_layers,   
            use_layernorm=config.use_layernorm,                 
            dropout_rate=config.dropout_rate,                  
            activation=config.activation_function,
        )
    
    # Load agent state
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    agent.current_temperature = checkpoint.get('temperature', 1.0)
    agent.actor.set_temperature(agent.current_temperature)
    
    # Load training state
    training_state = TrainingState.from_dict(checkpoint['training_state'])
    
    logger.info(f"Loaded checkpoint from episode {training_state.episode}, "
               f"timesteps {training_state.timesteps}")
    logger.info(f"Best reward so far: {training_state.best_reward:.2f}")
    
    return agent, training_state, config


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the most recent checkpoint in the directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) 
                  if f.endswith('.pt') and not f.startswith('best')]
    
    if not checkpoints:
        return None
    
    # Sort by modification time
    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    
    latest = os.path.join(checkpoint_dir, checkpoints[-1])
    logger.info(f"Found latest checkpoint: {latest}")
    
    return latest


def create_directories(config: TrainingConfig):
    """Create necessary directories."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(os.path.join(config.results_dir, "hyperparameters"), exist_ok=True) 

def train(config: TrainingConfig, resume_from: Optional[str] = None):
    """
    Main training function with checkpoint resume support.
    
    Args:
        config: Training configuration
        resume_from: Path to checkpoint to resume from (optional)
    """
    global env, osc_handler

    # Set random seed for reproducibility
    if config.random_seed is not None:
        import random
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.random_seed)
        logger.info(f"Random seed set to {config.random_seed}")
    
    # Set logging level
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    if config.use_variable_step_sizes:
        logger.info(f"Using variable step sizes: {config.available_step_sizes}")
        logger.info(f"Actions per motor: {2 * len(config.available_step_sizes) + 1}")
    else:
        logger.info(f"Using fixed step size: {config.per_motor_steps}")
    
    # Print experiment info
    logger.info(f"Experiment: {config.experiment_name}")
    if config.experiment_tags:
        logger.info(f"Tags: {', '.join(config.experiment_tags)}")
    if config.experiment_notes:
        logger.info(f"Notes: {config.experiment_notes}")

    config.print_summary()

    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logger.info(f"Logging level set to {config.log_level}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create directories
    create_directories(config)
    
    # Initialize training state
    training_state = TrainingState(max_history=config.history_maxlen)
    agent = None
    resumed = False
    
    # Resume from checkpoint if specified
    if resume_from:
        if resume_from == "latest":
            resume_from = find_latest_checkpoint(config.checkpoint_dir)
            if not resume_from:
                logger.warning("No checkpoint found to resume from. Starting fresh.")
        
        if resume_from and os.path.exists(resume_from):
            try:
                agent, training_state, loaded_config = load_checkpoint(resume_from, device=device)
                
                resumed = True
                # Optionally override config with loaded config
                # config = loaded_config  # Uncomment to use saved config
                
                logger.info(f"Resumed from checkpoint: {resume_from}")
                logger.info(f"Continuing from episode {training_state.episode}, "
                           f"timestep {training_state.timesteps}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                logger.info("Starting fresh training")
                agent = None
                training_state = TrainingState()
        else:
            logger.warning(f"Checkpoint not found: {resume_from}")
    
    session_log_path = setup_session_logging(config, training_state, resumed=resumed)
    logger.info(f"Session log: {session_log_path}")

    hyperparams_file = save_session_hyperparameters(
        config=config,
        training_state=training_state,
        hyperparams_dir=os.path.join(config.results_dir, "hyperparameters"),
        resumed=resumed,
        resumed_from=resume_from if resumed else None
    )
    logger.info(f"✓ Hyperparameters saved to: {hyperparams_file}")
    
    # 1. Setup audio system
    logger.info("Setting up audio system...")
    audio = EnhancedAudio(
        sample_rate=config.sample_rate,
        channels=config.channels,
        buffer_size=config.buffer_size,
        input_device=config.input_device,
        enable_spectral_loss=True,
        stft_scales=config.stft_scales,        
        stft_window_type=config.stft_window_type,  
        use_pyfftw=config.use_pyfftw,          
        fft_threads=config.fft_threads,
        use_normalized_loss=config.use_normalized_loss,
        min_signal_threshold=config.min_signal_threshold,
        weak_signal_penalty=config.weak_signal_penalty,
        normalization_method=config.normalization_method        
    )
    
    # 2. Create loss processor
    logger.info("Creating loss processor...")
    loss_processor = SimpleLossProcessor(
        spectral_loss_calculator=audio.spectral_loss,
        device=device,
        step_wait_time=config.step_wait_time,
        loss_clip_max=config.loss_clip_max,
        averaging_window_factor=config.averaging_window_factor,
        loss_history_buffer_size=config.loss_history_buffer_size
    )

    loss_processor.pause() # Pause the processor immediately after creation to avoid weak signal warnings during initialization

    osc_handler = OSCHandler()
    
    # 3. Setup motor controller
    motor_controller = None
    if config.use_motors:
        logger.info(f"Connecting to motors on {config.port1} and {config.port2}...")
        motor_controller = DualESP32StepperController(
            config.port1, config.port2, config.baudrate, debug=False
        )
        if not motor_controller.connect():
            logger.warning("Failed to connect to motors, continuing without motor control")
            config.use_motors = False
    
    # 4. Create environment
    logger.info("Creating motor environment...")
    env = MotorEnvironment(
        loss_processor=loss_processor,
        motor_controller=motor_controller,
        num_motors=config.num_motors,
        motor_steps=config.per_motor_steps,
        use_variable_step_sizes=config.use_variable_step_sizes,
        available_step_sizes=config.available_step_sizes,
        step_size_logits_bias=getattr(config, 'step_size_logits_bias', None),
        step_wait_time=config.step_wait_time,
        reset_wait_time=config.reset_wait_time,
        early_stopping_threshold=config.early_stopping_threshold,
        max_steps_without_improvement=config.max_steps_without_improvement,
        use_motors=config.use_motors,
        motor_speed=config.motor_speed,
        reward_scale=config.reward_scale,
        max_ccw_steps=config.max_ccw_steps,
        max_cw_steps=config.max_cw_steps,
        limit_penalty=config.limit_penalty,
        adaptive_hold_bias=config.hold_bias,
        manual_calibration=config.manual_calibration,
        reset_calibration = config.reset_calibration,
        target_loss=config.target_loss,
        initial_movement_penalty=config.initial_movement_penalty,
        final_movement_penalty=config.final_movement_penalty,
        penalty_decay_episodes=config.penalty_decay_episodes,
        danger_zone_ratio=config.danger_zone_ratio,
        critical_zone_ratio=config.critical_zone_ratio,
        ccw_safety_margin=config.ccw_safety_margin,
        stagnation_threshold=config.stagnation_threshold,
        stagnation_window=config.stagnation_window,
        motor_completion_timeout=config.motor_completion_timeout,
        stabilization_time=config.stabilization_time,
        use_improvement_bonus=config.use_improvement_bonus,
        use_consistency_bonus=config.use_consistency_bonus,
        use_breakthrough_bonus=config.use_breakthrough_bonus,
        use_movement_penalty=config.use_movement_penalty,
        use_stagnation_penalty=config.use_stagnation_penalty,
        use_efficiency_bonus=config.use_efficiency_bonus,
        use_proximity_bonus=config.use_proximity_bonus,
        improvement_bonus_weight=config.improvement_bonus_weight,
        consistency_bonus_weight=config.consistency_bonus_weight,
        breakthrough_bonus_weight=config.breakthrough_bonus_weight,
        movement_penalty_weight=config.movement_penalty_weight,
        efficiency_bonus_weight=config.efficiency_bonus_weight,
        proximity_threshold_close=config.proximity_threshold_close,
        proximity_threshold_very_close=config.proximity_threshold_very_close,
        proximity_bonus_close=config.proximity_bonus_close,
        proximity_bonus_very_close=config.proximity_bonus_very_close,
        episode_steps_before_breakthrough=config.episode_steps_before_breakthrough,
        motors_for_movement_penalty=config.motors_for_movement_penalty,
        observation_space_loss_max=config.observation_space_loss_max,
        observation_space_loss_min=config.observation_space_loss_min,
        use_early_stopping=config.use_early_stopping,      
        use_truncation=config.use_truncation,               
        min_episode_steps=config.min_episode_steps,    
        max_ep_length=config.max_ep_length,
        reward_threshold_for_early_stop=config.reward_threshold_for_early_stop,  
        log_motor_details=config.log_motor_details,
        stallguard_threshold=config.stallguard_threshold,
        stallguard_warnings_before_stop=config.stallguard_warnings_before_stop,
        use_per_motor_stallguard=config.use_per_motor_stallguard,
        per_motor_stallguard=config.per_motor_stallguard,
        motor_acceleration=config.motor_acceleration,
        motor_current_ma=config.motor_current_ma,
        enable_stallguard=config.enable_stallguard,
        initial_episode=training_state.episode if resumed else 0          
    )

    # Create signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 5. Create PPO agent if not loaded from checkpoint
    if agent is None:
        logger.info("Creating new PPO agent...")
        agent = MotorPPOAgent(
            state_dim=2,
            num_motors=config.num_motors,
            num_actions_per_motor=config.get_num_actions_per_motor(),
            device=device,
            hidden_size=config.hidden_size,
            lr_actor=config.lr_actor,
            lr_critic=config.lr_critic,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_param=config.clip_param,
            entropy_coef=config.entropy_coef,
            batch_size=config.batch_size,
            hold_bias=config.hold_bias,
            step_size_logits_bias=config.step_size_logits_bias,
            max_grad_norm=config.max_grad_norm,
            initial_temperature=config.initial_temperature,
            temperature_decay=config.temperature_decay,
            min_temperature=config.min_temperature,
            value_coef=config.value_coef,               
            normalize_advantages=config.normalize_advantages, 
            normalize_returns=config.normalize_returns,       
            use_gae=config.use_gae,
            actor_hidden_layers=config.actor_hidden_layers,      
            critic_hidden_layers=config.critic_hidden_layers,   
            use_layernorm=config.use_layernorm,                 
            dropout_rate=config.dropout_rate,                  
            activation=config.activation_function            
        )

    # Start OSC server
    logger.info("Starting OSC server")
    osc_handler.start()
    
    # Start audio
    logger.info("Starting audio system...")
    audio.start()

    logger.info("Unpausing loss processor...")
    loss_processor.unpause()
    
    # Wait for loss processor to be ready
    logger.info("Waiting for loss processor...")
    start_time = time.time()
    while not loss_processor.is_ready() and time.time() - start_time < 15:
        time.sleep(0.5)
    
    if not loss_processor.is_ready():
        logger.error("Loss processor not ready after timeout")
        return
    
    logger.info("Loss processor ready!")
    
    # 6. Training loop (continuing from checkpoint if loaded)
    logger.info(f"Starting/Resuming training from episode {training_state.episode + 1}...")
    
    while training_state.timesteps < config.total_timesteps:
        training_state.episode += 1
        obs, _ = env.reset()
        
        episode_reward = 0
        episode_loss_sum = 0
        
        for step in range(config.max_ep_length):
            # Select action
            actions, log_prob, value = agent.select_action(obs)
            
            # Take action
            next_obs, reward, terminated, truncated, info = env.step(actions)

            if 'recommended_hold_bias' in info:
                agent.actor.hold_bias = info['recommended_hold_bias']
            
            # Store transition
            agent.store_transition(obs, actions, log_prob, reward, value, terminated or truncated)
            
            # Update metrics
            training_state.timesteps += 1
            episode_reward += reward
            episode_loss_sum += info['spectral_loss']
            
            # Log step details based on config
            if step % config.log_frequency == 0 and config.log_motor_details:
                if config.use_variable_step_sizes and 'movement_details' in info:
                    # Enhanced logging with step sizes
                    movements = []
                    for motor_num in sorted(info['motors_moved']):
                        direction, step_size = info['movement_details'][motor_num]
                        # Convert direction number to string
                        dir_str = 'CCW' if direction == -1 else 'CW' if direction == 1 else 'HOLD'
                        movements.append(f"{motor_num}:{dir_str}{step_size}")  # ✅ CORRECT
                    motors_str = f"[{', '.join(movements)}]" if movements else "[]"
                else:
                    # Legacy logging
                    motors_str = str(info['motors_moved'])
                
                logger.info(f"Episode {training_state.episode}, Step {step}: "
                        f"Loss={info['spectral_loss']:.4f}, "
                        f"Reward={reward:.2f}, Motors moved: {motors_str}")
            
            # Move to next state
            obs = next_obs
            
            # Update policy
            if training_state.timesteps % config.update_interval == 0:
                with torch.no_grad():
                    next_value = agent.critic(torch.FloatTensor(obs).to(device)).item()
                
                actor_loss, critic_loss = agent.update(next_value, config.n_epochs)
                
                logger.info(f"Update at timestep {training_state.timesteps}: "
                          f"Actor loss={actor_loss:.4f}, Critic loss={critic_loss:.4f}, "
                          f"Temperature={agent.current_temperature:.3f}")
            
            # Check if done
            if terminated or truncated:
                break
        
        # Episode complete
        ep_length = step + 1
        avg_loss = episode_loss_sum / ep_length

        if training_state.episode % config.gc_interval == 0:
            gc.collect()
            # Also clean up spectral data
            if hasattr(audio.spectral_loss, 'cleanup_old_spectral_data'):
                audio.spectral_loss.cleanup_old_spectral_data(max_age=config.spectral_data_max_age)
            logger.debug(f"Garbage collection at episode {training_state.episode}")
                
        training_state.episode_rewards.append(episode_reward)
        training_state.episode_lengths.append(ep_length)
        training_state.episode_losses.append(avg_loss)
        
        # Calculate rolling averages
        window = min(10, len(training_state.episode_rewards))
        avg_reward = np.mean(list(training_state.episode_rewards)[-window:])
        avg_ep_loss = np.mean(list(training_state.episode_losses)[-window:])
        
        if training_state.episode % config.log_episode_summary_freq == 0:
            logger.info(f"Episode {training_state.episode}: Reward={episode_reward:.2f}, "
                    f"Length={ep_length}, Loss={avg_loss:.4f}, "
                    f"Avg Reward={avg_reward:.2f}, Avg Loss={avg_ep_loss:.4f}, "
                    f"Total Timesteps={training_state.timesteps}")
        
        # Position status logging
        if config.log_action_distribution and training_state.episode % config.log_position_status_freq == 0:
            pos_status = env.get_position_status()
            logger.info("Motor position status:")
            for status in pos_status:
                logger.info(f"  Motor {status['motor']}: pos={status['confirmed_position']}, "
                        f"uncertain={status['uncertain']}")
            
        # Adjust learning rate after warm-up
        if training_state.episode == config.lr_warmup_episodes + 1:
            agent.actor_optimizer.param_groups[0]['lr'] = config.lr_reduced_actor
            agent.critic_optimizer.param_groups[0]['lr'] = config.lr_reduced_critic
            logger.info(f"Reduced learning rates after {config.lr_warmup_episodes} episodes: "
                        f"actor={config.lr_reduced_actor}, critic={config.lr_reduced_critic}")
        
        # Render environment occasionally
        if training_state.episode % config.plot_frequency == 0:
            env.render()
        
        # Save checkpoint
        if training_state.episode % config.save_interval == 0:
            checkpoint_name = f"checkpoint_ep{training_state.episode}_ts{training_state.timesteps}_r{avg_reward:.2f}.pt"
            checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_name)
            
            is_best = avg_reward > training_state.best_reward
            save_checkpoint(agent, training_state, config, checkpoint_path, is_best)
            
            # Update best model if needed
            if is_best:
                training_state.best_reward = avg_reward
                best_path = os.path.join(config.checkpoint_dir, "best_model.pt")
                save_checkpoint(agent, training_state, config, best_path, is_best=True)
                logger.info(f"New best model! Avg reward: {training_state.best_reward:.2f}")
        
        # Plot progress
        if training_state.episode % config.plot_save_frequency == 0:
            plot_training_progress(
                training_state.episode_rewards,
                training_state.episode_losses,
                training_state.episode_lengths,
                save_path=os.path.join(config.results_dir, 
                                       f"progress_ep{training_state.episode}.png"),
                config=config
            )
    
    # Training complete
    logger.info("Training complete!")
    
    # Save final checkpoint
    final_path = os.path.join(config.checkpoint_dir, "final_model.pt")
    save_checkpoint(agent, training_state, config, final_path)
    
    # Final statistics
    logger.info(f"Final statistics:")
    logger.info(f"  Total episodes: {training_state.episode}")
    logger.info(f"  Total timesteps: {training_state.timesteps}")
    logger.info(f"  Best average reward: {training_state.best_reward:.2f}")
    logger.info(f"  Final average loss: {np.mean(training_state.episode_losses[-10:]):.4f}")
    
    # Get action distribution stats
    action_stats = env.get_action_distribution_stats()
    if action_stats:
        logger.info("\nAction distribution:")
        for motor in range(config.num_motors):
            ccw_pct = action_stats['action_percentages'][motor, 0]
            hold_pct = action_stats['action_percentages'][motor, 1]
            cw_pct = action_stats['action_percentages'][motor, 2]
            logger.info(f"  Motor {motor+1}: CCW={ccw_pct:.1f}%, "
                       f"HOLD={hold_pct:.1f}%, CW={cw_pct:.1f}%")
    
    # Cleanup
    logger.info("Cleaning up resources...")
    env.close()
    audio.stop()

    if 'osc_handler' in locals():
        logger.info("Stopping SuperCollider synths...")
        osc_handler.cleanup()


def plot_training_progress(rewards, losses, lengths, save_path, config):
    """Plot training progress with configurable parameters."""
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, 
        figsize=(config.plot_figsize_width, config.plot_figsize_height)
    )
    
    # Episode rewards
    ax1.plot(rewards, alpha=0.6)
    if len(rewards) > config.plot_smoothing_window:
        smoothed = np.convolve(
            rewards, 
            np.ones(config.plot_smoothing_window) / config.plot_smoothing_window, 
            mode='valid'
        )
        ax1.plot(range(len(smoothed)), smoothed, 'r-', linewidth=2)
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Training Progress')
    ax1.grid(True, alpha=0.3)
    
    # Spectral loss
    ax2.plot(losses, alpha=0.6)
    if len(losses) > config.plot_smoothing_window:
        smoothed = np.convolve(
            losses, 
            np.ones(config.plot_smoothing_window) / config.plot_smoothing_window, 
            mode='valid'
        )
        ax2.plot(range(len(smoothed)), smoothed, 'r-', linewidth=2)
    ax2.set_ylabel('Average Spectral Loss')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Episode lengths
    ax3.plot(lengths, alpha=0.6)
    ax3.set_ylabel('Episode Length')
    ax3.set_xlabel('Episode')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if config.save_plots:
        plt.savefig(save_path, dpi=config.plot_dpi)
        logger.info(f"✓ Plot saved to {save_path}")
    
    plt.close()


def list_checkpoints(checkpoint_dir: str):
    """List all available checkpoints."""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist")
        return
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    
    if not checkpoints:
        print("No checkpoints found")
        return
    
    print("\nAvailable checkpoints:")
    print("-" * 60)
    
    for checkpoint in sorted(checkpoints):
        path = os.path.join(checkpoint_dir, checkpoint)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        mod_time = datetime.fromtimestamp(os.path.getmtime(path))
        
        # Try to load metrics if available
        metrics_path = path.replace('.pt', '_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                print(f"{checkpoint:40} | {size_mb:6.2f} MB | {mod_time:%Y-%m-%d %H:%M}")
                print(f"  Episode: {metrics.get('episode', 'N/A'):5} | "
                     f"Timesteps: {metrics.get('timesteps', 'N/A'):8} | "
                     f"Best Reward: {metrics.get('best_reward', 'N/A'):.2f}")
        else:
            print(f"{checkpoint:40} | {size_mb:6.2f} MB | {mod_time:%Y-%m-%d %H:%M}")
    
    print("-" * 60)


def main():
    """Main entry point with resume support."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Motor control training with MSSL')

    # Config file argument (highest priority)
    parser.add_argument('--config', type=str, help='Path to config file (JSON)')
    
    # Audio arguments
    parser.add_argument('--input-device', type=int, help='Input audio device index')
    parser.add_argument('--list-devices', action='store_true', help='List audio devices')
    
    # Motor arguments
    parser.add_argument('--port1', type=str, default="/dev/cu.usbserial-0001",
                       help='Serial port for ESP32 #1 (odd motors)')
    parser.add_argument('--port2', type=str, default="/dev/cu.usbserial-1",
                       help='Serial port for ESP32 #2 (even motors)')
    parser.add_argument('--no-motors', action='store_true', help='Disable motor control')
    
    # Training arguments
    parser.add_argument('--episodes', type=int, help='Number of episodes')
    parser.add_argument('--timesteps', type=int, help='Total timesteps')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--hidden-size', type=int, help='Hidden layer size')
    
    # Checkpoint arguments
    parser.add_argument('--resume', type=str, help='Resume from checkpoint (path or "latest")')
    parser.add_argument('--list-checkpoints', action='store_true', 
                       help='List available checkpoints and exit')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory for checkpoints')
    
    # Other arguments
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--save-config', type=str, help='Save current config to file and exit')

    # Motor calibration
    parser.add_argument('--skip-calibration', action='store_true', 
                   help='Skip initial motor calibration')
    parser.add_argument('--calibration-mode', type=int, choices=[0, 1, 2],
                   help='Calibration mode: 0=center, 1=random (default), 2=skip')  
    parser.add_argument('--step-sizes', type=int, nargs='+',
                       help='Available step sizes (e.g., --step-sizes 25 50 75 100)')
    parser.add_argument('--disable-variable-steps', action='store_true',
                       help='Use fixed motor_steps instead')
    
    args = parser.parse_args()
    
    # List devices if requested
    if args.list_devices:
        from stft_audio import SimpleAudio
        SimpleAudio.list_available_devices()
        return
    
    # List checkpoints if requested
    if args.list_checkpoints:
        list_checkpoints(args.checkpoint_dir)
        return
    
    # Set debug level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load config (priority: --config file > default file > hardcoded defaults)
    if args.config:
        config = TrainingConfig.load(args.config)
    else:
        config = TrainingConfig.load_or_default("./config.json")
    
    # Override with command line arguments
    if args.input_device is not None:
        config.input_device = args.input_device
    if args.port1:
        config.port1 = args.port1
    if args.port2:
        config.port2 = args.port2
    if args.no_motors:
        config.use_motors = False
    if args.episodes:
        config.total_timesteps = args.episodes * config.max_ep_length
    if args.timesteps:
        config.total_timesteps = args.timesteps
    if args.lr:
        config.lr_actor = args.lr
        config.lr_critic = args.lr
    if args.hidden_size:
        config.hidden_size = args.hidden_size
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.skip_calibration:
        config.reset_calibration = 2
    if args.calibration_mode is not None:
        config.reset_calibration = args.calibration_mode
    if args.step_sizes:
        config.use_variable_step_sizes = True
        config.available_step_sizes = args.step_sizes   
    if args.disable_variable_steps:
        config.use_variable_step_sizes = False

    # Save config if requested
    if args.save_config:
        config.save(args.save_config)
        return
    
    # Run training with resume support
    train(config, resume_from=args.resume)      

if __name__ == "__main__":
    main()