# Enhanced main_motor_training.py with checkpoint resume functionality

import logging
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import signal
import json
import pickle
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any
from pythonosc import udp_client
from datetime import datetime
from collections import deque
import gc

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

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    logger.info("Received shutdown signal, cleaning up...")
    if env is not None:
        env.close()
    if 'osc_handler' in globals():
        try:
            logger.info("Cleaning up OSC handler...")
            osc_handler.cleanup()
        except Exception as e:
            logger.error(f"Error during OSC cleanup: {e}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Training configuration
@dataclass
class TrainingConfig:
    # Environment
    num_motors: int = 8
    early_stopping_threshold: float = 10
    max_steps_without_improvement: int = 120
    
    # Motor control
    use_motors: bool = True
    motor_speed: int = 200
    motor_reset_speed: int = 200
    motor_steps: int = 500
    step_wait_time: float = 1.5
    reset_wait_time: float = 0.3
    max_ccw_steps: List[int] = field(default_factory=lambda: [3000, 3000, 3000, 5000, 5000, 5000, 5000, 5000])
    max_cw_steps: List[int] = field(default_factory=lambda: [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000])
    limit_penalty: float = 0.0
    
    # Serial ports
    port1: str = "/dev/cu.usbserial-0001"
    port2: str = "/dev/cu.usbserial-1"
    baudrate: int = 115200
    
    # Audio
    input_device: Optional[int] = None
    sample_rate: int = 48000
    channels: int = 2
    buffer_size: int = 1024
    
    # PPO hyperparameters
    total_timesteps: int = 100000
    max_ep_length: int = 512
    update_interval: int = 64
    batch_size: int = 32
    n_epochs: int = 4
    
    # Learning rates
    lr_actor: float = 8e-4
    lr_critic: float = 8e-4
    
    # PPO specific
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    
    # Network
    hidden_size: int = 64
    
    # Reward
    reward_scale: float = 1.0
    
    # Saving
    save_interval: int = 1
    plot_frequency: int = 1
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    
    def to_dict(self):
        """Convert config to dictionary for saving."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)


class TrainingState:
    """Class to manage training state for checkpointing."""
    
    def __init__(self, max_history=1000):
        self.episode = 0
        self.timesteps = 0
        self.best_reward = -float('inf')
        # Is it a problem that they grow forever?/
        self.episode_rewards = []  
        self.episode_lengths = []   
        self.episode_losses = []   
        # self.episode_rewards = deque(maxlen=max_history)
        # self.episode_lengths = deque(maxlen=max_history)
        # self.episode_losses = deque(maxlen=max_history)
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
        'checkpoint_version': '1.0'
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
        'latest_rewards': training_state.episode_rewards[-10:] if training_state.episode_rewards else [],
        'latest_losses': training_state.episode_losses[-10:] if training_state.episode_losses else [],
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
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load configuration
    config = TrainingConfig.from_dict(checkpoint['config'])
    
    # Create or update agent
    if agent is None:
        agent = MotorPPOAgent(
            state_dim=2,
            num_motors=config.num_motors,
            device=device,
            hidden_size=config.hidden_size,
            lr_actor=config.lr_actor,
            lr_critic=config.lr_critic,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_param=config.clip_param,
            entropy_coef=config.entropy_coef,
            batch_size=config.batch_size
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


def train(config: TrainingConfig, resume_from: Optional[str] = None):
    """
    Main training function with checkpoint resume support.
    
    Args:
        config: Training configuration
        resume_from: Path to checkpoint to resume from (optional)
    """
    global env
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create directories
    create_directories(config)
    
    # Initialize training state
    training_state = TrainingState()
    agent = None
    
    # Resume from checkpoint if specified
    if resume_from:
        if resume_from == "latest":
            resume_from = find_latest_checkpoint(config.checkpoint_dir)
            if not resume_from:
                logger.warning("No checkpoint found to resume from. Starting fresh.")
        
        if resume_from and os.path.exists(resume_from):
            try:
                agent, training_state, loaded_config = load_checkpoint(resume_from, device=device)
                
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
    
    # 1. Setup audio system
    logger.info("Setting up audio system...")
    audio = EnhancedAudio(
        sample_rate=config.sample_rate,
        channels=config.channels,
        buffer_size=config.buffer_size,
        input_device=config.input_device,
        enable_spectral_loss=True
    )
    
    # 2. Create loss processor
    logger.info("Creating loss processor...")
    loss_processor = SimpleLossProcessor(
        spectral_loss_calculator=audio.spectral_loss,
        device=device
    )

    osc_handler = OSCHandler()
    setup_signal_handlers(osc_handler)
    
    # Start audio
    logger.info("Starting audio system...")
    audio.start()
    
    # Wait for loss processor to be ready
    logger.info("Waiting for loss processor...")
    start_time = time.time()
    while not loss_processor.is_ready() and time.time() - start_time < 15:
        time.sleep(0.5)
    
    if not loss_processor.is_ready():
        logger.error("Loss processor not ready after timeout")
        return
    
    logger.info("Loss processor ready!")
    
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
    
    # Start OSC server
    logger.info("Starting OSC server")
    osc_handler.start()
    
    # 4. Create environment
    logger.info("Creating motor environment...")
    env = MotorEnvironment(
        loss_processor=loss_processor,
        motor_controller=motor_controller,
        num_motors=config.num_motors,
        step_wait_time=config.step_wait_time,
        reset_wait_time=config.reset_wait_time,
        early_stopping_threshold=config.early_stopping_threshold,
        max_steps_without_improvement=config.max_steps_without_improvement,
        use_motors=config.use_motors,
        motor_speed=config.motor_speed,
        motor_reset_speed=config.motor_reset_speed,
        motor_steps=config.motor_steps,
        reward_scale=config.reward_scale,
        max_ccw_steps=config.max_ccw_steps,
        max_cw_steps=config.max_cw_steps,
        limit_penalty=config.limit_penalty
    )
    
    # 5. Create PPO agent if not loaded from checkpoint
    if agent is None:
        logger.info("Creating new PPO agent...")
        agent = MotorPPOAgent(
            state_dim=2,
            num_motors=config.num_motors,
            device=device,
            hidden_size=config.hidden_size,
            lr_actor=config.lr_actor,
            lr_critic=config.lr_critic,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_param=config.clip_param,
            entropy_coef=config.entropy_coef,
            batch_size=config.batch_size
        )
    
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
            
            # Store transition
            agent.store_transition(obs, actions, log_prob, reward, value, terminated or truncated)
            
            # Update metrics
            training_state.timesteps += 1
            episode_reward += reward
            episode_loss_sum += info['spectral_loss']
            
            # Log step details occasionally
            if step % 20 == 0:
                logger.info(f"Episode {training_state.episode}, Step {step}: "
                           f"Loss={info['spectral_loss']:.4f}, "
                           f"Reward={reward:.2f}, Motors moved: {info['motors_moved']}")
            
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

        if training_state.episode % 10 == 0:
            gc.collect()  # Force garbage collection
            # Also clean up spectral data
            if hasattr(audio.spectral_loss, 'cleanup_old_spectral_data'):
                audio.spectral_loss.cleanup_old_spectral_data()
            logger.info("Garbage Collection")
        
        training_state.episode_rewards.append(episode_reward)
        training_state.episode_lengths.append(ep_length)
        training_state.episode_losses.append(avg_loss)
        
        # Calculate rolling averages
        window = min(10, len(training_state.episode_rewards))
        avg_reward = np.mean(training_state.episode_rewards[-window:])
        avg_ep_loss = np.mean(training_state.episode_losses[-window:])
        
        logger.info(f"Episode {training_state.episode}: Reward={episode_reward:.2f}, "
                   f"Length={ep_length}, Loss={avg_loss:.4f}, "
                   f"Avg Reward={avg_reward:.2f}, Avg Loss={avg_ep_loss:.4f}, "
                   f"Total Timesteps={training_state.timesteps}")
        
        # Adjust learning rate after warm-up
        if training_state.episode > 20:
            agent.actor_optimizer.param_groups[0]['lr'] = 3e-4
            agent.critic_optimizer.param_groups[0]['lr'] = 3e-4
            if training_state.episode == 21:
                logger.info("Reduced learning rates to 3e-4 after episode 20")
        
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
        if training_state.episode % 20 == 0:
            plot_training_progress(
                training_state.episode_rewards,
                training_state.episode_losses,
                training_state.episode_lengths,
                save_path=os.path.join(config.results_dir, 
                                       f"progress_ep{training_state.episode}.png")
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
    env.close()
    audio.stop()


def plot_training_progress(rewards, losses, lengths, save_path):
    """Plot training progress."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    
    # Episode rewards
    ax1.plot(rewards, alpha=0.6)
    if len(rewards) > 10:
        smoothed = np.convolve(rewards, np.ones(10)/10, mode='valid')
        ax1.plot(range(len(smoothed)), smoothed, 'r-', linewidth=2)
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Training Progress')
    ax1.grid(True, alpha=0.3)
    
    # Spectral loss
    ax2.plot(losses, alpha=0.6)
    if len(losses) > 10:
        smoothed = np.convolve(losses, np.ones(10)/10, mode='valid')
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
    plt.savefig(save_path, dpi=150)
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
    
    # Create config
    config = TrainingConfig()
    
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
    
    # Run training with resume support
    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()