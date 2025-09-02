# main_motor_training.py - Simplified training for motor control

"""
For LLMs: This is a Python project for deep reinforcement learning on a motorised string instrument.
The instrument is a wooden board with 8 strings. The strings feed back due to each string having an
electromagnetic pickup routed through SuperCollider through an RME Fireface UFX III interface, and 
finally to contact exciters placed on the instrument body. The deep reinforcement learning network 
takes a multi-scale spectral loss, calculated between a target sound and the summed pickup signals, as its input. 
The output are 8 values sent to motors tuning the instrument's strings (each string is tuned by one motor). 
Each of the 8 values can either be: "turn clockwise", "hold", or "turn counterclockwise"
"""

import logging
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import signal
from dataclasses import dataclass
from typing import Optional
from pythonosc import udp_client

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

# Import reward configuration if you created it
# from config import RewardConfig

# Global for signal handling
env = None

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    logger.info("Received shutdown signal, cleaning up...")
    if env is not None:
        env.close()
    sys.exit(0)

    if 'osc_handler' in globals():
        try:
            logger.info("Cleaning up OSC handler...")
            osc_handler.cleanup()
        except Exception as e:
            logger.error(f"Error during OSC cleanup: {e}")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Training configuration
@dataclass
class TrainingConfig:
    # Environment
    num_motors: int = 8
    early_stopping_threshold: float = 20
    max_steps_without_improvement: int = 100
    
    # Motor control
    use_motors: bool = True
    motor_speed: int = 200
    motor_reset_speed: int = 200
    motor_steps: int = 1000
    step_wait_time: float = 2.0
    reset_wait_time: float = 0.3
    
    # Serial ports
    port1: str = "/dev/cu.usbserial-0001"
    port2: str = "/dev/cu.usbserial-1"
    baudrate: int = 115200
    
    # Audioç
    input_device: Optional[int] = None
    sample_rate: int = 44100
    channels: int = 2
    buffer_size: int = 1024
    
    # PPO hyperparameters
    total_timesteps: int = 100000
    max_ep_length: int = 512
    update_interval: int = 32 # Down from 128
    batch_size: int = 16  # Down from 64
    n_epochs: int = 4 # Down from 10
    
    # Learning rates
    lr_actor: float = 1e-3    # From 3e-4
    lr_critic: float = 1e-3   # From 3e-4
    
    # PPO specific
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    
    # Network
    hidden_size: int = 64  # Smaller for simpler task
    
    # Reward (simple version)
    reward_scale: float = 1.0
    
    # Saving
    save_interval: int = 1
    plot_frequency: int = 1
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"

def create_directories(config: TrainingConfig):
    """Create necessary directories."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)

def train(config: TrainingConfig):
    """Main training function."""
    global env
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create directories
    create_directories(config)
    
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
        reward_scale=config.reward_scale
    )
    
    # 5. Create PPO agent
    logger.info("Creating PPO agent...")
    agent = MotorPPOAgent(
        state_dim=1,  # Single loss value
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
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    best_reward = -float('inf')
    
    # 6. Training loop
    logger.info("Starting training...")
    time_step = 0
    episode = 0
    
    while time_step < config.total_timesteps:
        episode += 1
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
            time_step += 1
            episode_reward += reward
            episode_loss_sum += info['spectral_loss']
            
            # Log step details occasionally
            if step % 20 == 0:
                logger.info(f"Step {step}: Loss={info['spectral_loss']:.4f}, "
                           f"Reward={reward:.2f}, Motors moved: {info['motors_moved']}")
            
            # Move to next state
            obs = next_obs
            
            # Update policy
            if time_step % config.update_interval == 0:
                # Get value of final state
                with torch.no_grad():
                    next_value = agent.critic(torch.FloatTensor(obs).to(device)).item()
                
                # Update
                actor_loss, critic_loss = agent.update(next_value, config.n_epochs)
                
                logger.info(f"Update at step {time_step}: "
                          f"Actor loss={actor_loss:.4f}, Critic loss={critic_loss:.4f}, "
                          f"Temperature={agent.current_temperature:.3f}")
            
            # Check if done
            if terminated or truncated:
                break
        
        # Episode complete
        ep_length = step + 1
        avg_loss = episode_loss_sum / ep_length
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(ep_length)
        episode_losses.append(avg_loss)
        
        # Calculate rolling averages
        window = min(10, len(episode_rewards))
        avg_reward = np.mean(episode_rewards[-window:])
        avg_ep_loss = np.mean(episode_losses[-window:])
        
        logger.info(f"Episode {episode}: Reward={episode_reward:.2f}, "
                   f"Length={ep_length}, Loss={avg_loss:.4f}, "
                   f"Avg Reward={avg_reward:.2f}, Avg Loss={avg_ep_loss:.4f}")
        
        if episode > 20:
            agent.actor_optimizer.param_groups[0]['lr'] = 3e-4
            agent.critic_optimizer.param_groups[0]['lr'] = 3e-4
            if episode == 21:  # Log once when it changes
                logger.info("Reduced learning rates to 3e-4 after episode 20")
        
        # Render environment occasionally
        if episode % config.plot_frequency == 0:
            env.render()
        
        # Save checkpoint
        if episode % config.save_interval == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir, 
                f"motor_ppo_ep{episode}_r{avg_reward:.2f}.pt"
            )
            agent.save(checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Update best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_path = os.path.join(config.checkpoint_dir, "best_model.pt")
                agent.save(best_path)
                logger.info(f"New best model! Avg reward: {best_reward:.2f}")
        
        # Plot progress
        if episode % 20 == 0:
            plot_training_progress(
                episode_rewards, episode_losses, episode_lengths,
                save_path=os.path.join(config.results_dir, f"progress_ep{episode}.png")
            )
    
    # Training complete
    logger.info("Training complete!")
    final_path = os.path.join(config.checkpoint_dir, "final_model.pt")
    agent.save(final_path)
    
    # Final statistics
    logger.info(f"Final statistics:")
    logger.info(f"  Total episodes: {episode}")
    logger.info(f"  Best average reward: {best_reward:.2f}")
    logger.info(f"  Final average loss: {np.mean(episode_losses[-10:]):.4f}")
    
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

def main():
    """Main entry point."""
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
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--hidden-size', type=int, help='Hidden layer size')
    
    # Other arguments
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # List devices if requested
    if args.list_devices:
        from stft_audio import SimpleAudio
        SimpleAudio.list_available_devices()
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
    if args.lr:
        config.lr_actor = args.lr
        config.lr_critic = args.lr
    if args.hidden_size:
        config.hidden_size = args.hidden_size
    
    # TODO: Handle resume functionality
    if args.resume:
        logger.info(f"Resume functionality not yet implemented")
    
    # Run training
    train(config)

if __name__ == "__main__":
    main()