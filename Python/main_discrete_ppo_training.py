import logging
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import signal

from pythonosc import udp_client
from osc_handler import OSCHandler, setup_signal_handlers
from spectral_processor import SpectralProcessor
from motor_osc_env import DiscreteOSCAndMotorEnvironment  # Import from the new file
from ppo_agent import HybridPPOAgent
from Stepper_Control import DualESP32StepperController

def signal_handler(sig, frame):
    """Handle cleanup on signal (Ctrl+C, system terminate, etc)"""
    logger.info("Received termination signal, performing emergency motor stop...")
    
    if 'env' in globals() and hasattr(env, 'motor_controller') and env.motor_controller:
        try:
            # First try the global stop
            env.motor_controller.stop_motor(0)
            time.sleep(0.1)
            
            # Then stop each motor individually
            for motor_num in range(1, 9):  # Assuming 8 motors
                env.motor_controller.stop_motor(motor_num)
                time.sleep(0.05)
            
            # Wait for stops to take effect
            time.sleep(0.3)
            
            # Disconnect
            env.motor_controller.disconnect()
        except Exception as e:
            logger.error(f"Error during emergency stop: {e}")
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("discrete_ppo_training.log")
    ]
)
logger = logging.getLogger(__name__)

# Ensure output directory exists
os.makedirs("./training_results", exist_ok=True)

# Motor control configuration
USE_MOTORS = True  # Set to False to disable motor control and use only OSC
ESP32_PORT1 = "/dev/cu.usbserial-0001"  # Port for first ESP32 (odd motors) - adjust as needed
ESP32_PORT2 = "/dev/cu.usbserial-2"  # Port for second ESP32 (even motors) - adjust as needed
ESP32_BAUDRATE = 115200

# Training hyperparameters
NUM_OSCILLATORS = 8
NUM_MEL_BANDS = 40
TOTAL_TIMESTEPS = 100000
MAX_EP_LENGTH = 500
UPDATE_INTERVAL = 128  # Reduced from 256 for more frequent updates
BATCH_SIZE = 64
N_EPOCHS = 10  # Slightly reduced to prevent overfitting
GAMMA = 0.995  # Increased from 0.99 for better long-term reward consideration
GAE_LAMBDA = 0.95
CLIP_PARAM = 0.2
LR_ACTOR = 3e-4  # Increased from 1e-4 for faster learning
LR_CRITIC = 3e-4  # Increased from 1e-4 for faster learning
HIDDEN_SIZE = 256
STEP_WAIT_TIME = 3.5
RESET_WAIT_TIME = 0.3
ENTROPY_COEF = 0.005  # Reduced from default 0.01 to reduce random exploration
REWARD_SCALE = 1.0  # Scale factor for rewards to bring them to a better range
EARLY_STOPPING_THRESHOLD = 0.02  # Spectral distance threshold for early stopping

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")



def main():
    try:
        logger.info("Starting Discrete PPO training for oscillator/motor control")
        
        # Create OSC handler
        osc_handler = OSCHandler()
        setup_signal_handlers(osc_handler)
        
        # Create spectral processor
        spectral_processor = SpectralProcessor(buffer_size=10, num_bands=NUM_MEL_BANDS, device=device)
        
        # Register spectral processor with OSC handler
        osc_handler.register_callback(spectral_processor.receive_spectral_data)
        
        # Start OSC server
        logger.info("Starting OSC server")
        osc_handler.start()
        
        # Create motor controller if using motors
        motor_controller = None
        if USE_MOTORS:
            logger.info(f"Initializing motor controller on ports {ESP32_PORT1} and {ESP32_PORT2}")
            motor_controller = DualESP32StepperController(
                port1=ESP32_PORT1,
                port2=ESP32_PORT2,
                baudrate=ESP32_BAUDRATE,
                debug=False
            )
            if not motor_controller.connect():
                logger.warning("Failed to connect to one or both ESP32s. Continuing with limited motor control.")
        
        # Wait for initial data
        logger.info("Waiting for spectral data...")
        start_time = time.time()
        while not spectral_processor.ready and time.time() - start_time < 10:
            time.sleep(0.1)
        
        if not spectral_processor.ready:
            logger.warning("Timeout waiting for spectral data. Check SuperCollider connection.")
        else:
            logger.info("Spectral processor ready")
        
        # Create environment with discrete frequency actions and motor control
        env = DiscreteOSCAndMotorEnvironment(
            spectral_processor=spectral_processor,
            osc_client=osc_handler.client,
            motor_controller=motor_controller,
            port1=ESP32_PORT1,
            port2=ESP32_PORT2,
            baudrate=ESP32_BAUDRATE,
            num_oscillators=NUM_OSCILLATORS,
            amp_range=(0.0, 1.0),
            step_wait_time=STEP_WAIT_TIME,
            reset_wait_time=RESET_WAIT_TIME,
            reward_scale=REWARD_SCALE,
            early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
            use_motors=USE_MOTORS
        )
        
        # Get dimensions
        state_dim = NUM_MEL_BANDS * 2  # Target + Input mel bands
        
        # Create Hybrid PPO agent
        agent = HybridPPOAgent(
            state_dim=state_dim,
            num_oscillators=NUM_OSCILLATORS,
            device=device,
            hidden_size=HIDDEN_SIZE,
            lr_actor=LR_ACTOR,
            lr_critic=LR_CRITIC,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_param=CLIP_PARAM,
            batch_size=BATCH_SIZE,
            entropy_coef=ENTROPY_COEF
        )

        from torch.optim.lr_scheduler import ReduceLROnPlateau

        actor_scheduler = ReduceLROnPlateau(
            agent.actor_optimizer, 
            mode='max', 
            factor=0.5,
            patience=5,
            verbose=True
        )

        critic_scheduler = ReduceLROnPlateau(
            agent.critic_optimizer, 
            mode='max', 
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training loop
        time_step = 0
        episode = 0
        best_reward = float('-inf')
        
        episode_rewards = []
        episode_lengths = []
        spectral_distances = []
        update_times = []
        
        # Track frequency action distribution
        freq_action_counts = np.zeros((NUM_OSCILLATORS, 3))  # [osc, action] where action is 0=dec, 1=same, 2=inc
        
        logger.info("Starting training loop")
        
        while time_step < TOTAL_TIMESTEPS:
            episode += 1
            state = env.reset()

            logger.debug("Initial state shape: %s", state.shape)
            logger.debug("State min/max: %.4f/%.4f", np.min(state), np.max(state))
            logger.debug("Any NaNs in state: %s", np.isnan(state).any())

            ep_reward = 0
            ep_spectral_distance = 0
            
            for step in range(MAX_EP_LENGTH):
                # Convert state to tensor for agent
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
                # Select action
                actions, log_prob, value = agent.select_action(state_tensor)
                
                # Track frequency action distribution
                for i, action in enumerate(actions['freq_actions']):
                    freq_action_counts[i, action] += 1
                
                # Take action in environment
                next_state, reward, done, truncated, info = env.step(actions)
                
                # Store transition
                agent.store_transition(
                    state, actions, log_prob, reward, value, done or truncated
                )
                
                # Update counters and statistics
                time_step += 1
                ep_reward += reward
                ep_spectral_distance += info['spectral_distance']
                
                # Move to next state
                state = next_state
                
                # Check if it's time to update
                if time_step % UPDATE_INTERVAL == 0:
                    update_start = time.time()
                    
                    # Get value of current state for bootstrapping
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    with torch.no_grad():
                        next_value = agent.critic(state_tensor).item()
                    
                    # Update agent
                    actor_loss, critic_loss = agent.update(next_value, n_epochs=N_EPOCHS)
                    
                    update_duration = time.time() - update_start
                    update_times.append(update_duration)
                    
                    logger.info(f"Update at step {time_step}: "
                                f"Actor loss: {actor_loss:.4f}, "
                                f"Critic loss: {critic_loss:.4f}, "
                                f"Duration: {update_duration:.2f}s")
                    
                    # Save model checkpoint
                    agent.save(f"./training_results/discrete_ppo_checkpoint_{time_step}.pt")
                
                # Check if episode is done
                if done or truncated or step == MAX_EP_LENGTH - 1:
                    break
            
            # End of episode logging
            ep_length = step + 1
            avg_spectral_distance = ep_spectral_distance / ep_length
            
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            spectral_distances.append(avg_spectral_distance)
            
            # Calculate rolling statistics
            window_size = min(10, len(episode_rewards))
            avg_reward = np.mean(episode_rewards[-window_size:])
            avg_distance = np.mean(spectral_distances[-window_size:])
            
            logger.info(f"Episode {episode}: "
                       f"Reward: {ep_reward:.4f}, "
                       f"Length: {ep_length}, "
                       f"Avg Spectral Distance: {avg_spectral_distance:.6f}, "
                       f"Rolling Avg Reward: {avg_reward:.4f}")

            if episode % 10 == 0:
                # Update learning rate based on performance
                actor_scheduler.step(avg_reward)
                critic_scheduler.step(avg_reward)
            
            # Plot progress every 10 episodes
            if episode % 10 == 0 or time_step >= TOTAL_TIMESTEPS:
                # Plot rewards
                plt.figure(figsize=(14, 12))
                plt.subplot(3, 2, 1)
                plt.plot(episode_rewards)
                plt.title('Episode Rewards')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                
                # Plot spectral distances
                plt.subplot(3, 2, 2)
                plt.plot(spectral_distances)
                plt.title('Average Spectral Distance')
                plt.xlabel('Episode')
                plt.ylabel('Distance (MSE)')
                
                # Plot episode lengths
                plt.subplot(3, 2, 3)
                plt.plot(episode_lengths)
                plt.title('Episode Lengths')
                plt.xlabel('Episode')
                plt.ylabel('Steps')
                
                # Plot update times (if any)
                if update_times:
                    plt.subplot(3, 2, 4)
                    plt.plot(update_times)
                    plt.title('Policy Update Times')
                    plt.xlabel('Update')
                    plt.ylabel('Time (s)')
                
                # Plot frequency action distribution
                total_actions = np.sum(freq_action_counts, axis=1, keepdims=True)
                if np.all(total_actions > 0):  # Avoid division by zero
                    freq_action_percentages = freq_action_counts / total_actions
                    
                    plt.subplot(3, 2, 5)
                    x = np.arange(NUM_OSCILLATORS)
                    width = 0.25
                    
                    plt.bar(x - width, freq_action_percentages[:, 0], width, label='Decrease (-1)')
                    plt.bar(x, freq_action_percentages[:, 1], width, label='Maintain (0)')
                    plt.bar(x + width, freq_action_percentages[:, 2], width, label='Increase (1)')
                    
                    plt.xlabel('Oscillator')
                    plt.ylabel('Action Percentage')
                    plt.title('Frequency Action Distribution')
                    plt.xticks(x)
                    plt.legend()
                
                plt.tight_layout()
                plt.savefig(f"./training_results/discrete_progress_episode_{episode}.png")
                plt.close()
                
                # Save trained model periodically
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    agent.save("./training_results/discrete_ppo_best_model.pt")
                    logger.info(f"Saved new best model with avg reward: {best_reward:.4f}")
            
        # Save final trained model
        agent.save("./training_results/discrete_ppo_final_model.pt")
        logger.info("Training complete. Final model saved.")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if 'agent' in locals():
            agent.save("./training_results/discrete_ppo_interrupted_model.pt")
            logger.info("Saved model at interruption point")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        # Clean up resources
        if 'env' in locals():
            env.close()
        if 'osc_handler' in locals():
            osc_handler.cleanup()
        logger.info("Application stopped")

if __name__ == "__main__":
    main()