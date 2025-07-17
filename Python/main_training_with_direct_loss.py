# main_training_with_direct_loss.py
import logging
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import atexit
import signal

from pythonosc import udp_client
from osc_handler import OSCHandler, setup_signal_handlers

# Import your existing stft_audio functionality
from stft_audio import EnhancedAudio

# Import the simple processors and environments
from simple_spectral_loss_processor import SimpleLossProcessor
from simple_loss_environment import SimpleLossMotorEnvironment

from ppo_agent import HybridPPOAgent
from Stepper_Control import DualESP32StepperController

env = None  # Global variable for emergency shutdown

def emergency_shutdown():
    """Handle cleanup on program exit"""
    global env
    
    if env is not None and hasattr(env, 'motor_controller') and env.motor_controller:
        logger.info("Emergency shutdown: stopping and disabling all motors")
        try:
            env.motor_controller.stop_motor(0)
            time.sleep(0.5)
            # Then disable all drivers to eliminate holding current and sound
            env.motor_controller.send_command(0, "DISABLE", 1)
            time.sleep(0.5)
            env.motor_controller.disconnect()
        except Exception as e:
            logger.error(f"Error during emergency shutdown: {e}")

def signal_handler(sig, frame):
    """Handle cleanup on signal (Ctrl+C, system terminate, etc)"""
    logger.info("Received termination signal, performing emergency motor stop...")
    
    if 'env' in globals() and hasattr(env, 'motor_controller') and env.motor_controller:
        try:
            env.motor_controller.stop_motor(0)
            time.sleep(0.1)
            
            for motor_num in range(1, 9):
                env.motor_controller.stop_motor(motor_num)
                time.sleep(0.05)
            
            time.sleep(0.3)
            env.motor_controller.disconnect()
        except Exception as e:
            logger.error(f"Error during emergency stop: {e}")

    if 'osc_handler' in globals():
        try:
            logger.info("Cleaning up OSC handler...")
            osc_handler.cleanup()
        except Exception as e:
            logger.error(f"Error during OSC cleanup: {e}")
    
    logger.info("Emergency cleanup complete")
    
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
        logging.FileHandler("ppo_training_direct_loss.log")
    ]
)
logger = logging.getLogger(__name__)

# Ensure output directory exists
os.makedirs("./training_results", exist_ok=True)

# Motor control configuration
USE_MOTORS = True
MOTOR_SPEED = 200
MOTOR_RESET_SPEED = 200
MOTOR_MOVE_STEPS = 1000

# Audio configuration for stft_audio.py
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 2  # Required for spectral loss
AUDIO_BUFFER_SIZE = 1024

# AUDIO DEVICE CONFIGURATION
# Set to specific device indices or None for default
# Use --list-devices to see available devices
INPUT_DEVICE = None   # Set to device index (e.g., 2) or None for default
OUTPUT_DEVICE = None  # Set to device index (e.g., 3) or None for default

# Set to True to use interactive device selection on startup
INTERACTIVE_DEVICE_SELECTION = False

# Training hyperparameters
NUM_OSCILLATORS = 8
TOTAL_TIMESTEPS = 100000
MAX_EP_LENGTH = 512
UPDATE_INTERVAL = 128
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.995
GAE_LAMBDA = 0.95
CLIP_PARAM = 0.2
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
HIDDEN_SIZE = 128      # Small since we only have 1 input
STEP_WAIT_TIME = 3.5
RESET_WAIT_TIME = 0.3
ENTROPY_COEF = 0.005
REWARD_SCALE = 10      # Increased for loss scale
EARLY_STOPPING_THRESHOLD = 0.01

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def main():
    global ESP32_PORT1, ESP32_PORT2, ESP32_BAUDRATE, INPUT_DEVICE, OUTPUT_DEVICE
    global INTERACTIVE_DEVICE_SELECTION, USE_MOTORS, AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, REWARD_SCALE
    try:
        logger.info("Starting PPO training with direct multi-scale spectral loss from stft_audio.py")
        
        # Handle device selection using existing SimpleAudio functionality
        input_device = INPUT_DEVICE
        output_device = OUTPUT_DEVICE
        
        if INTERACTIVE_DEVICE_SELECTION:
            logger.info("Starting interactive device selection...")
            # Create temporary audio instance for device selection
            temp_audio = EnhancedAudio(
                sample_rate=AUDIO_SAMPLE_RATE,
                channels=AUDIO_CHANNELS,
                buffer_size=AUDIO_BUFFER_SIZE,
                enable_spectral_loss=False  # Don't start loss calculation yet
            )
            # Use the existing select_devices method
            temp_audio.select_devices()
            input_device = temp_audio.input_device
            output_device = temp_audio.output_device
            del temp_audio  # Clean up
        
        # Create OSC handler
        osc_handler = OSCHandler()
        setup_signal_handlers(osc_handler)
        
        # Start OSC server
        logger.info("Starting OSC server")
        osc_handler.start()
        
        # Create enhanced audio system with channel constraint applied during construction
        logger.info("Creating Enhanced Audio System with Multi-Scale Spectral Loss")
        
        # Validate the input device for spectral loss requirements
        if input_device is not None:
            try:
                import pyaudio
                p = pyaudio.PyAudio()
                device_info = p.get_device_info_by_index(input_device)
                device_channels = device_info['maxInputChannels']
                p.terminate()
                
                logger.info(f"Selected device has {device_channels} input channels")
                
                if device_channels < 2:
                    logger.error(f"Device {input_device} only has {device_channels} input channels")
                    logger.error("Spectral loss requires at least 2 input channels")
                    logger.error("Please select a device with stereo input or use --list-devices to see options")
                    return
                elif device_channels >= 2:
                    logger.info(f"Device {input_device} has {device_channels} input channels")
                    logger.info("The system will use only the first 2 channels for spectral loss")
                    logger.info("Make sure your target audio is on channel 1 and instrument feedback is on channel 2")
                
            except Exception as e:
                logger.warning(f"Could not validate device {input_device}: {e}")

        # Create a custom audio class that fixes channels during construction, before audio starts
        class ChannelFixedEnhancedAudio(EnhancedAudio):
            def __init__(self, *args, **kwargs):
                # Extract parameters before calling parent
                desired_channels = kwargs.get('channels', 2)
                
                # Call parent constructor 
                super().__init__(*args, **kwargs)
                
                # Immediately override input_channels if device detection set it wrong
                if hasattr(self, 'input_channels') and self.input_channels != desired_channels:
                    logger.info(f"Fixing channel count: device reported {self.input_channels}, using {desired_channels}")
                    self.input_channels = desired_channels
                
            def start(self):
                # Override start to ensure channels are correct before starting
                if hasattr(self, 'input_channels') and self.input_channels != self.channels:
                    logger.info(f"Final channel fix before start: using {self.channels} channels")
                    self.input_channels = self.channels
                super().start()

        # Create enhanced audio system with the channel-fixed class
        enhanced_audio = ChannelFixedEnhancedAudio(
            sample_rate=AUDIO_SAMPLE_RATE,
            channels=AUDIO_CHANNELS,  # This should always be 2 for spectral loss
            buffer_size=AUDIO_BUFFER_SIZE,
            input_device=input_device,
            output_device=output_device,
            enable_spectral_loss=True  # This enables your MultiScaleSpectralLoss
        )
        
        # Create simple loss processor that gets loss from enhanced_audio
        logger.info("Creating SimpleLossProcessor to interface with stft_audio.py")
        loss_processor = SimpleLossProcessor(
            spectral_loss_calculator=enhanced_audio.spectral_loss,
            device=device
        )
        
        # Start the enhanced audio system
        logger.info("Starting enhanced audio system")
        enhanced_audio.start()
        
        # Wait for audio system to initialize
        time.sleep(2.0)
        
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
        
        # Wait for initial loss data
        logger.info("Waiting for spectral loss data from stft_audio.py...")
        start_time = time.time()
        while not loss_processor.is_ready() and time.time() - start_time < 15:
            time.sleep(0.5)
            logger.info(f"Waiting for loss data... ({time.time() - start_time:.1f}s)")
        
        if not loss_processor.is_ready():
            logger.warning("Timeout waiting for spectral loss data. Check audio input and ensure stereo audio is available.")
        else:
            logger.info("SimpleLossProcessor ready - receiving loss values from stft_audio.py")
        
        # Create environment that uses the loss processor
        global env
        env = SimpleLossMotorEnvironment(
            loss_processor=loss_processor,
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
            use_motors=USE_MOTORS,
            motor_speed=MOTOR_SPEED,
            motor_reset_speed=MOTOR_RESET_SPEED,
            motor_steps=MOTOR_MOVE_STEPS
        )
        
        # State dimension is now just 1 (single loss value)
        state_dim = 1
        logger.info(f"State dimension: {state_dim} (single spectral loss value from stft_audio.py)")
        
        # Create PPO agent
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

        # Learning rate schedulers
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
        spectral_losses = []
        update_times = []
        
        # Track frequency action distribution
        freq_action_counts = np.zeros((NUM_OSCILLATORS, 3))
        
        logger.info("Starting training loop with direct spectral loss input")
        
        while time_step < TOTAL_TIMESTEPS:
            episode += 1
            state, _ = env.reset()

            logger.debug("Initial state (loss value): %.6f", state[0])

            ep_reward = 0
            ep_loss_sum = 0
            
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
                ep_loss_sum += info['spectral_loss']
                
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
                    agent.save(f"./training_results/ppo_direct_loss_checkpoint_{time_step}.pt")
                
                # Check if episode is done
                if done or truncated or step == MAX_EP_LENGTH - 1:
                    break
            
            # End of episode logging
            ep_length = step + 1
            avg_spectral_loss = ep_loss_sum / ep_length
            
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            spectral_losses.append(avg_spectral_loss)
            
            # Calculate rolling statistics
            window_size = min(10, len(episode_rewards))
            avg_reward = np.mean(episode_rewards[-window_size:])
            avg_loss = np.mean(spectral_losses[-window_size:])
            
            # Get loss processor stats
            loss_stats = loss_processor.get_performance_stats()
            
            logger.info(f"Episode {episode}: "
                       f"Reward: {ep_reward:.4f}, "
                       f"Length: {ep_length}, "
                       f"Avg Loss: {avg_spectral_loss:.6f}, "
                       f"Best Loss: {loss_stats['best_loss']:.6f}, "
                       f"Rolling Avg Reward: {avg_reward:.4f}")

            if episode % 10 == 0:
                # Update learning rate based on performance
                actor_scheduler.step(avg_reward)
                critic_scheduler.step(avg_reward)
            
            # Plot progress every 10 episodes
            if episode % 10 == 0 or time_step >= TOTAL_TIMESTEPS:
                # Create comprehensive training plots
                plt.figure(figsize=(16, 12))
                
                # Plot episode rewards
                plt.subplot(3, 2, 1)
                plt.plot(episode_rewards)
                plt.title('Episode Rewards')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                
                # Plot spectral losses
                plt.subplot(3, 2, 2)
                plt.plot(spectral_losses)
                plt.title('Spectral Loss per Episode')
                plt.xlabel('Episode')
                plt.ylabel('Loss Value')
                
                # Plot episode lengths
                plt.subplot(3, 2, 3)
                plt.plot(episode_lengths)
                plt.title('Episode Lengths')
                plt.xlabel('Episode')
                plt.ylabel('Steps')
                
                # Plot update times
                if update_times:
                    plt.subplot(3, 2, 4)
                    plt.plot(update_times)
                    plt.title('Policy Update Times')
                    plt.xlabel('Update')
                    plt.ylabel('Time (s)')
                
                # Plot frequency action distribution
                total_actions = np.sum(freq_action_counts, axis=1, keepdims=True)
                if np.all(total_actions > 0):
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
                
                # Plot reward vs loss correlation
                if len(episode_rewards) > 1 and len(spectral_losses) > 1:
                    plt.subplot(3, 2, 6)
                    plt.scatter(spectral_losses, episode_rewards, alpha=0.6)
                    plt.xlabel('Average Spectral Loss')
                    plt.ylabel('Episode Reward')
                    plt.title('Reward vs Spectral Loss')
                    # Add trend line
                    if len(spectral_losses) > 2:
                        z = np.polyfit(spectral_losses, episode_rewards, 1)
                        p = np.poly1d(z)
                        plt.plot(spectral_losses, p(spectral_losses), "r--", alpha=0.8)
                
                plt.tight_layout()
                plt.savefig(f"./training_results/direct_loss_progress_episode_{episode}.png")
                plt.close()
                
                # Save best model
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    agent.save("./training_results/ppo_direct_loss_best_model.pt")
                    logger.info(f"Saved new best model with avg reward: {best_reward:.4f} (avg loss: {avg_loss:.6f})")
            
        # Save final model
        agent.save("./training_results/ppo_direct_loss_final_model.pt")
        logger.info("Training complete. Final model saved.")
        
        # Print final statistics
        if spectral_losses:
            final_stats = loss_processor.get_performance_stats()
            logger.info(f"Final training statistics:")
            logger.info(f"  Best reward: {best_reward:.4f}")
            logger.info(f"  Best loss achieved: {final_stats['best_loss']:.6f}")
            logger.info(f"  Data points processed: {final_stats['data_points']}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if 'agent' in locals():
            agent.save("./training_results/ppo_direct_loss_interrupted_model.pt")
            logger.info("Saved model at interruption point")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        # Clean up resources
        if 'enhanced_audio' in locals():
            enhanced_audio.stop()
        if 'env' in locals():
            env.close()
        if 'osc_handler' in locals():
            osc_handler.cleanup()
        logger.info("Application stopped")

if __name__ == "__main__":
    import argparse
    
    # Add command line argument support - MOVE THIS TO THE TOP
    parser = argparse.ArgumentParser(description='PPO Training with Direct Spectral Loss')
    parser.add_argument('--input-device', type=int, help='Input audio device index')
    parser.add_argument('--output-device', type=int, help='Output audio device index')
    parser.add_argument('--list-devices', action='store_true', help='List audio devices and exit')
    parser.add_argument('--interactive-devices', action='store_true', help='Use interactive device selection')
    parser.add_argument('--no-motors', action='store_true', help='Disable motor control')
    
    # Serial port arguments
    parser.add_argument('--port1', type=str, default="/dev/cu.usbserial-0001", help='Serial port for ESP32 #1 (odd motors)')
    parser.add_argument('--port2', type=str, default="/dev/cu.usbserial-2", help='Serial port for ESP32 #2 (even motors)')
    parser.add_argument('--baudrate', type=int, default=115200, help='Serial communication baud rate')
    
    parser.add_argument('--sample-rate', type=int, default=44100, help='Audio sample rate')
    parser.add_argument('--channels', type=int, default=2, help='Audio channels (must be 2 for spectral loss)')
    parser.add_argument('--reward-scale', type=float, default=10, help='Reward scale factor')
    
    args = parser.parse_args()
    
    # List devices and exit if requested (use existing SimpleAudio method)
    if args.list_devices:
        try:
            from stft_audio import SimpleAudio
            print("\n" + "="*80)
            print("AVAILABLE AUDIO DEVICES FOR TRAINING")
            print("="*80)
            SimpleAudio.list_available_devices()
            print("="*80)
            print("\nTo use a device, note its index and run:")
            print("python main_training_with_direct_loss.py --input-device INDEX")
            print("\nFor interactive selection, run:")
            print("python main_training_with_direct_loss.py --interactive-devices")
        except ImportError:
            print("Error: Cannot import audio modules")
        sys.exit(0)
    
    # Override configuration with command line arguments
    if args.input_device is not None:
        INPUT_DEVICE = args.input_device
        logger.info(f"Using input device from command line: {INPUT_DEVICE}")
    
    if args.output_device is not None:
        OUTPUT_DEVICE = args.output_device
        logger.info(f"Using output device from command line: {OUTPUT_DEVICE}")
    
    if args.interactive_devices:
        INTERACTIVE_DEVICE_SELECTION = True
        logger.info("Interactive device selection enabled")
    
    if args.no_motors:
        USE_MOTORS = False
        logger.info("Motor control disabled via command line")
    
    # Serial port configuration
    ESP32_PORT1 = args.port1
    ESP32_PORT2 = args.port2  
    ESP32_BAUDRATE = args.baudrate
    
    # print(f"Using serial ports: {ESP32_PORT1}, {ESP32_PORT2} at {ESP32_BAUDRATE} baud")

    logger.info(f"Using serial ports: {ESP32_PORT1}, {ESP32_PORT2} at {ESP32_BAUDRATE} baud")
    
    # Apply other argument overrides
    if args.sample_rate != 44100:
        AUDIO_SAMPLE_RATE = args.sample_rate
        logger.info(f"Using sample rate from command line: {AUDIO_SAMPLE_RATE}")
    
    if args.channels != 2:
        if args.channels < 2:
            logger.warning("Spectral loss requires 2 channels. Continuing with 2 channels.")
        else:
            AUDIO_CHANNELS = args.channels
            logger.info(f"Using {AUDIO_CHANNELS} audio channels")
    
    if args.reward_scale != 10:
        REWARD_SCALE = args.reward_scale
        logger.info(f"Using reward scale from command line: {REWARD_SCALE}")
    
    # Now call main() - all configuration is set
    main()