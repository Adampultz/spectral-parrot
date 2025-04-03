import numpy as np
import torch
import logging
from pythonosc import udp_client
import time
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)

class DiscreteOSCEnvironment(gym.Env):
    """
    A custom Gym environment with discrete actions for oscillator frequencies
    and continuous actions for oscillator amplitudes.
    """
    
    def __init__(self, spectral_processor, osc_client, num_oscillators=8, 
             amp_range=(0.0, 1.0), step_wait_time=1.0, reset_wait_time=0.3,
             reward_scale=0.1, early_stopping_threshold=0.02):
        """
        Initialize the environment.
        
        Args:
            spectral_processor: SpectralProcessor instance to handle mel spectrogram data
            osc_client: OSC client to send messages to SuperCollider
            num_oscillators: Number of oscillators to control
            amp_range: Range of amplitudes (min, max) for the oscillators
            step_wait_time: Time to wait after each step (in seconds)
            reset_wait_time: Time to wait after reset (in seconds)
        """
        super(DiscreteOSCEnvironment, self).__init__()
        
        self.spectral_processor = spectral_processor
        self.osc_client = osc_client
        self.num_oscillators = num_oscillators
        self.amp_range = amp_range
        self.amp_range_db = (-70, 0)
        self.step_wait_time = step_wait_time
        self.reset_wait_time = reset_wait_time
        self.reward_scale = reward_scale
        self.early_stopping_threshold = early_stopping_threshold
        
        # Define action and observation spaces
        num_mel_bands = spectral_processor.num_bands

        self.previous_mse = None
    
        # Track best performance for early stopping
        self.best_spectral_distance = float('inf')
        self.steps_without_improvement = 0
        self.max_steps_without_improvement = 100 
        
        # Discrete action space for frequencies (3 actions per oscillator: decrease, maintain, increase)
        # We'll use an internal representation of 0, 1, 2 but convert to -1, 0, 1 when sending to SC
        self.freq_action_space = spaces.MultiDiscrete([3] * num_oscillators)
        
        # Continuous action space for amplitudes
        self.amp_action_space = spaces.Box(
            low=-3.0, high=3.0,  # Using unbounded values (will be passed through sigmoid)
            shape=(num_oscillators,), 
            dtype=np.float32
        )
        
        # Combined action space (not actually used by gym but useful for documentation)
        self.action_space = spaces.Dict({
            'freq_actions': self.freq_action_space,
            'amp_actions': self.amp_action_space
        })
        
        # Observation space: target and input mel spectrograms concatenated
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(num_mel_bands * 2,), 
            dtype=np.float32
        )
        
        # Current oscillator parameters (used for tracking only, not for calculating new values)
        self.current_amps = np.zeros(num_oscillators)
    
    def amp_to_db(self, amplitude):
        """Convert amplitude to decibels"""
        # Add a small epsilon to avoid log(0)
        return 20 * np.log10(np.maximum(amplitude, 1e-10))
    
    def linlin(self, value, in_min, in_max, out_min, out_max, clip=False):
        """
        Convert a value from input range to output range.
        Similar to SuperCollider's .linlin() function.
        """
        if clip:
            # Clip input to the specified range
            value = np.clip(value, in_min, in_max)
        
        # Linear mapping from input range to output range
        return out_min + (value - in_min) / (in_max - in_min) * (out_max - out_min)
    
    def _process_actions(self, actions):
        """
        Process the discrete frequency actions and continuous amplitude actions.
        
        Args:
            actions: Dictionary with 'freq_actions' and 'amp_actions'
            
        Returns:
            freq_commands: Discrete frequency commands (0=decrease, 1=maintain, 2=increase)
            amp_values: Amplitude values (0-1)
        """
        # Extract actions
        freq_actions = actions['freq_actions']
        amp_actions = actions['amp_actions']
        
        # Process amplitude actions (continuous values through sigmoid)
        amp_actions_norm = 1.0 / (1.0 + np.exp(-amp_actions))
        amp_actions_db = self.amp_to_db(amp_actions_norm)
        amp_values = self.linlin(amp_actions_db, self.amp_range_db[0], self.amp_range_db[1], 0.0, 1.0, True)
        
        # Store current amplitudes for tracking
        self.current_amps = amp_values
        
        return freq_actions, amp_values

    def reset(self):
        """
        Reset the environment.
        
        Returns:
            np.array: Initial observation
        """

        # Reset tracking variables
        self.previous_mse = None
        self.best_spectral_distance = float('inf')
        self.steps_without_improvement = 0
        # Send reset message to SuperCollider
        # This should communicate that we're starting a new episode, SuperCollider can handle initialization
        logger.info("Sending reset message to SuperCollider")
        self.osc_client.send_message("/reset_oscillators", True)
        
        # Allow more time for SuperCollider to process the reset and generate new random frequencies
        # Using a longer wait time to ensure everything is properly initialized
        time.sleep(self.reset_wait_time)
        
        # Wait for the spectral processor to receive updated data after reset
        start_time = time.time()
        timeout = 2.0  # seconds
        
        while time.time() - start_time < timeout:
            # Check if we have fresh spectral data
            observation = self.spectral_processor.get_observation().cpu().numpy().flatten()
            if np.any(observation != 0):  # If data is non-zero, we have some signal
                break
            time.sleep(0.1)
            
        if time.time() - start_time >= timeout:
            logger.warning("Timeout waiting for spectral data after reset")
        
        # Get the initial observation after reset
        observation = self.spectral_processor.get_observation().cpu().numpy().flatten()
        logger.debug("Reset complete, initial observation shape: %s", observation.shape)
        
        # Return initial observation
        return observation
    
    def step(self, actions):
        """
        Take actions in the environment.
        
        Args:
            actions: Dictionary with 'freq_actions' and 'amp_actions'
            
        Returns:
            observation: Next state observation
            reward: Reward for the action
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Process actions to get the commands and amplitude values
        freq_commands, amp_values = self._process_actions(actions)
        
        # Send commands to SuperCollider
        # The first part of the message is the discrete frequency commands
        # The second part is the continuous amplitude values
        all_params = np.concatenate([freq_commands, amp_values])
        self.send_parameters_to_sc(all_params)
        
        # Allow fixed time for oscillators to stabilize based on the hyperparameter
        logger.debug(f"Waiting {self.step_wait_time}s after step")
        time.sleep(self.step_wait_time)
        
        # Get new observation
        observation = self.spectral_processor.get_observation().cpu().numpy().flatten()
        
        # Calculate reward based on spectral difference
        reward, current_mse = self.spectral_processor.calculate_reward(
            reward_scale=self.reward_scale, 
            previous_mse=self.previous_mse
        )
        
         # Update previous MSE for next step
        self.previous_mse = current_mse
        
        # Calculate spectral distance (positive value)
        spectral_distance = current_mse
        
        # Check for early stopping (very good match)
        terminated = False
        if spectral_distance < self.early_stopping_threshold:
            logger.info(f"Early stopping: Reached excellent spectral match ({spectral_distance:.6f})")
            terminated = True
            # Add a bonus reward for finding an excellent match
            reward += 2.0 * self.reward_scale
        
        # Check for lack of improvement
        if spectral_distance < self.best_spectral_distance:
            self.best_spectral_distance = spectral_distance
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

        truncated = False
        if self.steps_without_improvement >= self.max_steps_without_improvement:
            logger.info(f"Truncating episode: No improvement for {self.steps_without_improvement} steps")
            truncated = True
        
        # Additional info
        info = {
            'frequency_commands': freq_commands.copy(),
            'amplitudes': amp_values.copy(),
            'spectral_distance': spectral_distance,
            'steps_without_improvement': self.steps_without_improvement,
            'best_spectral_distance': self.best_spectral_distance
        }
        
        return observation, reward, terminated, truncated, info
    
    def send_parameters_to_sc(self, params):
        """
        Send current oscillator parameters to SuperCollider.
        """
        try:
            # Extract frequency commands and amplitudes
            freq_commands = params[:self.num_oscillators]
            amp_values = params[self.num_oscillators:]
            
            # Convert frequency commands from [0,1,2] to [-1,0,1]
            # 0 -> -1, 1 -> 0, 2 -> 1
            freq_commands_converted = freq_commands - 1
            
            # Recombine
            converted_params = np.concatenate([freq_commands_converted, amp_values])
            
            # Send to SuperCollider
            # The structure should be [freq_cmd1, freq_cmd2, ..., freq_cmdN, amp1, amp2, ..., ampN]
            # where freq_cmd is -1, 0, or 1 (decrease, maintain, or increase)
            self.osc_client.send_message("/set_osc_params", converted_params.tolist())
            
            # Log occasionally to avoid flooding
            if np.random.random() < 0.05:  # 5% chance to log
                logger.debug(f"Sent parameters to SC: {converted_params}")
        except Exception as e:
            logger.error(f"Error sending parameters to SuperCollider: {e}")

    def close(self):
        """
        Clean up resources.
        """
        # Any cleanup needed when environment is closed
        pass