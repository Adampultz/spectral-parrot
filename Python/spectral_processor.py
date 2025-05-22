import torch
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)
class SpectralProcessor:
    """
    Processes mel spectrograms from OSC messages for use in reinforcement learning.
    Handles separate target and input data with improved reward calculation.
    """
    def __init__(self, buffer_size=10, num_bands=40, device='cpu'):
        """
        Initialize the spectral processor.
        
        Args:
            buffer_size: Size of the buffer to store recent spectrograms
            num_bands: Number of mel bands in each spectrogram (per target/input)
            device: The torch device to use ('cpu' or 'cuda')
        """
        self.num_bands = num_bands
        self.device = device
        self.buffer_size = buffer_size
        
        # Separate buffers for target and input data
        self.target_buffer = deque(maxlen=buffer_size)
        self.input_buffer = deque(maxlen=buffer_size)
        
        # Initialize buffers with zeros
        for _ in range(buffer_size):
            self.target_buffer.append(torch.zeros(num_bands, device=device))
            self.input_buffer.append(torch.zeros(num_bands, device=device))
        
        # Track whether we've received enough data to begin training
        self.ready = False
        
        # Track historical performance
        self.best_mse = float('inf')
        self.mse_history = deque(maxlen=50)
        
    def receive_spectral_data(self, target_data, input_data):
        """
        Process incoming separate target and input mel spectrogram data from OSC.
        This method is designed to be used as a callback for the OSCHandler.
        
        Args:
            target_data: Target mel spectrogram (numpy array)
            input_data: Input mel spectrogram (numpy array)
        """
        # Ensure the data lengths are what we expect
        if len(target_data) != self.num_bands or len(input_data) != self.num_bands:
            logger.warning(f"Received data lengths don't match expected {self.num_bands}: target={len(target_data)}, input={len(input_data)}")
            return
            
        # Convert numpy arrays to torch tensors
        target_tensor = torch.FloatTensor(target_data).to(self.device)
        input_tensor = torch.FloatTensor(input_data).to(self.device)
        
        # Add to buffers
        self.target_buffer.append(target_tensor)
        self.input_buffer.append(input_tensor)
        
        # Check if we have enough data
        if not self.ready and len(self.target_buffer) == self.buffer_size:
            self.ready = True
            logger.info("Spectral processor ready for training")
            
    def get_observation(self):
        """
        Get the current observation for the agent.
        
        Returns:
            torch.Tensor: The observation tensor (combined target and input)
        """
        if not self.ready:
            logger.warning("Attempting to get observation before spectral processor is ready")
            
        # Get the most recent spectrograms
        target_mel = self.target_buffer[-1]
        input_mel = self.input_buffer[-1]
        
        # Combine them for the observation
        observation = torch.cat([target_mel, input_mel])
        observation = torch.nan_to_num(observation, nan=0.0)
        observation = torch.clamp(observation, 0.0, 1.0)
        
        return observation.unsqueeze(0)  # Add batch dimension [1, 2*num_bands]
    
    def calculate_reward(self, reward_scale=0.1, previous_mse=None):
        """
        Calculate reward based on spectral difference between target and input.
        Includes reward scaling and improvement bonus.
        
        Args:
            reward_scale: Scale factor to apply to the reward
            previous_mse: Previous MSE value, if available, to calculate improvement
        
        Returns:
            float: The scaled reward and current MSE value
        """
        # Get the most recent spectrograms
        target_mel = self.target_buffer[-1]
        input_mel = self.input_buffer[-1]
        
        # Calculate MSE
        mse = torch.nn.functional.mse_loss(target_mel, input_mel)
        mse_value = mse.item()
        
        # Add to history
        self.mse_history.append(mse_value)
        
        # Update best MSE
        if mse_value < self.best_mse:
            self.best_mse = mse_value
        
        # Base reward is negative MSE (higher reward for smaller difference)
        base_reward = -mse_value
        
        # Add improvement bonus if we have previous MSE
        improvement_bonus = 0
        if previous_mse is not None:
            # Calculate how much the MSE improved
            improvement = previous_mse - mse_value
            if improvement > 0:
                # Reward improvement proportionally
                improvement_bonus = improvement * 5.0
        
        # Add bonus for approaching the best performance
        best_approach_bonus = 0
        if self.mse_history:
            # Compare with recent average
            recent_avg_mse = np.mean(list(self.mse_history))
            if mse_value < recent_avg_mse:
                # Reward being better than recent average
                best_approach_bonus = (recent_avg_mse - mse_value) * 2.0
        
        # Calculate final reward
        reward = (base_reward + improvement_bonus + best_approach_bonus) * reward_scale
        
        # Log occasionally
        if np.random.random() < 0.01:  # Log about 1% of the time
            logger.debug(f"Reward components: base={base_reward*reward_scale:.4f}, "
                        f"improvement={improvement_bonus*reward_scale:.4f}, "
                        f"approach={best_approach_bonus*reward_scale:.4f}, "
                        f"total={reward:.4f}")
        
        # Return both the reward and the current MSE for future comparison
        return reward, mse_value
    
    def get_performance_stats(self):
        """
        Get statistics about the spectral matching performance.
        
        Returns:
            dict: Performance statistics
        """
        if not self.mse_history:
            return {
                'current_mse': float('inf'),
                'best_mse': float('inf'),
                'avg_mse': float('inf'),
                'mse_std': 0.0
            }
        
        return {
            'current_mse': self.mse_history[-1],
            'best_mse': self.best_mse,
            'avg_mse': np.mean(list(self.mse_history)),
            'mse_std': np.std(list(self.mse_history))
        }
    
    def reset_stats(self):
        """
        Reset the performance statistics.
        """
        self.best_mse = float('inf')
        self.mse_history.clear()