# simple_spectral_loss_processor.py
import torch
import numpy as np
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)

class SimpleLossProcessor:
    """
    Simple processor that gets the multi-scale spectral loss value 
    from the existing stft_audio.py implementation and provides it 
    as a single observation value for the RL agent.
    """
    
    def __init__(self, spectral_loss_calculator, device='cpu'):
        """
        Initialize the loss processor.
        
        Args:
            spectral_loss_calculator: Instance of MultiScaleSpectralLoss from stft_audio.py
            device: The torch device to use ('cpu' or 'cuda')
        """
        self.spectral_loss_calculator = spectral_loss_calculator
        self.device = device
        
        # Track current loss and history
        self.current_loss = 0.0
        self.loss_history = deque(maxlen=100)
        self.previous_loss = None

        self.current_direction = 0
        
        # Track best performance for reward calculation
        self.best_loss = float('inf')
        
        # Track whether we've received data
        self.ready = False
        self.data_received_count = 0
        
        # Add callback to the spectral loss calculator to get loss values
        self.spectral_loss_calculator.add_loss_callback(self._receive_loss_data)
        
        logger.info("SimpleLossProcessor initialized - will receive loss from stft_audio.py")
        
    def _receive_loss_data(self, loss_data):
        """
        Callback to receive loss data from the MultiScaleSpectralLoss calculator.
        
        Args:
            loss_data: Dictionary with 'total_loss' and other loss information
        """
        if loss_data and 'total_loss' in loss_data:
            self.current_loss = float(loss_data['total_loss'])
            self.loss_history.append(self.current_loss)
            
            # Update best loss
            if self.current_loss < self.best_loss:
                self.best_loss = self.current_loss

            if 'direction' in loss_data:
                self.current_direction = float(loss_data['direction'])
            
            # Track data reception
            self.data_received_count += 1
            if not self.ready and self.data_received_count >= 5:  # Wait for a few samples
                self.ready = True
                logger.info("SimpleLossProcessor ready - receiving loss values")
                
            # Log occasionally
            if self.data_received_count % 50 == 0:
                logger.debug(f"Received loss update #{self.data_received_count}: {self.current_loss:.6f}")
        
    def get_observation(self):
        """
        Get the current observation for the agent (single loss value).
        
        Returns:
            torch.Tensor: The observation tensor containing the current loss value
        """
        if not self.ready:
            logger.warning("Attempting to get observation before loss processor is ready")
            # Return a reasonable default value
            loss_value = 10.0  # High loss indicates poor match
            direction_value = 0.0  # Neutral when not ready
        else:
            loss_value = self.current_loss
            direction_value = self.current_direction
        
        # Clamp to reasonable range and ensure positive
        loss_clamped = np.clip(loss_value, 0.0, 100.0)
        
        # Return as tensor with batch dimension [1, 1]
        observation = torch.tensor([[loss_clamped, direction_value]], 
                                 device=self.device, dtype=torch.float32)
        
        return observation
    
    # Probably redundant, remove if so
    # def calculate_reward(self, reward_scale=1.0, previous_loss=None):
    #     """
    #     Calculate reward based on spectral loss.
    #     Lower loss = higher reward.
        
    #     Args:
    #         reward_scale: Scale factor to apply to the reward
    #         previous_loss: Previous loss value for improvement calculation
        
    #     Returns:
    #         tuple: (reward, current_loss)
    #     """
    #     # Use stored previous loss if not provided
    #     if previous_loss is None:
    #         previous_loss = self.previous_loss
        
    #     # Base reward is negative loss (higher reward for smaller loss)
    #     base_reward = -self.current_loss
        
    #     # Add improvement bonus if we have previous loss
    #     improvement_bonus = 0
    #     if previous_loss is not None:
    #         improvement = previous_loss - self.current_loss
    #         if improvement > 0:
    #             # Reward improvement proportionally
    #             improvement_bonus = improvement * 5.0
        
    #     # Add bonus for approaching the best performance
    #     best_approach_bonus = 0
    #     if len(self.loss_history) > 1:
    #         recent_avg_loss = np.mean(list(self.loss_history)[-10:])  # Last 10 values
    #         if self.current_loss < recent_avg_loss:
    #             best_approach_bonus = (recent_avg_loss - self.current_loss) * 2.0
        
    #     # Calculate final reward
    #     reward = (base_reward + improvement_bonus + best_approach_bonus) * reward_scale
        
    #     # Update previous loss for next calculation
    #     self.previous_loss = self.current_loss
        
    #     # Log occasionally
    #     if np.random.random() < 0.05:  # Log about 5% of the time
    #         logger.debug(f"Loss-based reward: loss={self.current_loss:.4f}, "
    #                     f"base={base_reward*reward_scale:.4f}, "
    #                     f"improvement={improvement_bonus*reward_scale:.4f}, "
    #                     f"approach={best_approach_bonus*reward_scale:.4f}, "
    #                     f"total={reward:.4f}")
        
    #     return reward, self.current_loss
    
    def get_performance_stats(self):
        """
        Get statistics about the spectral loss performance.
        
        Returns:
            dict: Performance statistics
        """
        if not self.loss_history:
            return {
                'current_loss': self.current_loss if self.ready else float('inf'),
                'best_loss': self.best_loss,
                'avg_loss': float('inf'),
                'loss_std': 0.0,
                'data_points': 0
            }
        
        return {
            'current_loss': self.current_loss,
            'best_loss': self.best_loss,
            'avg_loss': np.mean(list(self.loss_history)),
            'loss_std': np.std(list(self.loss_history)),
            'data_points': len(self.loss_history)
        }
    
    def reset_stats(self):
        """
        Reset the performance statistics but keep the loss calculator connection.
        """
        self.best_loss = float('inf')
        self.loss_history.clear()
        self.previous_loss = None
        # Don't reset current_loss or ready state - keep receiving data
        
        logger.debug("SimpleLossProcessor stats reset")
    
    def get_current_loss(self):
        """Get the current loss value directly."""
        return self.current_loss
    
    def is_ready(self):
        """Check if the processor is ready (receiving loss data)."""
        return self.ready