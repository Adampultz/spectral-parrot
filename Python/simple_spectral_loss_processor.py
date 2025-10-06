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
    
    def __init__(self, spectral_loss_calculator, device='cpu', step_wait_time=1.0, loss_clip_max=50.0,
                 averaging_window_factor=1.0,loss_history_buffer_size=200):
        """
        Initialize the loss processor.
        
        Args:
            spectral_loss_calculator: Instance of MultiScaleSpectralLoss from stft_audio.py
            device: The torch device to use ('cpu' or 'cuda')
        """
        self.spectral_loss_calculator = spectral_loss_calculator
        self.device = device
        self.step_wait_time = step_wait_time
        self.loss_clip_max = loss_clip_max
        self.averaging_window_factor = averaging_window_factor

        # Calculate number of entries needed for averaging window (step_wait_time / 2)
        self.averaging_window_seconds = step_wait_time * averaging_window_factor
        self.averaging_entries = self._calculate_averaging_entries()
        
        # Track current loss and history
        self.current_loss = 0.0
        self.current_direction = 0
        self.loss_clip_max = loss_clip_max
        
        # Simple circular buffer - stores dict entries like before but no complex timestamp logic
        buffer_size = max(loss_history_buffer_size, self.averaging_entries * 2)  # At least 2x what we need
        self.loss_history = deque(maxlen=buffer_size)
        self.previous_loss = None
        
        # Track best performance for reward calculation
        self.best_loss = float('inf')
        
        # Track whether we've received data
        self.ready = False
        self.data_received_count = 0
        
        # Add callback to the spectral loss calculator to get loss values
        self.spectral_loss_calculator.add_loss_callback(self._receive_loss_data)
        logger.info("SimpleLossProcessor initialized - will receive loss from stft_audio.py")
        logger.info(f"  Step wait time: {step_wait_time}s")
        logger.info(f"  Averaging window: {self.averaging_window_seconds}s")
        logger.info(f"  Averaging over last {self.averaging_entries} entries")
        logger.info(f"  Buffer size: {buffer_size}")

    def clip_outlier_loss(self, current_loss, min_threshold=0.0, max_threshold=50.0):
        """Clip losses to absolute thresholds"""
        if current_loss < min_threshold:
            logger.warning(f"Extreme low loss clipped: {current_loss:.2f} -> {min_threshold}")
            return min_threshold
        elif current_loss > max_threshold:
            logger.warning(f"Extreme high loss clipped: {current_loss:.2f} -> {max_threshold}")
            return max_threshold
        else:
            return current_loss
        
    def _calculate_averaging_entries(self):
        """
        Calculate how many loss history entries we need for the averaging window.
        Based on estimated update frequency of ~21ms (48kHz, largest hop=1024).
        
        Returns:
            int: Number of entries to average over
        """
        # Estimate: loss updates every ~21ms (based on 1024 samples at 48kHz)
        estimated_update_period = 0.021  # seconds
        
        entries_needed = int(np.ceil(self.averaging_window_seconds / estimated_update_period))
        
        # Ensure we have at least a few entries, but not too many
        entries_needed = max(5, min(entries_needed, 200))
        
        logger.debug(f"Averaging entries: {self.averaging_window_seconds}s / {estimated_update_period}s ≈ {entries_needed}")
        return entries_needed  

    def update_step_wait_time(self, new_step_wait_time):
        """
        Update the step wait time and recalculate averaging parameters.
        
        Args:
            new_step_wait_time: New step wait time in seconds
        """
        self.step_wait_time = new_step_wait_time
        self.averaging_window_seconds = new_step_wait_time
        self.averaging_entries = self._calculate_averaging_entries()
        
        logger.info(f"Updated step wait time to {new_step_wait_time}s")
        logger.info(f"New averaging window: {self.averaging_window_seconds}s ({self.averaging_entries} entries)")
          
        
    def _receive_loss_data(self, loss_data):
        """
        Callback to receive loss data from the MultiScaleSpectralLoss calculator.
        Simple approach - just store in circular buffer.
        
        Args:
            loss_data: Dictionary with 'total_loss' and other loss information
        """
        if loss_data and 'total_loss' in loss_data:
            raw_loss = float(loss_data['total_loss'])
        
            # Apply outlier detection and clipping
            self.current_loss = self.clip_outlier_loss(raw_loss, min_threshold=0.0, max_threshold=self.loss_clip_max)
            
            # Log when clipping occurs
            if abs(raw_loss - self.current_loss) > 0.01:
                logger.warning(f"Loss clipped: {raw_loss:.2f} -> {self.current_loss:.2f}")
                self.current_loss = float(loss_data['total_loss'])
                self.current_direction = float(loss_data.get('direction', 0.0))
            
            # Store in simple circular buffer - no timestamps needed!
            entry = {
                'loss': self.current_loss,
                'direction': self.current_direction
            }
            self.loss_history.append(entry)
            
            # Update best loss
            if self.current_loss < self.best_loss:
                self.best_loss = self.current_loss
            
            # Track data reception
            self.data_received_count += 1
            if not self.ready and self.data_received_count >= 5:
                self.ready = True
                logger.info("SimpleLossProcessor ready - receiving loss values")
                
            # Log occasionally
            if self.data_received_count % 50 == 0:
                logger.debug(f"Loss update #{self.data_received_count}: {self.current_loss:.6f}")
        
    
    def get_observation(self):
        """
        Get observation averaged over the last N entries (simple circular buffer approach).
        
        Returns:
            torch.Tensor: The observation tensor with averaged loss and direction
        """
        if not self.ready:
            logger.warning("Attempting to get observation before loss processor is ready")
            loss_value = 10.0
            direction_value = 0.0
        else:
            # Simple approach: average over the last N entries
            num_recent = min(self.averaging_entries, len(self.loss_history))
            
            if num_recent > 0:
                # Get the last N entries from the circular buffer
                recent_entries = list(self.loss_history)[-num_recent:]
                
                # Average them
                loss_value = np.mean([entry['loss'] for entry in recent_entries])
                
                # Average direction then discretize back to -1, 0, or 1
                direction_avg = np.mean([entry['direction'] for entry in recent_entries])
                if direction_avg > 0.1:
                    direction_value = 1.0
                elif direction_avg < -0.1:
                    direction_value = -1.0
                else:
                    direction_value = 0.0
                
                # Occasionally log the averaging info
                if num_recent > 1 and np.random.random() < 0.05:
                    logger.debug(f"Averaged {num_recent} entries: {loss_value:.4f} "
                                f"(current: {self.current_loss:.4f})")
            else:
                # Fall back to current values if no history
                loss_value = self.current_loss
                direction_value = self.current_direction
        
        # Clamp to reasonable range and ensure positive
        loss_clamped = np.clip(loss_value, 0.0, 100.0)
        
        # Return as tensor with batch dimension [1, 1]
        observation = torch.tensor([[loss_clamped, direction_value]], 
                                 device=self.device, dtype=torch.float32)
        
        return observation
    
    def get_instantaneous_observation(self):
        """
        Get the instantaneous (non-averaged) observation for debugging.
        
        Returns:
            torch.Tensor: The instantaneous observation tensor
        """
        if not self.ready:
            loss_value = 10.0
            direction_value = 0.0
        else:
            loss_value = self.current_loss
            direction_value = self.current_direction
        
        loss_clamped = np.clip(loss_value, 0.0, 100.0)
        
        observation = torch.tensor([[loss_clamped, direction_value]], 
                                 device=self.device, dtype=torch.float32)
        
        return observation
    
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
                'data_points': 0,
                'averaging_entries': self.averaging_entries,
                'averaging_window': self.averaging_window_seconds
            }
        
        # Get recent values for averaging stats
        num_recent = min(self.averaging_entries, len(self.loss_history))
        recent_entries = list(self.loss_history)[-num_recent:] if num_recent > 0 else []
        recent_losses = [entry['loss'] for entry in recent_entries] if recent_entries else []
        
        return {
            'current_loss': self.current_loss,
            'averaged_loss': np.mean(recent_losses) if recent_losses else self.current_loss,
            'best_loss': self.best_loss,
            'avg_loss': np.mean([entry['loss'] for entry in self.loss_history]),
            'loss_std': np.std(recent_losses) if recent_losses else 0.0,
            'data_points': len(recent_entries),
            'total_data_points': len(self.loss_history),
            'averaging_entries': self.averaging_entries,
            'averaging_window': self.averaging_window_seconds
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
        """Get the current (instantaneous) loss value directly."""
        return self.current_loss
    
    def is_ready(self):
        """Check if the processor is ready (receiving loss data)."""
        return self.ready