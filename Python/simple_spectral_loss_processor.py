# simple_spectral_loss_processor.py - REFACTORED
"""
Improved loss processor with better timing calculations and moving average.

KEY IMPROVEMENTS:
1. Dynamic calculation of expected loss rate based on actual data
2. More robust moving average with outlier rejection
3. Better handling of paused state
4. Improved ready state detection
"""

import torch
import numpy as np
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)


class SimpleLossProcessor:
    """
    Processes multi-scale spectral loss values from synchronized STFT.
    Provides smoothed loss observations for RL agent.
    """
    
    def __init__(self, 
                 spectral_loss_calculator, 
                 device='cpu', 
                 step_wait_time=1.0, 
                 loss_clip_max=50.0,
                 averaging_window_factor=1.0,
                 loss_history_buffer_size=200,
                 outlier_rejection_threshold=3.0):
        """
        Initialize the loss processor with adaptive timing.
        
        Args:
            spectral_loss_calculator: Instance of MultiScaleSpectralLoss
            device: The torch device to use ('cpu' or 'cuda')
            step_wait_time: Time to wait between motor steps (seconds)
            loss_clip_max: Maximum loss value before clipping
            averaging_window_factor: Multiplier for averaging window relative to step_wait_time
            loss_history_buffer_size: Maximum number of loss entries to keep
            outlier_rejection_threshold: Reject losses beyond this many std devs from mean
        """
        self.spectral_loss_calculator = spectral_loss_calculator
        self.device = device
        self.paused = False
        self.step_wait_time = step_wait_time
        self.loss_clip_max = loss_clip_max
        self.averaging_window_factor = averaging_window_factor
        self.outlier_rejection_threshold = outlier_rejection_threshold

        # Adaptive averaging window calculation
        self.averaging_window_seconds = step_wait_time * averaging_window_factor
        
        # Will be calculated dynamically based on actual loss rate
        self.averaging_entries = None
        self.loss_rate_hz = None  # Measured loss computation rate
        
        # Track current loss and history
        self.current_loss = 0.0
        self.current_direction = 0
        
        # Circular buffer with timestamps for adaptive window
        self.loss_history = deque(maxlen=loss_history_buffer_size)
        self.previous_loss = None
        
        # Track best performance
        self.best_loss = float('inf')
        
        # Ready state tracking
        self.ready = False
        self.data_received_count = 0
        self.first_loss_time = None
        self.last_loss_time = None
        
        # For measuring actual loss computation rate
        self.loss_timestamps = deque(maxlen=50)  # Track last 50 timestamps
        
        # Add callback to receive loss values
        self.spectral_loss_calculator.add_loss_callback(self._receive_loss_data)
        
        logger.info("SimpleLossProcessor initialized (adaptive timing)")
        logger.info(f"  Step wait time: {step_wait_time}s")
        logger.info(f"  Target averaging window: {self.averaging_window_seconds}s")
        logger.info(f"  Buffer size: {loss_history_buffer_size}")
        logger.info(f"  Outlier rejection: ±{outlier_rejection_threshold} std dev")
        logger.info(f"  Loss computation rate will be measured dynamically")

    def pause(self):
        """Pause loss processing (e.g., during motor calibration)."""
        self.paused = True
        self.spectral_loss_calculator.suppress_warnings = True
        logger.info("Loss processor PAUSED - weak signal warnings suppressed")

    def unpause(self):
        """Resume loss processing after pause."""
        self.paused = False
        self.spectral_loss_calculator.suppress_warnings = False
        logger.info("Loss processor UNPAUSED - resuming normal operation")
        
    def _measure_loss_rate(self):
        """
        Dynamically measure the actual loss computation rate.
        Returns the measured rate in Hz, or None if insufficient data.
        """
        if len(self.loss_timestamps) < 10:
            return None
        
        # Calculate time differences between consecutive losses
        timestamps = list(self.loss_timestamps)
        time_diffs = np.diff(timestamps)
        
        # Remove outliers (e.g., during pauses)
        median_diff = np.median(time_diffs)
        valid_diffs = time_diffs[time_diffs < median_diff * 3]
        
        if len(valid_diffs) == 0:
            return None
        
        # Average time between losses
        avg_period = np.mean(valid_diffs)
        
        if avg_period > 0:
            return 1.0 / avg_period
        return None
    
    def _update_averaging_window(self):
        """
        Update the number of entries to average based on measured loss rate.
        """
        rate = self._measure_loss_rate()
        
        if rate is not None and rate > 0:
            # Calculate how many loss values we expect in the averaging window
            expected_entries = int(rate * self.averaging_window_seconds)
            
            # Ensure reasonable bounds (at least 5, at most buffer_size/2)
            self.averaging_entries = max(5, min(expected_entries, len(self.loss_history) // 2))
            
            # Update cached rate
            if self.loss_rate_hz is None or abs(rate - self.loss_rate_hz) > 0.5:
                self.loss_rate_hz = rate
                logger.debug(f"Loss computation rate: {rate:.1f} Hz")
                logger.debug(f"Averaging over {self.averaging_entries} entries "
                           f"({self.averaging_entries/rate:.2f}s window)")
        elif self.averaging_entries is None:
            # Fallback: estimate based on typical STFT rates
            # For buffer_size=1024 at 48kHz with hop=512, expect ~93 Hz
            estimated_rate = self.spectral_loss_calculator.sample_rate / 512  # Conservative estimate
            self.averaging_entries = max(10, int(estimated_rate * self.averaging_window_seconds))
            logger.debug(f"Using estimated averaging window: {self.averaging_entries} entries")

    def _receive_loss_data(self, loss_result):
        """
        Callback to receive loss data from the spectral loss calculator.
        Now with outlier rejection and adaptive windowing.
        """
        if self.paused:
            return
        
        current_time = time.time()
        
        # Track timestamps for rate measurement
        self.loss_timestamps.append(current_time)
        
        # Update timing tracking
        if self.first_loss_time is None:
            self.first_loss_time = current_time
        self.last_loss_time = current_time
        
        # Extract loss value
        total_loss = loss_result.get('total_loss', 0.0)
        direction = loss_result.get('direction', 0)
        
        # Clip extreme values
        total_loss = np.clip(total_loss, 0.0, self.loss_clip_max)
        
        # Store in history with timestamp
        self.loss_history.append({
            'loss': total_loss,
            'direction': direction,
            'timestamp': current_time,
            'scale_losses': loss_result.get('scale_losses', {})
        })
        
        self.data_received_count += 1
        
        # Update averaging window periodically
        if self.data_received_count % 20 == 0:
            self._update_averaging_window()
        
        # Mark as ready after receiving enough data
        if not self.ready and self.data_received_count >= 5:
            self._update_averaging_window()  # Ensure we have valid averaging_entries
            self.ready = True
            rate_str = f"{self.loss_rate_hz:.1f}" if self.loss_rate_hz is not None else "~46"
            logger.info(f"SimpleLossProcessor ready - receiving loss values at ~{rate_str} Hz")
        
        # Update current values with robust averaging
        self._update_current_loss()
    
    def _update_current_loss(self):
        """
        Update current loss using robust moving average with outlier rejection.
        """
        if not self.loss_history:
            return
        
        # Ensure we have valid averaging_entries
        if self.averaging_entries is None:
            self._update_averaging_window()
        
        # Get recent losses within the averaging window
        n_avg = min(self.averaging_entries if self.averaging_entries else 10, 
                    len(self.loss_history))
        recent_losses = [entry['loss'] for entry in list(self.loss_history)[-n_avg:]]
        recent_directions = [entry['direction'] for entry in list(self.loss_history)[-n_avg:]]
        
        if len(recent_losses) < 2:
            # Not enough data for outlier rejection
            self.current_loss = recent_losses[-1]
            self.current_direction = recent_directions[-1]
            return
        
        # Calculate statistics for outlier rejection
        losses_array = np.array(recent_losses)
        mean_loss = np.mean(losses_array)
        std_loss = np.std(losses_array)
        
        # Reject outliers if we have enough data
        if len(recent_losses) >= 5 and std_loss > 0:
            # Keep only values within threshold standard deviations
            valid_mask = np.abs(losses_array - mean_loss) <= (self.outlier_rejection_threshold * std_loss)
            
            if np.sum(valid_mask) > 0:
                # Use filtered values
                filtered_losses = losses_array[valid_mask]
                filtered_directions = [d for d, v in zip(recent_directions, valid_mask) if v]
                
                self.current_loss = float(np.mean(filtered_losses))
                self.current_direction = int(np.sign(np.sum(filtered_directions)))
            else:
                # All values were outliers - use median instead
                self.current_loss = float(np.median(losses_array))
                self.current_direction = int(np.sign(np.sum(recent_directions)))
        else:
            # Not enough data or no variance - simple average
            self.current_loss = float(mean_loss)
            self.current_direction = int(np.sign(np.sum(recent_directions)))
        
        # Track best loss
        if self.current_loss < self.best_loss:
            self.best_loss = self.current_loss

    def get_observation(self):
        """
        Get the current loss value as an observation for the RL agent.
        Returns a tensor for compatibility with motor_environment.
        
        Returns:
            torch.Tensor: Observation tensor [1, 2] with [loss, direction]
        """
        if not self.ready:
            loss_value = 10.0
            direction_value = 0.0
        else:
            loss_value = self.current_loss
            direction_value = float(self.current_direction)
        
        # Return as tensor with batch dimension [1, 2] for compatibility
        observation = torch.tensor([[loss_value, direction_value]], 
                                   device=self.device, dtype=torch.float32)
        
        return observation

    def get_reward(self, baseline_loss=None):
        """
        Calculate reward based on loss improvement.
        
        Args:
            baseline_loss: Optional baseline to compare against
        
        Returns:
            Negative of current loss (lower loss = higher reward)
        """
        if not self.ready:
            return 0.0
        
        # Simple negative loss as reward
        reward = -self.current_loss
        
        # Optional: compare to baseline
        if baseline_loss is not None and self.previous_loss is not None:
            improvement = self.previous_loss - self.current_loss
            reward = improvement
        
        return reward

    def get_loss_stats(self):
        """
        Get statistics about recent loss values.
        """
        if not self.loss_history:
            return {
                'current_loss': 0.0,
                'mean_loss': 0.0,
                'std_loss': 0.0,
                'min_loss': 0.0,
                'max_loss': 0.0,
                'best_loss': float('inf'),
                'n_samples': 0,
                'loss_rate_hz': 0.0,
                'averaging_entries': 0
            }
        
        recent_losses = [entry['loss'] for entry in self.loss_history]
        
        return {
            'current_loss': self.current_loss,
            'mean_loss': float(np.mean(recent_losses)),
            'std_loss': float(np.std(recent_losses)),
            'min_loss': float(np.min(recent_losses)),
            'max_loss': float(np.max(recent_losses)),
            'best_loss': self.best_loss,
            'n_samples': len(self.loss_history),
            'loss_rate_hz': self.loss_rate_hz if self.loss_rate_hz else 0.0,
            'averaging_entries': self.averaging_entries if self.averaging_entries else 0
        }

    def reset(self):
        """
        Reset the loss processor state (e.g., at episode start).
        Clears history but keeps measured timing parameters.
        """
        self.loss_history.clear()
        self.current_loss = 0.0
        self.current_direction = 0
        self.previous_loss = None
        self.best_loss = float('inf')
        self.ready = False
        self.data_received_count = 0
        # Keep loss_rate_hz and averaging_entries - they're valid across episodes
        
        logger.info("Loss processor reset - cleared history, kept timing parameters")

    def reset_stats(self):
        """
        Alias for reset() - for backwards compatibility.
        """
        self.reset()

    def wait_until_ready(self, timeout=10.0):
        """
        Wait until the loss processor has received enough data to be ready.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if ready, False if timeout
        """
        start_time = time.time()
        
        while not self.ready:
            if time.time() - start_time > timeout:
                logger.warning(f"Loss processor not ready after {timeout}s timeout")
                return False
            time.sleep(0.1)
        
        return True

    def get_averaged_loss(self, use_robust=True):
        """
        Get the averaged loss value with optional robust estimation.
        
        Args:
            use_robust: If True, use outlier rejection
            
        Returns:
            Averaged loss value
        """
        if not self.loss_history:
            return 0.0
        
        if self.averaging_entries is None:
            self._update_averaging_window()
        
        n_avg = min(self.averaging_entries if self.averaging_entries else 10,
                    len(self.loss_history))
        recent_losses = [entry['loss'] for entry in list(self.loss_history)[-n_avg:]]
        
        if not use_robust or len(recent_losses) < 5:
            return float(np.mean(recent_losses))
        
        # Robust estimation
        losses_array = np.array(recent_losses)
        mean_loss = np.mean(losses_array)
        std_loss = np.std(losses_array)
        
        if std_loss > 0:
            valid_mask = np.abs(losses_array - mean_loss) <= (self.outlier_rejection_threshold * std_loss)
            if np.sum(valid_mask) > 0:
                return float(np.mean(losses_array[valid_mask]))
        
        return float(np.median(losses_array))

    def is_ready(self):
        """Check if the loss processor is ready."""
        return self.ready
    
    def get_direction(self):
        """Get the current direction indicator."""
        return self.current_direction
    
    def get_current_loss_value(self):
        """
        Alias for get_observation() - for backwards compatibility.
        Returns the current smoothed loss value.
        """
        return self.get_observation()
    
    def get_history_length(self):
        """Get the number of entries in loss history."""
        return len(self.loss_history)