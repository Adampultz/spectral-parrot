# stft_audio.py - REFACTORED FOR SYNCHRONIZATION
"""
Enhanced audio processing with synchronized stereo STFT and multi-scale spectral loss.

KEY CHANGES FOR SYNCHRONIZATION:
1. Single stereo buffer instead of separate channel buffers
2. Synchronized STFT computation for both channels
3. Immediate loss computation when both channels are ready
4. Eliminated timestamp-based synchronization race conditions
"""

import numpy as np
import time
from typing import Callable, List, Optional, Tuple, Dict
from scipy import signal
from collections import deque
import threading
import logging

from main_MSSL import SimpleAudio

try:
    import pyfftw
    PYFFTW_AVAILABLE = True
    print("pyfftw available - using optimized FFT")
except ImportError:
    PYFFTW_AVAILABLE = False
    print("pyfftw not available - falling back to numpy FFT")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StereoSTFTProcessor:
    """
    Synchronized stereo STFT processor - processes both channels together.
    This eliminates timing mismatches between channels.
    """
    
    def __init__(self, 
                 window_size: int = 2048, 
                 hop_length: int = 512, 
                 sample_rate: int = 48000,
                 window_type: str = 'hann',
                 use_pyfftw: bool = True,
                 fft_threads: int = 2):
        """
        Initialize synchronized stereo STFT processor.
        """
        if window_size <= 0 or not (window_size & (window_size - 1)) == 0:
            raise ValueError("Window size must be a positive power of 2")
        if hop_length <= 0 or hop_length > window_size:
            raise ValueError("Hop length must be positive and <= window_size")
            
        self.window_size = window_size
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.window_type = window_type
        self.use_pyfftw = use_pyfftw and PYFFTW_AVAILABLE
        self.fft_threads = fft_threads
        
        # Create window function
        if window_type == 'hann':
            self.window = np.hanning(window_size).astype(np.float32)
        elif window_type == 'hamming':
            self.window = np.hamming(window_size).astype(np.float32)
        elif window_type == 'blackman':
            self.window = np.blackman(window_size).astype(np.float32)
        else:
            self.window = np.ones(window_size, dtype=np.float32)
            
        self.freqs = np.fft.rfftfreq(window_size, 1/sample_rate)
        
        # SYNCHRONIZED STEREO BUFFERS - both channels in one structure
        self.stereo_buffer = np.zeros((2, window_size), dtype=np.float32)
        self.buffer_position = 0
        self.total_samples_processed = 0
        
        # Pre-allocated FFT buffers for BOTH channels
        self.reorder_buffers = np.zeros((2, window_size), dtype=np.float32)
        self.windowed_buffers = np.zeros((2, window_size), dtype=np.float32)
        self.fft_output_buffers = np.zeros((2, window_size // 2 + 1), dtype=np.complex64)
        self.magnitude_buffers = np.zeros((2, window_size // 2 + 1), dtype=np.float32)
        
        # Setup pyfftw if available
        if self.use_pyfftw:
            try:
                # Create aligned arrays for pyfftw
                self.fft_inputs = [
                    pyfftw.empty_aligned(window_size, dtype='float32'),
                    pyfftw.empty_aligned(window_size, dtype='float32')
                ]
                self.fft_outputs = [
                    pyfftw.empty_aligned(window_size // 2 + 1, dtype='complex64'),
                    pyfftw.empty_aligned(window_size // 2 + 1, dtype='complex64')
                ]
                
                # Create FFT objects for both channels
                self.fft_objects = [
                    pyfftw.FFTW(
                        self.fft_inputs[0], self.fft_outputs[0],
                        direction='FFTW_FORWARD',
                        flags=('FFTW_MEASURE',),
                        threads=fft_threads
                    ),
                    pyfftw.FFTW(
                        self.fft_inputs[1], self.fft_outputs[1],
                        direction='FFTW_FORWARD',
                        flags=('FFTW_MEASURE',),
                        threads=fft_threads
                    )
                ]
                logger.info(f"pyfftw initialized with {fft_threads} threads")
            except Exception as e:
                logger.warning(f"pyfftw initialization failed: {e}. Falling back to numpy FFT.")
                self.use_pyfftw = False
        
        # Callbacks for STFT results - now includes both channels
        self.stft_callbacks = []
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Performance monitoring
        self.fft_times = deque(maxlen=100)
        self.last_fft_time = 0
        
        total_kb = self._get_fft_buffer_memory()
        fft_method = "pyfftw" if self.use_pyfftw else "numpy"
        logger.info(f"Stereo STFT: Pre-allocated {total_kb:.1f} KB (using {fft_method})")
        
    def _get_fft_buffer_memory(self) -> float:
        """Calculate memory used by pre-allocated FFT buffers."""
        total_bytes = (
            self.stereo_buffer.nbytes +
            self.reorder_buffers.nbytes +
            self.windowed_buffers.nbytes + 
            self.fft_output_buffers.nbytes +
            self.magnitude_buffers.nbytes
        )
        
        if self.use_pyfftw:
            for i in range(2):
                total_bytes += self.fft_inputs[i].nbytes + self.fft_outputs[i].nbytes
            
        return total_bytes / 1024
        
    def add_stft_callback(self, callback: Callable):
        """
        Add a callback function to process STFT results.
        Callback signature: callback(ch1_freqs, ch1_mags, ch2_mags, ch1_complex, ch2_complex)
        """
        self.stft_callbacks.append(callback)
        
    def process_stereo_data(self, channel_1_data: np.ndarray, channel_2_data: np.ndarray):
        """
        Process synchronized stereo audio data.
        Both channels are processed together, eliminating timing mismatches.
        
        Args:
            channel_1_data: Audio samples for channel 1
            channel_2_data: Audio samples for channel 2
        """
        with self.lock:
            # Ensure both channels have same length
            n_samples = min(len(channel_1_data), len(channel_2_data))
            if n_samples == 0:
                return
            
            channel_1_data = channel_1_data[:n_samples].astype(np.float32)
            channel_2_data = channel_2_data[:n_samples].astype(np.float32)
            
            # Calculate hop boundaries
            initial_hop_index = self.total_samples_processed // self.hop_length
            
            # Fill circular buffer for both channels simultaneously
            src_idx = 0
            dst_idx = self.buffer_position
            
            while src_idx < n_samples:
                chunk_size = min(self.window_size - dst_idx, n_samples - src_idx)
                
                # Copy to both channel buffers simultaneously
                self.stereo_buffer[0, dst_idx:dst_idx + chunk_size] = \
                    channel_1_data[src_idx:src_idx + chunk_size]
                self.stereo_buffer[1, dst_idx:dst_idx + chunk_size] = \
                    channel_2_data[src_idx:src_idx + chunk_size]
                
                src_idx += chunk_size
                dst_idx = (dst_idx + chunk_size) % self.window_size
            
            # Update position tracking
            self.total_samples_processed += n_samples
            self.buffer_position = dst_idx
            
            # Calculate how many hops were completed
            final_hop_index = self.total_samples_processed // self.hop_length
            hops_completed = final_hop_index - initial_hop_index
            
            # Compute FFT for each completed hop (both channels together)
            for hop_num in range(hops_completed):
                start_time = time.perf_counter()
                self._compute_synchronized_fft()
                fft_time = time.perf_counter() - start_time
                self.fft_times.append(fft_time)
                self.last_fft_time = fft_time
                    
    def _compute_synchronized_fft(self):
        """
        Compute FFT for both channels simultaneously.
        This ensures perfect synchronization.
        """
        pos = self.buffer_position
        
        # Reorder both channel buffers
        for ch in range(2):
            if pos == 0:
                self.reorder_buffers[ch] = self.stereo_buffer[ch]
            else:
                remaining = self.window_size - pos
                self.reorder_buffers[ch, :remaining] = self.stereo_buffer[ch, pos:]
                self.reorder_buffers[ch, remaining:] = self.stereo_buffer[ch, :pos]
        
        # Apply window to both channels
        np.multiply(self.reorder_buffers[0], self.window, out=self.windowed_buffers[0])
        np.multiply(self.reorder_buffers[1], self.window, out=self.windowed_buffers[1])
        
        # Compute FFT for both channels
        if self.use_pyfftw:
            self.fft_inputs[0][:] = self.windowed_buffers[0]
            self.fft_inputs[1][:] = self.windowed_buffers[1]
            self.fft_objects[0]()
            self.fft_objects[1]()
            self.fft_output_buffers[0] = self.fft_outputs[0]
            self.fft_output_buffers[1] = self.fft_outputs[1]
        else:
            self.fft_output_buffers[0] = np.fft.rfft(self.windowed_buffers[0])
            self.fft_output_buffers[1] = np.fft.rfft(self.windowed_buffers[1])
        
        # Extract magnitudes for both channels
        np.abs(self.fft_output_buffers[0], out=self.magnitude_buffers[0])
        np.abs(self.fft_output_buffers[1], out=self.magnitude_buffers[1])
        
        # Call callbacks with synchronized data
        for callback in self.stft_callbacks:
            try:
                callback(
                    self.freqs,
                    self.magnitude_buffers[0],
                    self.magnitude_buffers[1],
                    self.fft_output_buffers[0],
                    self.fft_output_buffers[1]
                )
            except Exception as e:
                logger.error(f"Error in STFT callback: {e}")
                
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        if not self.fft_times:
            return {
                'avg_fft_time_ms': 0,
                'max_fft_time_ms': 0,
                'min_fft_time_ms': 0,
                'last_fft_time_ms': 0,
                'fft_method': 'pyfftw' if self.use_pyfftw else 'numpy'
            }
            
        times_ms = [t * 1000 for t in self.fft_times]
        
        return {
            'avg_fft_time_ms': np.mean(times_ms),
            'max_fft_time_ms': np.max(times_ms),
            'min_fft_time_ms': np.min(times_ms),
            'last_fft_time_ms': self.last_fft_time * 1000,
            'fft_method': 'pyfftw' if self.use_pyfftw else 'numpy'
        }


class MultiScaleSpectralLoss:
    """
    Multi-scale spectral loss calculator using synchronized stereo STFT.
    Uses hybrid precision: float32 for audio processing, float64 for loss computation.
    """
    
    def __init__(self, 
                 sample_rate=48000,
                 scales=None,
                 window_type='hann',
                 use_pyfftw=True,         
                 fft_threads=2,
                 use_normalized_loss=True,             
                 min_signal_threshold=0.1,              
                 weak_signal_penalty=50.0,              
                 normalization_method="l2"):
        """
        Initialize multi-scale spectral loss calculator with synchronized processing.
        """
        self.sample_rate = sample_rate
        self.window_type = window_type

        self.use_normalized_loss = use_normalized_loss
        self.min_signal_threshold = min_signal_threshold
        self.weak_signal_penalty = weak_signal_penalty
        self.normalization_method = normalization_method

        if use_normalized_loss:
            logger.info(f"Using normalized loss (method: {normalization_method})")
            logger.info(f"Min signal threshold: {min_signal_threshold}, weak signal penalty: {weak_signal_penalty}")
        else:
            logger.info("Using original non-normalized loss")
        
        if scales is None:
            self.scales = [512, 1024, 2048, 4096]
        else:
            self.scales = scales
            
        # Create synchronized stereo STFT processors for each scale
        self.stft_processors = {}
        total_memory_kb = 0
        
        for window_size in self.scales:
            hop_length = window_size // 4
            processor = StereoSTFTProcessor(
                window_size=window_size,
                hop_length=hop_length,
                sample_rate=sample_rate,
                window_type=window_type,
                use_pyfftw=use_pyfftw,
                fft_threads=fft_threads
            )
            
            # Add callback - now receives both channels simultaneously
            scale_key = f'scale_{window_size}'
            processor.add_stft_callback(
                lambda freqs, ch1_mags, ch2_mags, ch1_complex, ch2_complex, sk=scale_key: 
                self._store_spectral_data(sk, ch1_mags, ch2_mags)
            )
            
            self.stft_processors[scale_key] = processor
            total_memory_kb += processor._get_fft_buffer_memory()
            
        logger.info(f"Multi-scale Loss: {len(self.scales)} scales, {total_memory_kb:.1f} KB total")
            
        # Storage for spectral data - now stores both channels together
        self.spectral_data = {}  # Key: scale, Value: {'ch1': array, 'ch2': array}
        
        # Loss history
        self.loss_history = []
        self.current_losses = {}
        
        # Callbacks for loss results
        self.loss_callbacks = []
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Track ready state for each scale
        self.scales_ready = {f'scale_{ws}': False for ws in self.scales}
        
    def add_loss_callback(self, callback: Callable):
        """Add callback for loss computation results."""
        self.loss_callbacks.append(callback)

    def process_audio_channels(self, channel_1_data, channel_2_data):
        """
        Process audio data for both channels across all STFT scales.
        Channels are processed together, ensuring synchronization.
        """
        # Process all scales with synchronized stereo data
        for scale_name, processor in self.stft_processors.items():
            processor.process_stereo_data(channel_1_data, channel_2_data)
            
        # Compute loss if we have data from all scales
        self._compute_losses_if_ready()
            
    def _store_spectral_data(self, scale, ch1_magnitudes, ch2_magnitudes):
        """
        Store synchronized amplitude spectra from both channels.
        Both channels arrive together, eliminating timing mismatches.
        """
        with self.lock:
            self.spectral_data[scale] = {
                'ch1': ch1_magnitudes.copy(),
                'ch2': ch2_magnitudes.copy(),
                'timestamp': time.time()
            }
            self.scales_ready[scale] = True
    
    def _compute_losses_if_ready(self):
        """
        Compute losses if we have synchronized data from all scales.
        Much simpler now - no timestamp checking needed.
        """
        with self.lock:
            # Check if all scales are ready
            if not all(self.scales_ready.values()):
                return
            
            # We have data from all scales - compute loss immediately
            self._compute_spectral_loss()
            
            # Reset ready flags
            for scale in self.scales_ready:
                self.scales_ready[scale] = False
            
    def _compute_spectral_loss(self):
        """
        Compute multi-scale spectral loss from synchronized data.
        Guaranteed to have matched data from both channels.
        """
        total_loss = 0.0
        scale_losses = {}
        total_direction = 0
        
        for scale_name in self.spectral_data.keys():
            data = self.spectral_data[scale_name]
            
            # Convert to float64 for precision in loss computation
            stft_ch1 = data['ch1'].astype(np.float64)
            stft_ch2 = data['ch2'].astype(np.float64)
            
            # Flatten for loss computation
            stft_x = stft_ch1.flatten()
            stft_y = stft_ch2.flatten()
            
            # Compute loss based on selected method
            if self.use_normalized_loss:
                scale_loss, direction = self._compute_normalized_loss(stft_x, stft_y)
            else:
                scale_loss, direction = self._compute_original_loss(stft_x, stft_y)
            
            scale_losses[scale_name] = float(scale_loss)
            total_loss += scale_loss
            total_direction += direction
        
        # Average direction across scales
        avg_direction = np.sign(total_direction) if total_direction != 0 else 0
        
        # Store results - ensure all values are serializable
        loss_result = {
            'total_loss': float(total_loss),
            'scale_losses': scale_losses,
            'direction': float(avg_direction),
            'timestamp': time.time()
        }
        
        self.current_losses = loss_result
        self.loss_history.append(loss_result)
        
        # Call all registered callbacks
        for callback in self.loss_callbacks:
            try:
                callback(loss_result)
            except Exception as e:
                logger.error(f"Error in loss callback: {e}")
                import traceback
                logger.error(traceback.format_exc())

    def _compute_normalized_loss(self, stft_x, stft_y):
        """
        Compute normalized loss (volume-invariant).
        """
        eps = 1e-8
        norm_x = np.linalg.norm(stft_x) + eps
        norm_y = np.linalg.norm(stft_y) + eps
        
        # Safety check: if instrument signal is too weak, penalize heavily
        if norm_y < self.min_signal_threshold:
            logger.warning(f"Weak signal detected: {norm_y:.4f} < {self.min_signal_threshold}")
            return self.weak_signal_penalty, 1
        
        if self.normalization_method == "cosine":
            cosine_sim = np.dot(stft_x, stft_y) / (norm_x * norm_y)
            cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
            cosine_distance = 1.0 - cosine_sim
            scale_loss = cosine_distance * 10.0
            
        else:  # "l2"
            stft_x_norm = stft_x / norm_x
            stft_y_norm = stft_y / norm_y
            
            diff = stft_x_norm - stft_y_norm
            frobenius_loss = np.linalg.norm(diff)
            
            l1_loss = np.sum(np.abs(diff))
            log_l1 = np.log1p(l1_loss) if l1_loss > 1e-12 else 0.0
            
            scale_loss = frobenius_loss + log_l1
        
        direction = np.sign(norm_x - norm_y)
        
        return scale_loss, direction

    def _compute_original_loss(self, stft_x, stft_y):
        """
        Original non-normalized loss.
        """
        diff = stft_x - stft_y

        max_abs_diff = np.max(np.abs(diff))
        if max_abs_diff < 1e-12:
            return 0.0, 0

        overall_bias = np.sum(diff)
        direction = np.sign(overall_bias)
        
        frobenius_norm_diff = np.linalg.norm(diff)
        frobenius_norm_x = np.linalg.norm(stft_x)
        frobenius_norm_y = np.linalg.norm(stft_y)
        
        if frobenius_norm_x < 1e-8 and frobenius_norm_y < 1e-8:
            normalized_frobenius = 0.0
        elif frobenius_norm_x < 1e-8:
            if frobenius_norm_y > 1e-8:
                normalized_frobenius = frobenius_norm_diff / frobenius_norm_y
            else:
                normalized_frobenius = 0.0
        else:
            normalized_frobenius = frobenius_norm_diff / frobenius_norm_x
        
        l1_norm_diff = np.sum(np.abs(diff))
        log_l1 = np.log1p(l1_norm_diff) if l1_norm_diff > 1e-12 else 0.0
        
        scale_loss = normalized_frobenius + log_l1
        
        return scale_loss, direction
                    
    def get_current_losses(self):
        """Get the most recent loss computation."""
        with self.lock:
            return self.current_losses.copy() if self.current_losses else None
            
    def get_loss_history(self, n_recent=10):
        """Get recent loss history."""
        with self.lock:
            return self.loss_history[-n_recent:] if self.loss_history else []
            
    def get_average_losses(self, n_recent=10):
        """Get average losses over recent computations."""
        with self.lock:
            recent_history = self.loss_history[-n_recent:] if self.loss_history else []
                
        if not recent_history:
            return None
                
        total_losses = [h['total_loss'] for h in recent_history if 'total_loss' in h]
        if total_losses:
            return {'total_loss': np.mean(total_losses)}
        return None


class EnhancedAudio(SimpleAudio):
    """
    Enhanced audio class with synchronized stereo STFT and multi-scale spectral loss.
    """
    
    def __init__(self, 
                sample_rate=44100, 
                channels=2,
                buffer_size=1024, 
                input_device=None,
                output_device=None,
                stft_window_size=2048,
                stft_hop_length=512,
                stft_window_type='hann',
                enable_spectral_loss=True,
                stft_scales=None,        
                use_pyfftw=True,       
                fft_threads=2,
                use_normalized_loss=True,              
                min_signal_threshold=0.1,             
                weak_signal_penalty=50.0,             
                normalization_method="l2"):
        """
        Initialize enhanced audio system with synchronized stereo processing.
        """
        super().__init__(sample_rate, channels, buffer_size, input_device, output_device)
        
        # Initialize multi-scale spectral loss with synchronized processing
        self.enable_spectral_loss = enable_spectral_loss
        if enable_spectral_loss and channels >= 2:
            self.spectral_loss = MultiScaleSpectralLoss(
                sample_rate=sample_rate,
                scales=stft_scales if stft_scales else [512, 1024, 2048, 4096],
                window_type=stft_window_type,
                use_pyfftw=use_pyfftw,  
                fft_threads=fft_threads,
                use_normalized_loss=use_normalized_loss,          
                min_signal_threshold=min_signal_threshold,       
                weak_signal_penalty=weak_signal_penalty,         
                normalization_method=normalization_method
            )
        else:
            self.spectral_loss = None
        
        # Add STFT processing to the audio callback chain
        self.add_callback(self._stft_callback)
        
        # Storage for latest channel data
        self.latest_channel_data = {
            'channel_1': None,
            'channel_2': None
        }
        
        logger.info("EnhancedAudio initialized with synchronized stereo processing")
        
    def _stft_callback(self, audio_data: np.ndarray):
        """
        Internal callback to process audio through synchronized stereo STFT.
        """
        # Debug format on first call
        if not hasattr(self, '_format_logged'):
            self._format_logged = True
            logger.info(f"Audio format: dtype={audio_data.dtype}, shape={audio_data.shape}, "
                       f"range=[{np.min(audio_data):.3f}, {np.max(audio_data):.3f}]")
        
        # Normalize if needed
        if audio_data.dtype in [np.int16, np.int32, np.int64]:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_data = audio_data.astype(np.float32) / max_val
        else:
            audio_data = audio_data.astype(np.float32)
            max_val = np.max(np.abs(audio_data))
            if max_val > 10.0:
                logger.warning(f"Audio values up to {max_val:.1f} - possible format mismatch")
        
        # Separate channels
        if self.channels == 1:
            channel_1_data = audio_data
            channel_2_data = audio_data
        else:
            if len(audio_data.shape) == 1:
                # Interleaved stereo
                channel_1_data = audio_data[0::2]
                channel_2_data = audio_data[1::2]
            else:
                # Already separated
                channel_1_data = audio_data[:, 0] if audio_data.shape[1] > 0 else audio_data.flatten()
                channel_2_data = audio_data[:, 1] if audio_data.shape[1] > 1 else channel_1_data
        
        # Store latest channel data
        self.latest_channel_data['channel_1'] = channel_1_data
        self.latest_channel_data['channel_2'] = channel_2_data
        
        # Process through synchronized spectral loss
        if self.spectral_loss and self.channels >= 2:
            self.spectral_loss.process_audio_channels(channel_1_data, channel_2_data)
            
    def add_spectral_loss_callback(self, callback: Callable):
        """Add a callback for spectral loss results."""
        if self.spectral_loss:
            self.spectral_loss.add_loss_callback(callback)
            
    def get_current_loss(self):
        """Get the current spectral loss value."""
        if self.spectral_loss:
            return self.spectral_loss.get_current_losses()
        return None
        
    def get_loss_history(self, n_recent=10):
        """Get recent loss history."""
        if self.spectral_loss:
            return self.spectral_loss.get_loss_history(n_recent)
        return []