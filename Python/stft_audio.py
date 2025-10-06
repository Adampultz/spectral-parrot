# stft_audio.py
"""
Enhanced audio processing with real-time multi-scale spectral loss capabilities.
Extends the SimpleAudio class from main_MSSL.py

OPTIMIZATIONS:
- Pre-allocated FFT buffers to eliminate real-time allocations
- Hybrid float32/float64 precision for performance and accuracy
- Integrated CPU monitoring for performance measurement
- REMOVED phase/angle calculations for better performance
- ADDED batch processing for improved buffer management
"""

import numpy as np
import time
from typing import Callable, List, Optional, Tuple
from scipy import signal
from collections import deque
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Import the original SimpleAudio class
from main_MSSL import SimpleAudio

# Import CPU monitoring
from cpu_monitor import AudioPerformanceMonitor

try:
    import pyfftw
    PYFFTW_AVAILABLE = True
    print("pyfftw available - using optimized FFT")
except ImportError:
    PYFFTW_AVAILABLE = False
    print("pyfftw not available - falling back to numpy FFT")
    print("Install with: pip install pyfftw")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class STFTProcessor:
    """
    Real-time Short-Time Fourier Transform processor for audio channels.
    OPTIMIZED: Pre-allocated buffers and pyfftw for maximum performance.
    """
    
    def __init__(self, 
                 window_size: int = 2048, 
                 hop_length: int = 512, 
                 sample_rate: int = 48000,
                 window_type: str = 'hann',
                 use_pyfftw: bool = True,
                 fft_threads: int = 2):
        """
        Initialize STFT processor with pre-allocated buffers and optional pyfftw.
        
        Args:
            window_size (int): Size of the FFT window (must be power of 2)
            hop_length (int): Number of samples between successive frames
            sample_rate (int): Audio sample rate
            window_type (str): Type of window function ('hann', 'hamming', 'blackman')
            use_pyfftw (bool): Use pyfftw if available (default: True)
            fft_threads (int): Number of threads for pyfftw (default: 2)
        """
        # Validate parameters
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
        
        # Create window function (float32 for audio processing)
        if window_type == 'hann':
            self.window = np.hanning(window_size).astype(np.float32)
        elif window_type == 'hamming':
            self.window = np.hamming(window_size).astype(np.float32)
        elif window_type == 'blackman':
            self.window = np.blackman(window_size).astype(np.float32)
        else:
            self.window = np.ones(window_size, dtype=np.float32)
            
        # Frequency bins (keep float64 for precision in calculations)
        self.freqs = np.fft.rfftfreq(window_size, 1/sample_rate)
        
        # Audio buffers (float32 for memory efficiency)
        self.buffers = {
            'channel_1': np.zeros(window_size, dtype=np.float32),
            'channel_2': np.zeros(window_size, dtype=np.float32)
        }
        
        # Position in buffer - UPDATED for batch processing
        self.buffer_positions = {
            'channel_1': 0,
            'channel_2': 0,
            'channel_1_total': 0,  # Total samples processed
            'channel_2_total': 0   # Total samples processed
        }
        
        # PRE-ALLOCATED FFT BUFFERS (eliminates real-time allocations)
        
        # 1. Buffer for reordering circular buffer (replaces np.concatenate)
        self.reorder_buffer = np.zeros(window_size, dtype=np.float32)
        
        # 2. Buffer for windowed data (replaces ordered_buffer * self.window)
        self.windowed_buffer = np.zeros(window_size, dtype=np.float32)
        
        # 3. Buffer for FFT output (replaces np.fft.rfft allocation)
        self.fft_output_buffer = np.zeros(window_size//2 + 1, dtype=np.complex64)
        
        # 4. Buffer for magnitude results (phase buffer kept for compatibility but unused)
        self.magnitude_buffer = np.zeros(window_size//2 + 1, dtype=np.float32)
        self.phase_buffer = np.zeros(window_size//2 + 1, dtype=np.float32)  # Kept for API compatibility
        
        # PYFFTW SETUP - Pre-plan FFT for maximum performance
        if self.use_pyfftw:
            try:
                # Create aligned arrays for pyfftw
                self.fft_input = pyfftw.empty_aligned(window_size, dtype='float32')
                self.fft_output_pyfftw = pyfftw.empty_aligned(window_size//2 + 1, dtype='complex64')
                
                # Plan the FFT (this takes time but makes execution much faster)
                # FFTW_MEASURE would be more optimal but takes longer to plan
                self.fft_plan = pyfftw.FFTW(
                    self.fft_input, 
                    self.fft_output_pyfftw,
                    direction='FFTW_FORWARD',
                    flags=['FFTW_ESTIMATE'],  # Fast planning
                    threads=self.fft_threads
                )
                
                # Also create inverse FFT plan if needed later
                self.ifft_input = pyfftw.empty_aligned(window_size//2 + 1, dtype='complex64')
                self.ifft_output = pyfftw.empty_aligned(window_size, dtype='float32')
                self.ifft_plan = pyfftw.FFTW(
                    self.ifft_input,
                    self.ifft_output,
                    direction='FFTW_BACKWARD',
                    flags=['FFTW_ESTIMATE'],
                    threads=self.fft_threads
                )
                
                logger.info(f"pyfftw initialized with {self.fft_threads} threads")
                
            except Exception as e:
                logger.warning(f"Failed to initialize pyfftw: {e}. Falling back to numpy FFT.")
                self.use_pyfftw = False
        
        # Callbacks for STFT results
        self.stft_callbacks = []
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Performance monitoring
        self.fft_times = deque(maxlen=100)
        self.last_fft_time = 0
        
        # Print memory usage info
        total_kb = self._get_fft_buffer_memory()
        fft_method = "pyfftw" if self.use_pyfftw else "numpy"
        print(f"STFT Processor: Pre-allocated {total_kb:.1f} KB of FFT buffers (using {fft_method})")
        
    def _get_fft_buffer_memory(self) -> float:
        """Calculate memory used by pre-allocated FFT buffers."""
        total_bytes = (
            self.reorder_buffer.nbytes +
            self.windowed_buffer.nbytes + 
            self.fft_output_buffer.nbytes +
            self.magnitude_buffer.nbytes +
            self.phase_buffer.nbytes
        )
        
        # Add pyfftw buffers if used
        if self.use_pyfftw:
            total_bytes += (
                self.fft_input.nbytes +
                self.fft_output_pyfftw.nbytes +
                self.ifft_input.nbytes +
                self.ifft_output.nbytes
            )
            
        return total_bytes / 1024
        
    def add_stft_callback(self, callback: Callable):
        """
        Add a callback function to process STFT results.
        
        Args:
            callback: Function that takes (channel, freqs, magnitudes, phases, stft_complex)
                     Note: phases will be None as phase calculation is disabled
        """
        self.stft_callbacks.append(callback)
        
    def process_channel_data(self, channel_name: str, audio_data: np.ndarray):
        """
        Process audio data for a specific channel using batch operations.
        Optimized to reduce sample-by-sample processing overhead.
        
        Args:
            channel_name: 'channel_1' or 'channel_2'
            audio_data: Audio samples for this channel
        """
        with self.lock:
            # Convert to float32
            audio_data = audio_data.astype(np.float32)
            n_samples = len(audio_data)
            
            if n_samples == 0:
                return
            
            # Get current state
            buffer = self.buffers[channel_name]
            pos = self.buffer_positions[channel_name]
            
            # Track how many samples have been processed since last reset
            # This is important for accurate hop boundary detection
            samples_since_start = self.buffer_positions.get(f'{channel_name}_total', pos)
            
            # Calculate initial hop position
            initial_hop_index = samples_since_start // self.hop_length
            
            # Copy audio data to circular buffer in chunks
            src_idx = 0  # Index in source audio_data
            dst_idx = pos  # Index in destination buffer
            
            while src_idx < n_samples:
                # Calculate chunk size (don't go past buffer end)
                chunk_size = min(self.window_size - dst_idx, n_samples - src_idx)
                
                # Copy chunk
                buffer[dst_idx:dst_idx + chunk_size] = audio_data[src_idx:src_idx + chunk_size]
                
                # Update indices
                src_idx += chunk_size
                dst_idx = (dst_idx + chunk_size) % self.window_size
            
            # Update total samples processed
            samples_since_start += n_samples
            self.buffer_positions[f'{channel_name}_total'] = samples_since_start
            
            # Calculate final hop position
            final_hop_index = samples_since_start // self.hop_length
            hops_completed = final_hop_index - initial_hop_index
            
            # Update buffer position
            self.buffer_positions[channel_name] = dst_idx
            
            # Perform FFT for each completed hop
            for hop_num in range(hops_completed):
                # Time the FFT computation
                start_time = time.perf_counter()
                
                # Use optimized FFT computation
                if self.use_pyfftw:
                    self._compute_fft_with_pyfftw(channel_name)
                else:
                    self._compute_fft_with_preallocation(channel_name)
                
                # Record timing
                fft_time = time.perf_counter() - start_time
                self.fft_times.append(fft_time)
                self.last_fft_time = fft_time
                    
    def _compute_fft_with_pyfftw(self, channel_name: str):
        """
        Compute FFT using pyfftw for maximum performance.
        Phase calculation removed for better performance.
        """
        pos = self.buffer_positions[channel_name]
        buffer = self.buffers[channel_name]
        
        # OPTIMIZATION 1: Reorder buffer without allocation
        if pos == 0:
            # Buffer is already in order, just copy
            self.reorder_buffer[:] = buffer
        else:
            # Manually reorder into pre-allocated buffer
            remaining = self.window_size - pos
            self.reorder_buffer[:remaining] = buffer[pos:]
            self.reorder_buffer[remaining:] = buffer[:pos]
        
        # OPTIMIZATION 2: Apply window directly into pyfftw input buffer
        # This combines reordering and windowing
        np.multiply(self.reorder_buffer, self.window, out=self.fft_input)
        
        # OPTIMIZATION 3: Execute pre-planned FFT
        # This is MUCH faster than numpy's FFT
        self.fft_plan()  # Results are now in self.fft_output_pyfftw
        
        # Copy to our standard output buffer for compatibility
        self.fft_output_buffer[:] = self.fft_output_pyfftw
        
        # OPTIMIZATION 4: Extract magnitude only (phase calculation removed)
        np.abs(self.fft_output_buffer, out=self.magnitude_buffer)
        # Phase calculation removed: self.phase_buffer[:] = np.angle(self.fft_output_buffer)
        
        # Call callbacks with pre-allocated buffers (None for phase)
        for callback in self.stft_callbacks:
            try:
                callback(channel_name, self.freqs, self.magnitude_buffer, 
                        None, self.fft_output_buffer)  # Pass None for phase
            except Exception as e:
                logger.error(f"Error in STFT callback: {e}")
                    
    def _compute_fft_with_preallocation(self, channel_name: str):
        """
        Compute FFT using pre-allocated buffers (numpy fallback).
        Phase calculation removed for better performance.
        """
        pos = self.buffer_positions[channel_name]
        buffer = self.buffers[channel_name]
        
        # OPTIMIZATION 1: Reorder buffer without allocation
        if pos == 0:
            # Buffer is already in order, just copy
            self.reorder_buffer[:] = buffer
        else:
            # Manually reorder into pre-allocated buffer
            remaining = self.window_size - pos
            self.reorder_buffer[:remaining] = buffer[pos:]
            self.reorder_buffer[remaining:] = buffer[:pos]
        
        # OPTIMIZATION 2: Apply window without allocation  
        np.multiply(self.reorder_buffer, self.window, out=self.windowed_buffer)
        
        # OPTIMIZATION 3: FFT without allocation
        self.fft_output_buffer[:] = np.fft.rfft(self.windowed_buffer)
        
        # OPTIMIZATION 4: Extract magnitude only (phase calculation removed)
        np.abs(self.fft_output_buffer, out=self.magnitude_buffer)
        # Phase calculation removed: self.phase_buffer[:] = np.angle(self.fft_output_buffer)
        
        # Call callbacks with pre-allocated buffers (None for phase)
        for callback in self.stft_callbacks:
            try:
                callback(channel_name, self.freqs, self.magnitude_buffer, 
                        None, self.fft_output_buffer)  # Pass None for phase
            except Exception as e:
                logger.error(f"Error in STFT callback: {e}")
                
    def get_performance_stats(self) -> dict:
        """Get performance statistics for FFT computation."""
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
        
    def process_audio_chunk(self, audio_data: np.ndarray, channels: int = 2):
        """
        Process incoming audio data and compute STFT when enough samples are available.
        Legacy method for compatibility.
        
        Args:
            audio_data: Audio data as numpy array
            channels: Number of audio channels
        """
        if channels == 1:
            # Mono audio
            channel_1_data = audio_data.astype(np.float32)
            self.process_channel_data('channel_1', channel_1_data)
            self.process_channel_data('channel_2', channel_1_data)  # Duplicate to both
        else:
            # Handle stereo audio
            if len(audio_data.shape) == 1:
                # Interleaved stereo data
                channel_1_data = audio_data[0::2].astype(np.float32)
                channel_2_data = audio_data[1::2].astype(np.float32)
            else:
                # Already separated channels
                channel_1_data = audio_data[:, 0].astype(np.float32) if audio_data.shape[1] > 0 else audio_data.flatten().astype(np.float32)
                channel_2_data = audio_data[:, 1].astype(np.float32) if audio_data.shape[1] > 1 else channel_1_data
            
            self.process_channel_data('channel_1', channel_1_data)
            self.process_channel_data('channel_2', channel_2_data)
            
    def cleanup(self):
        """Clean up resources (important for pyfftw)."""
        if self.use_pyfftw:
            # Clean up FFTW plans
            try:
                del self.fft_plan
                del self.ifft_plan
                # FFTW cleanup
                pyfftw.forget_wisdom()
            except:
                pass

class MultiScaleSpectralLoss:
    """
    Multi-scale spectral loss calculator.
    Uses hybrid precision: float32 for audio processing, float64 for loss computation.
    """
    
    def __init__(self, 
                 sample_rate=48000,
                 scales=None,
                 window_type='hann'):
        """
        Initialize multi-scale spectral loss calculator.
        
        Args:
            sample_rate (int): Audio sample rate
            scales (list): List of STFT window sizes (hop_length = window_size/4)
            window_type (str): Window function type
        """
        self.sample_rate = sample_rate
        self.window_type = window_type
        
        # Default scales (STFT window sizes)
        if scales is None:
            self.scales = [512, 1024, 2048, 4096]
        else:
            self.scales = scales
            
        # Create STFT processors for each scale (window_size, hop_length = window_size/4)
        self.stft_processors = {}
        total_memory_kb = 0
        for window_size in self.scales:
            hop_length = window_size // 4  # Use hop size n/4, overlap of 0.75 
            processor = STFTProcessor(
                window_size=window_size,
                hop_length=hop_length,
                sample_rate=sample_rate,
                window_type=window_type
            )
            
            # Add callback for this specific processor and scale
            scale_key = f'scale_{window_size}'
            processor.add_stft_callback(
                lambda ch, freqs, mags, phases, complex_data, sk=scale_key: 
                self._store_spectral_data(sk, ch, mags)
            )
            
            self.stft_processors[scale_key] = processor
            total_memory_kb += processor._get_fft_buffer_memory()
            
        print(f"Multi-scale Loss: {len(self.scales)} scales, {total_memory_kb:.1f} KB total FFT buffers")
            
        # Storage for spectral data (amplitude spectra)
        self.spectral_data = {
            'channel_1': {},
            'channel_2': {}
        }
        
        # Loss history
        self.loss_history = []
        self.current_losses = {}
        
        # Callbacks for loss results
        self.loss_callbacks = []
        
        # Lock for thread safety
        self.lock = threading.Lock()

        self.executor = ThreadPoolExecutor(max_workers=len(self.scales))
        
    def add_loss_callback(self, callback: Callable):
        """Add callback for loss computation results."""
        self.loss_callbacks.append(callback)
        
    def process_audio_channels(self, channel_1_data: np.ndarray, channel_2_data: np.ndarray):
        # Process all scales sequentially - much faster for small buffers!
        for scale_name, processor in self.stft_processors.items():
            processor.process_channel_data('channel_1', channel_1_data)
            processor.process_channel_data('channel_2', channel_2_data)
            
        # Check if we have data from both channels for all scales to compute loss
        self._compute_losses_if_ready()
            
    def _store_spectral_data(self, scale, channel, magnitudes):
        """Store amplitude spectra for loss computation."""
        with self.lock:
            # Store reference to magnitude buffer (magnitudes are float32)
            # Will convert to float64 in loss computation for precision

            if scale in self.spectral_data[channel]:
                # Only keep the most recent data
                pass  # Current implementation already overwrites

            self.spectral_data[channel][scale] = {
                'amplitude': magnitudes.copy(),  # Copy needed since buffer is reused
                'timestamp': time.time()
            }
    
    def cleanup_old_spectral_data(self, max_age=1.0):
        """Remove spectral data older than max_age seconds."""
        current_time = time.time()
        with self.lock:
            for channel in self.spectral_data:
                for scale in list(self.spectral_data[channel].keys()):
                    if current_time - self.spectral_data[channel][scale]['timestamp'] > max_age:
                        del self.spectral_data[channel][scale]
        
    def _compute_losses_if_ready(self):
        """Compute losses if we have data from both channels for all scales."""
        with self.lock:
            # Check if we have recent data from both channels for all scales
            current_time = time.time()
            max_age = 0.1  # 100ms tolerance
            
            ready_scales = []
            for scale_name in self.stft_processors.keys():
                if (scale_name in self.spectral_data['channel_1'] and 
                    scale_name in self.spectral_data['channel_2']):
                    
                    ch1_time = self.spectral_data['channel_1'][scale_name]['timestamp']
                    ch2_time = self.spectral_data['channel_2'][scale_name]['timestamp']
                    
                    if (current_time - ch1_time < max_age and 
                        current_time - ch2_time < max_age):
                        ready_scales.append(scale_name)
            
            if len(ready_scales) == len(self.stft_processors):
                self._compute_spectral_loss(ready_scales)
            
    def _compute_spectral_loss(self, scales):
        """
        Compute multi-scale spectral loss.
        S(x, y) = Σ_{n∈N} [‖STFT_n(x) - STFT_n(y)‖_F / ‖STFT_n(x)‖_F + log(‖STFT_n(x) - STFT_n(y)‖_1)]
        """
        total_loss = 0.0
        summed_directions = 0
        scale_losses = {}
        
        for scale_name in scales:
            ch1_data = self.spectral_data['channel_1'][scale_name]
            ch2_data = self.spectral_data['channel_2'][scale_name]
            
            scale_loss, direction = self._compute_single_scale_loss(ch1_data, ch2_data)
            scale_losses[scale_name] = scale_loss
            
            # Add to total loss (sum over all scales as in RAVE equation 5)
            total_loss += scale_loss
            summed_directions += direction

        final_direction = np.sign(summed_directions)  
            
        # Store current losses
        self.current_losses = {
            'total_loss': total_loss,
            'by_scale': scale_losses,
            'direction': final_direction,  # Will be -1, 0, or +1
            'timestamp': time.time()
        }
        
        # Add to history
        self.loss_history.append(self.current_losses.copy())
        
        # Keep only recent history (last 100 computations)
        if len(self.loss_history) > 100:
            self.loss_history.pop(0)
            
        # Call callbacks
        for callback in self.loss_callbacks:
            try:
                callback(self.current_losses)
            except Exception as e:
                print(f"Error in loss callback: {e}")
                
    def _compute_single_scale_loss(self, ch1_data, ch2_data):
        """
        Compute spectral loss at a single scale using shape-based comparison.
        This ignores overall volume differences and focuses on spectral shape.
        FIXED: Handles identical signals correctly and removes log bias.
        """
        # Get amplitude spectra (these are float32 from STFT)
        stft_x_f32 = ch1_data['amplitude']  
        stft_y_f32 = ch2_data['amplitude']  
        
        # Convert to float64 for precise loss computation
        stft_x = stft_x_f32.astype(np.float64)
        stft_y = stft_y_f32.astype(np.float64)
        
        # Ensure same length (should already be, but just in case)
        min_len = min(len(stft_x), len(stft_y))
        stft_x = stft_x[:min_len]
        stft_y = stft_y[:min_len]
        
        # Compute difference
        diff = stft_x - stft_y

        # EARLY EXIT: If signals are truly identical, return 0 immediately
        max_abs_diff = np.max(np.abs(diff))
        if max_abs_diff < 1e-12:  # More stringent threshold
            return 0.0, 0

        # Get the sign of the difference
        overall_bias = np.sum(diff)
        direction = np.sign(overall_bias)
        
        # Component 1: Normalized Frobenius norm
        frobenius_norm_diff = np.linalg.norm(diff)
        frobenius_norm_x = np.linalg.norm(stft_x)
        frobenius_norm_y = np.linalg.norm(stft_y)
        
        # Handle silent/identical signals properly
        if frobenius_norm_x < 1e-8 and frobenius_norm_y < 1e-8:
            # Both signals are silent - no difference
            normalized_frobenius = 0.0
        elif frobenius_norm_x < 1e-8:
            # Reference is silent but target has signal - use target norm
            if frobenius_norm_y > 1e-8:
                normalized_frobenius = frobenius_norm_diff / frobenius_norm_y
            else:
                normalized_frobenius = 0.0
        else:
            # Normal case - reference has signal
            normalized_frobenius = frobenius_norm_diff / frobenius_norm_x
        
        # Component 2: FIXED Log L1 norm
        l1_norm_diff = np.sum(np.abs(diff))
        
        # FIX: Use small epsilon instead of +1.0 to avoid bias
        if l1_norm_diff > 1e-12:  # Stricter threshold
            eps = 1e-15  # Small epsilon to prevent log(0)
            log_l1 = np.log(l1_norm_diff + eps)
        else:
            log_l1 = 0.0  # No difference = no log component
        
        # Total loss for this scale
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
            
        # Average the total loss
        total_losses = [h['total_loss'] for h in recent_history if 'total_loss' in h]
        if total_losses:
            return {'total_loss': np.mean(total_losses)}
        return None


class EnhancedAudio(SimpleAudio):
    """
    Enhanced audio class with STFT processing and multi-scale spectral loss.
    Inherits from the original SimpleAudio class.
    """
    
    def __init__(self, 
                 sample_rate=44100, 
                 channels=2,  # Default to stereo for spectral loss
                 buffer_size=1024, 
                 input_device=None,
                 output_device=None,
                 stft_window_size=2048,
                 stft_hop_length=512,
                 stft_window_type='hann',
                 enable_spectral_loss=True):
        """
        Initialize enhanced audio system with STFT processing and spectral loss.
        """
        # Initialize parent class
        super().__init__(sample_rate, channels, buffer_size, input_device, output_device)
        
        # Initialize STFT processor (kept for API compatibility)
        self.stft_processor = STFTProcessor(
            window_size=stft_window_size,
            hop_length=stft_hop_length,
            sample_rate=sample_rate,
            window_type=stft_window_type
        )
        
        # Initialize multi-scale spectral loss if enabled
        self.enable_spectral_loss = enable_spectral_loss
        if enable_spectral_loss and channels >= 2:
            self.spectral_loss = MultiScaleSpectralLoss(
                sample_rate=sample_rate,
                scales=[512, 1024, 2048, 4096],  # STFT scales
                window_type=stft_window_type
            )
        else:
            self.spectral_loss = None
        
        # Add STFT processing to the audio callback chain
        self.add_callback(self._stft_callback)
        
        # Storage for latest STFT results (kept for API compatibility)
        self.latest_stft = {
            'channel_1': {'freqs': None, 'magnitudes': None, 'phases': None, 'timestamp': None},
            'channel_2': {'freqs': None, 'magnitudes': None, 'phases': None, 'timestamp': None}
        }
        
        # Add callback to store latest results (if needed for other purposes)
        self.stft_processor.add_stft_callback(self._store_latest_stft)
        
        # Storage for separated channel data
        self.latest_channel_data = {
            'channel_1': None,
            'channel_2': None
        }
        
    def _stft_callback(self, audio_data: np.ndarray):
        """Internal callback to process audio through STFT and spectral loss."""
        
        # Debug: Check input data type and range
        if not hasattr(self, '_format_logged'):
            self._format_logged = True
            print(f"\nAudio format debug:")
            print(f"  dtype: {audio_data.dtype}")
            print(f"  shape: {audio_data.shape}")
            print(f"  min: {np.min(audio_data)}")
            print(f"  max: {np.max(audio_data)}")
            print(f"  mean: {np.mean(audio_data)}")
            
            # Check if data looks like integers
            if audio_data.dtype in [np.int16, np.int32, np.int64]:
                print("  ⚠️  Audio is in INTEGER format!")
            elif np.max(np.abs(audio_data)) > 10:
                print("  ⚠️  Audio values exceed normal float range (-1 to +1)!")
        
        # Convert to float32 and normalize if needed
        if audio_data.dtype in [np.int16, np.int32, np.int64]:
            # Integer audio - need to normalize
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                # Assume 24-bit or similar
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_data = audio_data.astype(np.float32) / max_val
        else:
            # Already float, but check range
            audio_data = audio_data.astype(np.float32)
            
            # If values are way out of normal audio range, there might be a format issue
            max_val = np.max(np.abs(audio_data))
            if max_val > 10.0:
                print(f"⚠️  WARNING: Audio values up to {max_val:.1f} - possible format mismatch!")
        
        # Separate channels
        if self.channels == 1:
            channel_1_data = audio_data
            channel_2_data = audio_data  # Duplicate for processing
        else:
            if len(audio_data.shape) == 1:
                # Interleaved stereo data
                channel_1_data = audio_data[0::2]
                channel_2_data = audio_data[1::2]
            else:
                # Already separated channels
                channel_1_data = audio_data[:, 0] if audio_data.shape[1] > 0 else audio_data.flatten()
                channel_2_data = audio_data[:, 1] if audio_data.shape[1] > 1 else channel_1_data
        
        # Store latest channel data
        self.latest_channel_data['channel_1'] = channel_1_data
        self.latest_channel_data['channel_2'] = channel_2_data
        
        # Process through STFT and spectral loss
        self.stft_processor.process_channel_data('channel_1', channel_1_data)
        if self.channels >= 2:
            self.stft_processor.process_channel_data('channel_2', channel_2_data)
        
        if self.spectral_loss and self.channels >= 2:
            self.spectral_loss.process_audio_channels(channel_1_data, channel_2_data)
            
    def _store_latest_stft(self, channel, freqs, magnitudes, phases, stft_complex):
        """Store the latest STFT results (for API compatibility)."""
        self.latest_stft[channel] = {
            'freqs': freqs.copy(),
            'magnitudes': magnitudes.copy(),
            'phases': None,  # Phase calculation disabled
            'stft_complex': stft_complex.copy(),
            'timestamp': time.time()
        }
        
    def add_spectral_loss_callback(self, callback: Callable):
        """Add a callback for spectral loss results."""
        if self.spectral_loss:
            self.spectral_loss.add_loss_callback(callback)
        else:
            print("Spectral loss not enabled or insufficient channels")
        
    def get_latest_stft(self, channel: str = 'channel_1'):
        """
        Get the latest STFT results for a channel.
        
        Args:
            channel (str): 'channel_1' or 'channel_2'
            
        Returns:
            dict: Latest STFT data or None if no data available
        """
        return self.latest_stft.get(channel)
    
    def get_frequency_bins(self):
        """Get the frequency bins for the STFT."""
        return self.stft_processor.freqs
        
    def get_current_spectral_loss(self):
        """Get the current spectral loss between channels."""
        if self.spectral_loss:
            return self.spectral_loss.get_current_losses()
        return None
        
    def get_spectral_loss_history(self, n_recent=10):
        """Get recent spectral loss history."""
        if self.spectral_loss:
            return self.spectral_loss.get_loss_history(n_recent)
        return []
        
    def get_average_spectral_loss(self, n_recent=10):
        """Get average spectral loss over recent computations."""
        if self.spectral_loss:
            return self.spectral_loss.get_average_losses(n_recent)
        return None


# Spectral loss callback functions
def print_spectral_loss(loss_data):
    """Print spectral loss metrics with scale info."""
    if not loss_data or 'total_loss' not in loss_data:
        return
        
    total_loss = loss_data['total_loss']
    timestamp = loss_data.get('timestamp', 0)
    
    # Scale info
    scale_info = ""
    if 'by_scale' in loss_data:
        scale_count = len(loss_data['by_scale'])
        scale_info = f" ({scale_count} scales)"
    
    # Only print timestamp in debug mode
    if hasattr(print_spectral_loss, 'debug_mode') and print_spectral_loss.debug_mode:
        print(f"Loss: {total_loss:.4f}{scale_info} @{timestamp:.3f}")
    else:
        # Just print the loss value inline (will be overwritten by monitoring)
        pass

def debug_spectral_loss(loss_data):
    """Debug function to show detailed loss computation."""
    # Only run if debug mode is enabled
    if not hasattr(debug_spectral_loss, 'debug_mode') or not debug_spectral_loss.debug_mode:
        return
        
    if not loss_data or 'by_scale' not in loss_data:
        print("DEBUG: No scale data available")
        return
        
    print(f"\nDEBUG: Loss Breakdown:")
    total = 0
    for scale_name, scale_loss in loss_data['by_scale'].items():
        print(f"  {scale_name}: {scale_loss:.4f}")
        total += scale_loss
    print(f"  TOTAL: {total:.4f}")
    print(f"  STORED: {loss_data['total_loss']:.4f}")


def enhanced_monitoring(loss_data):
    """Simplified monitoring showing only spectral loss."""
    # Display loss only - always show if available
    if loss_data and 'total_loss' in loss_data:
        print(f"\rSpectral Loss: {loss_data['total_loss']:.4f}          ", end='')
    else:
        print(f"\rSpectral Loss: N/A          ", end='')


def print_usage():
    """Print command line usage information."""
    print("Usage: python stft_audio.py [OPTIONS]")
    print("\nOptions:")
    print("  --list                    List available audio devices and exit")
    print("  --input-device INDEX      Use specific input device (see --list for indices)")
    print("  --output-device INDEX     Use specific output device (see --list for indices)")
    print("  --no-monitoring           Disable audio monitoring")
    print("  --no-spectral-loss        Disable multi-scale spectral loss computation")
    print("  --no-loss-display         Disable real-time loss display (computation still runs)")
    print("  --channels N              Number of channels (1=mono, 2=stereo, default=2)")
    print("  --window-size N           STFT window size in samples (default=2048)")
    print("  --hop-length N            STFT hop length in samples (default=512)")
    print("  --sample-rate N           Sample rate in Hz (default=44100)")
    print("  --debug                   Enable debug output (detailed loss breakdown)")
    print("  --help                    Show this help message")
    print("\nExamples:")
    print("  python stft_audio.py --list")
    print("  python stft_audio.py --input-device 2 --output-device 3")
    print("  python stft_audio.py --input-device 1 --no-monitoring --no-spectral-loss")
    print("  python stft_audio.py --input-device 0 --channels 1 --window-size 4096")
    print("  python stft_audio.py --debug  # Show detailed loss breakdown")
    print("  python stft_audio.py --no-loss-display  # Compute loss but don't display")
    print("\nPerformance Monitoring:")
    print("  [cpu] - Show current CPU usage")
    print("  [monitor] - Start/stop CPU monitoring")
    print("  [test] - Run performance test")
    print("  [profile] - Start/stop code profiling")
    print("\nSpectral Loss:")
    print("  Multi-scale spectral loss compares amplitude spectra of channel 1 and 2")
    print("  using approach with multiple STFT scales. Requires stereo input.")
    print("  Combines normalized Frobenius norm and log L1 norm at scales: 512, 1024, 2048, 4096.")


if __name__ == "__main__":
    import sys
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced Audio System with Spectral Loss', add_help=False)
    parser.add_argument('--list', action='store_true', help='List available audio devices')
    parser.add_argument('--input-device', type=int, help='Input device index')
    parser.add_argument('--output-device', type=int, help='Output device index')
    parser.add_argument('--no-monitoring', action='store_true', help='Disable audio monitoring')
    parser.add_argument('--no-spectral-loss', action='store_true', help='Disable multi-scale spectral loss')
    parser.add_argument('--no-loss-display', action='store_true', help='Disable real-time loss display')
    parser.add_argument('--channels', type=int, default=2, help='Number of audio channels (default: 2)')
    parser.add_argument('--window-size', type=int, default=2048, help='STFT window size (default: 2048)')
    parser.add_argument('--hop-length', type=int, default=512, help='STFT hop length (default: 512)')
    parser.add_argument('--sample-rate', type=int, default=44100, help='Sample rate in Hz (default: 44100)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--help', action='store_true', help='Show help message')
    
    try:
        args = parser.parse_args()
    except:
        print_usage()
        sys.exit(1)
    
    # Show help
    if args.help:
        print_usage()
        sys.exit(0)
    
    # List devices if requested
    if args.list:
        SimpleAudio.list_available_devices()
        sys.exit(0)
        
    # Validate device indices if provided
    if args.input_device is not None or args.output_device is not None:
        # Get device count to validate indices
        import pyaudio
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        p.terminate()
        
        if args.input_device is not None:
            if args.input_device < 0 or args.input_device >= device_count:
                print(f"Error: Input device index {args.input_device} is invalid.")
                print(f"Valid range: 0 to {device_count - 1}")
                print("Use --list to see available devices.")
                sys.exit(1)
                
        if args.output_device is not None:
            if args.output_device < 0 or args.output_device >= device_count:
                print(f"Error: Output device index {args.output_device} is invalid.")
                print(f"Valid range: 0 to {device_count - 1}")
                print("Use --list to see available devices.")
                sys.exit(1)
    
    # Warn about spectral loss requirements
    enable_spectral_loss = not args.no_spectral_loss
    if enable_spectral_loss and args.channels < 2:
        print("Warning: Spectral loss requires stereo input (--channels 2). Disabling spectral loss.")
        enable_spectral_loss = False
    
    # Create enhanced audio system with STFT
    print("Creating Enhanced Audio System with Spectral Loss...")
    print(f"Configuration: {args.channels} channels, {args.sample_rate}Hz, window={args.window_size}, hop={args.hop_length}")
    if enable_spectral_loss:
        print("Multi-scale spectral loss: ENABLED")
    else:
        print("Multi-scale spectral loss: DISABLED")
    if args.debug:
        print("Debug output: ENABLED")
    
    # Check loss display setting
    enable_loss_display = not args.no_loss_display
    if enable_spectral_loss and not enable_loss_display:
        print("Real-time loss display: DISABLED (loss computation still active)")
    
    # Determine monitoring setting
    enable_monitoring = not args.no_monitoring
    
    try:
        audio = EnhancedAudio(
            channels=args.channels,
            sample_rate=args.sample_rate,
            input_device=args.input_device,
            output_device=args.output_device,
            stft_window_size=args.window_size,
            stft_hop_length=args.hop_length,
            stft_window_type='hann',
            enable_spectral_loss=enable_spectral_loss
        )
        
        # Set monitoring if not disabled
        if enable_monitoring:
            audio.is_monitoring = True
            
    except Exception as e:
        print(f"Error creating audio system: {e}")
        print("Use --list to see available devices, or --help for usage information.")
        sys.exit(1)
    
    # CREATE PERFORMANCE MONITOR
    perf_monitor = AudioPerformanceMonitor()
    
    # Add spectral loss callbacks if enabled
    if enable_spectral_loss:
        print("Adding spectral loss callbacks...")
        
        # Set debug mode on callback functions
        print_spectral_loss.debug_mode = args.debug
        debug_spectral_loss.debug_mode = args.debug
        
        # Conditionally add the main monitoring callback
        if enable_loss_display:
            audio.add_spectral_loss_callback(enhanced_monitoring)
            # Set up monitoring reference
            enhanced_monitoring.audio_ref = audio
            print("Real-time loss display: ENABLED")
        else:
            print("Real-time loss display: DISABLED")
        
        # Only add debug callbacks if debug mode is enabled
        if args.debug:
            print("Debug mode: ENABLED - detailed loss breakdown will be shown")
            audio.add_spectral_loss_callback(print_spectral_loss)
            audio.add_spectral_loss_callback(debug_spectral_loss)
        else:
            print("Debug mode: DISABLED - use --debug to see detailed loss breakdown")
    
    # Print device information
    try:
        if args.input_device is not None:
            input_info = audio.p.get_device_info_by_index(args.input_device)
            print(f"Using input device {args.input_device}: {input_info['name']}")
        else:
            print("Using default input device")
            
        if enable_monitoring:
            if args.output_device is not None:
                output_info = audio.p.get_device_info_by_index(args.output_device)
                print(f"Using output device {args.output_device}: {output_info['name']}")
            else:
                print("Using default output device")
        else:
            print("Audio monitoring disabled")
    except:
        pass
    
    # Start the audio system
    print("\nStarting optimized audio system with spectral loss analysis...")
    try:
        audio.start()
    except Exception as e:
        print(f"Error starting audio system: {e}")
        print("This might be due to:")
        print("- Selected device not available or in use")
        print("- Insufficient permissions")
        print("- Hardware compatibility issues")
        sys.exit(1)
    
    print("\nReal-time spectral loss analysis is now running!")
    if enable_spectral_loss:
        print("Multi-scale spectral loss analysis active - comparing channels 1 & 2")
        print("Loss values represent spectral differences between channels:")
        print("- Higher values = more difference")
        print("- Lower values = more similarity")
        if not args.debug:
            print("\nTip: Use --debug flag to see detailed loss breakdown by scale")
    
    if enable_loss_display and enable_spectral_loss:
        print("\nYou should see spectral loss values above.")
    
    print("\nCommands:")
    print("  [r] - Start recording")
    print("  [s] - Stop recording") 
    print("  [w] - Write recording to file")
    print("  [m] - Toggle monitoring")
    if enable_spectral_loss:
        print("  [l] - Show recent loss history")
        print("  [a] - Show average loss")
        print("  [d] - Toggle debug output")
        if enable_loss_display:
            print("  [t] - Toggle loss display")
    # PERFORMANCE MONITORING COMMANDS
    print("  [cpu] - Show current CPU usage")
    print("  [monitor] - Start/stop CPU monitoring")  
    print("  [test] - Run performance test")
    print("  [profile] - Start/stop profiling")
    print("  [q] - Quit")
    print("\nPress Enter after each command:")
    
    # Main control loop
    running = True
    input_prompt = "> "
    if args.debug and enable_spectral_loss:
        print("\nNote: Debug output will appear above this prompt.")
    
    while running:
        try:
            cmd = input(input_prompt).lower().strip()
            
            if cmd == 'r':
                audio.start_recording()
            elif cmd == 's':
                audio.stop_recording()
            elif cmd == 'w':
                filename = input("Enter filename (default: recording.wav): ").strip()
                if not filename:
                    filename = "recording.wav"
                audio.save_recording(filename)
            elif cmd == 'm':
                audio.toggle_monitoring()
            elif cmd == 'l' and enable_spectral_loss:
                history = audio.get_spectral_loss_history(5)
                if history:
                    print("\n--- Recent Spectral Loss History ---")
                    for i, loss_data in enumerate(history[-5:]):
                        loss = loss_data.get('total_loss', 'N/A')
                        print(f"Frame {i+1}: Loss = {loss:.4f}")
                        # Only show scale breakdown in debug mode
                        if args.debug and 'by_scale' in loss_data:
                            for scale, value in loss_data['by_scale'].items():
                                print(f"  {scale}: {value:.4f}")
                else:
                    print("No spectral loss history available")
            elif cmd == 'a' and enable_spectral_loss:
                avg_loss = audio.get_average_spectral_loss(10)
                if avg_loss and 'total_loss' in avg_loss:
                    print(f"\n--- Average Spectral Loss (last 10 frames) ---")
                    print(f"Average Loss: {avg_loss['total_loss']:.4f}")
                else:
                    print("No spectral loss data available")
            elif cmd == 'd' and enable_spectral_loss:
                # Toggle debug mode
                args.debug = not args.debug
                print_spectral_loss.debug_mode = args.debug
                debug_spectral_loss.debug_mode = args.debug
                
                if args.debug:
                    # Add debug callbacks if not already added
                    if debug_spectral_loss not in [cb.__func__ if hasattr(cb, '__func__') else cb 
                                                        for cb in audio.spectral_loss.loss_callbacks]:
                        audio.add_spectral_loss_callback(print_spectral_loss)
                        audio.add_spectral_loss_callback(debug_spectral_loss)
                    print("Debug mode: ENABLED - detailed loss breakdown will be shown")
                else:
                    print("Debug mode: DISABLED")
            elif cmd == 't' and enable_spectral_loss:
                # Toggle loss display
                if enhanced_monitoring in [cb.__func__ if hasattr(cb, '__func__') else cb 
                                          for cb in audio.spectral_loss.loss_callbacks]:
                    # Remove the callback
                    audio.spectral_loss.loss_callbacks = [
                        cb for cb in audio.spectral_loss.loss_callbacks 
                        if (cb.__func__ if hasattr(cb, '__func__') else cb) != enhanced_monitoring
                    ]
                    print("Real-time loss display: DISABLED")
                else:
                    # Add the callback back
                    audio.add_spectral_loss_callback(enhanced_monitoring)
                    enhanced_monitoring.audio_ref = audio
                    print("Real-time loss display: ENABLED")
            # PERFORMANCE MONITORING COMMANDS
            elif cmd == 'cpu':
                perf_monitor.print_current_stats()
            elif cmd == 'monitor':
                if perf_monitor.cpu_monitor.is_monitoring:
                    perf_monitor.stop_monitoring()
                else:
                    perf_monitor.start_monitoring()
            elif cmd == 'test':
                duration = input("Test duration in seconds (default 30): ").strip()
                duration = int(duration) if duration else 30
                perf_monitor.run_performance_test(duration)
            elif cmd == 'profile':
                if perf_monitor.profiler.is_profiling:
                    perf_monitor.stop_profiling()
                else:
                    perf_monitor.start_profiling()
            elif cmd == 'q':
                running = False
            elif cmd == '':
                pass  # Ignore empty commands
            else:
                if cmd in ['l', 'a', 'd'] and not enable_spectral_loss:
                    print("Spectral loss not enabled. Use without --no-spectral-loss flag.")
                elif cmd == 't' and (not enable_spectral_loss or args.no_loss_display):
                    if not enable_spectral_loss:
                        print("Spectral loss not enabled. Use without --no-spectral-loss flag.")
                    else:
                        print("Loss display toggle not available with --no-loss-display flag.")
                else:
                    valid_cmds = "r, s, w, m, cpu, monitor, test, profile, q"
                    if enable_spectral_loss:
                        valid_cmds = "r, s, w, m, l, a, d, cpu, monitor, test, profile, q"
                        if enable_loss_display:
                            valid_cmds = "r, s, w, m, l, a, d, t, cpu, monitor, test, profile, q"
                    print(f"Unknown command. Use {valid_cmds}")
                
        except KeyboardInterrupt:
            running = False
        except Exception as e:
            print(f"Error: {e}")
    
    # Clean up
    print("\nShutting down...")
    perf_monitor.stop_monitoring()
    audio.stop()
    print("Enhanced audio system with spectral loss shutdown complete")