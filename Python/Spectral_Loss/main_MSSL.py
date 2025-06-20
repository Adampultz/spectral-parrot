# simple_audio.py
import pyaudio
import wave
import numpy as np
import time
from typing import Callable, List, Optional


class SimpleAudio:
    """
    A simplified audio interface with callback functionality.
    Handles audio input and output with options to record when needed.
    """
    
    @staticmethod
    def list_available_devices():
        """
        List all available audio devices without creating an instance.
        
        Returns:
            List of device info dictionaries
        """
        p = pyaudio.PyAudio()
        devices = []
        
        print("\nAvailable Audio Devices:")
        print("-" * 80)
        print(f"{'Index':<6} | {'Name':<40} | {'Inputs':<8} | {'Outputs':<8} | {'Default'}")
        print("-" * 80)
        
        try:
            default_input = p.get_default_input_device_info()['index']
            default_output = p.get_default_output_device_info()['index']
        except:
            default_input = default_output = -1
        
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            devices.append(info)
            
            # Mark default devices
            default_marks = []
            if i == default_input:
                default_marks.append("input")
            if i == default_output:
                default_marks.append("output")
            default_str = ", ".join(default_marks) if default_marks else ""
            
            print(f"{i:<6} | {info['name'][:40]:<40} | {info['maxInputChannels']:<8} | "
                  f"{info['maxOutputChannels']:<8} | {default_str}")
            
        print("-" * 80)
        p.terminate()
        
        return devices
    
    def __init__(self, 
                 sample_rate=44100, 
                 channels=1, 
                 buffer_size=1024, 
                 input_device=None,
                 output_device=None):
        """
        Initialize the audio system.
        
        Args:
            sample_rate (int): Sample rate in Hz
            channels (int): Number of audio channels (1=mono, 2=stereo)
            buffer_size (int): Frames per buffer
            input_device (int, optional): Input device index
            output_device (int, optional): Output device index
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_size = buffer_size
        self.input_device = input_device
        self.output_device = output_device
        
        self.p = pyaudio.PyAudio()

        # Get device info and channel counts
        if input_device is not None:
            input_info = self.p.get_device_info_by_index(input_device)
            self.input_channels = int(input_info['maxInputChannels'])
            self.input_device = input_device
        else:
            default_input = self.p.get_default_input_device_info()
            self.input_channels = int(default_input['maxInputChannels'])
            self.input_device = default_input['index']

        if output_device is not None:
            output_info = self.p.get_device_info_by_index(output_device)
            self.output_channels = int(output_info['maxOutputChannels'])
            self.output_device = output_device
        else:
            default_output = self.p.get_default_output_device_info()
            self.output_channels = int(default_output['maxOutputChannels'])
            self.output_device = default_output['index']

        print(f"Input configuration: {self.input_channels} channels")
        print(f"Output configuration: {self.output_channels} channels")

        self.input_stream = None
        self.output_stream = None
        
        self.is_running = False
        self.is_recording = False
        self.is_monitoring = False
        
        self.frames = []  # Recorded frames
        self._callbacks = []  # User callbacks
        
    def list_devices(self):
        """List all available audio devices"""
        return SimpleAudio.list_available_devices()
    
    def add_callback(self, callback: Callable[[np.ndarray], None]):
        """
        Add a function to be called for each audio buffer.
        
        Args:
            callback: Function that takes a numpy array of audio data
        """
        self._callbacks.append(callback)
        
    def _process_callbacks(self, audio_data: np.ndarray):
        """Process audio data through all callbacks"""

        if len(audio_data.shape) == 1:
            audio_data = audio_data.reshape(-1, self.input_channels)
    
        for callback in self._callbacks:
            callback(audio_data)
            
    def select_devices(self):
        """Interactively select input and output devices"""
        self.list_devices()
        
        # Select input device
        while True:
            try:
                choice = input("\nSelect input device index (or Enter for default): ").strip()
                if choice == "":
                    self.input_device = None
                    break
                choice = int(choice)
                if 0 <= choice < self.p.get_device_count():
                    info = self.p.get_device_info_by_index(choice)
                    if info['maxInputChannels'] > 0:
                        self.input_device = choice
                        break
                    else:
                        print("Selected device has no input channels. Please select another.")
                else:
                    print("Invalid device index.")
            except ValueError:
                print("Please enter a number or press Enter.")
                
        # Select output device if monitoring
        monitor = input("\nEnable audio monitoring? (y/n): ").lower().strip() == 'y'
        self.is_monitoring = monitor
        
        if monitor:
            while True:
                try:
                    choice = input("Select output device index (or Enter for default): ").strip()
                    if choice == "":
                        self.output_device = None
                        break
                    choice = int(choice)
                    if 0 <= choice < self.p.get_device_count():
                        info = self.p.get_device_info_by_index(choice)
                        if info['maxOutputChannels'] > 0:
                            self.output_device = choice
                            break
                        else:
                            print("Selected device has no output channels. Please select another.")
                    else:
                        print("Invalid device index.")
                except ValueError:
                    print("Please enter a number or press Enter.")
    
    def start(self):
        """Start the audio system"""
        if self.is_running:
            print("Audio system already running.")
            return
                
        self.is_running = True

        # Define callback function first
        def input_callback(in_data, frame_count, time_info, status):
            # Convert audio data to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            
            # Process through user callbacks
            self._process_callbacks(audio_data)
            
            # Record if enabled
            if self.is_recording:
                self.frames.append(in_data)
                
            # Monitor if enabled
            if self.is_monitoring and self.output_stream:
                self.output_stream.write(in_data)
                
            return (in_data, pyaudio.paContinue)
    
        # Open input stream
        self.input_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.input_device,
            frames_per_buffer=self.buffer_size,
            stream_callback=input_callback
        )
        """Start the audio system"""
        if self.is_running:
            print("Audio system already running.")
            return
            
        self.is_running = True
        
        # Set up output stream if monitoring
        if self.is_monitoring:
            self.output_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.output_device,
                frames_per_buffer=self.buffer_size
            )
        
        # Callback function for audio input
        def input_callback(in_data, frame_count, time_info, status):
            # Convert audio data to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            
            # Process through user callbacks
            self._process_callbacks(audio_data)
            
            # Record if enabled
            if self.is_recording:
                self.frames.append(in_data)
                
            # Monitor if enabled
            if self.is_monitoring and self.output_stream:
                self.output_stream.write(in_data)
                
            return (in_data, pyaudio.paContinue)
        
        # Open input stream
        self.input_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.input_device,
            frames_per_buffer=self.buffer_size,
            stream_callback=input_callback
        )
        
        # Print info about the active devices
        try:
            if self.input_device is None:
                input_info = self.p.get_default_input_device_info()
            else:
                input_info = self.p.get_device_info_by_index(self.input_device)
            print(f"Using input device: {input_info['name']}")
            
            if self.is_monitoring:
                if self.output_device is None:
                    output_info = self.p.get_default_output_device_info()
                else:
                    output_info = self.p.get_device_info_by_index(self.output_device)
                print(f"Using output device: {output_info['name']}")
        except:
            pass
            
        print("Audio system started")
    
    def stop(self):
        """Stop the audio system"""
        if not self.is_running:
            print("Audio system not running.")
            return
            
        self.is_running = False
        self.is_recording = False
        
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None
            
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None
            
        print("Audio system stopped")
    
    def start_recording(self):
        """Start recording audio"""
        if not self.is_running:
            print("Audio system not running. Starting...")
            self.start()
        
        self.frames = []
        self.is_recording = True
        print("Recording started")
    
    def stop_recording(self):
        """Stop recording audio"""
        if not self.is_recording:
            print("Not recording.")
            return
            
        self.is_recording = False
        print("Recording stopped")
        
    def toggle_monitoring(self):
        """Toggle audio monitoring on/off"""
        # Can't toggle if not running
        if not self.is_running:
            print("Start the audio system first.")
            return
            
        # If monitoring is on, turn it off
        if self.is_monitoring and self.output_stream:
            self.is_monitoring = False
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None
            print("Monitoring disabled")
        # If monitoring is off, turn it on
        else:
            self.is_monitoring = True
            self.output_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.output_device,
                frames_per_buffer=self.buffer_size
            )
            print("Monitoring enabled")
    
    def save_recording(self, filename="recording.wav"):
        """Save the recorded audio to a file"""
        if not self.frames:
            print("No recorded audio to save.")
            return
            
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        
        print(f"Recording saved to {filename}")
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'is_running') and self.is_running:
            self.stop()
        if hasattr(self, 'p') and self.p:
            self.p.terminate()


# Example audio callback functions
def print_audio_level(audio_data):
    """Print a simple VU meter based on audio level"""
    level = np.abs(audio_data).mean()
    bars = int(level / 100)
    print(f"Level: {'|' * bars}{' ' * (30 - bars)} {level:.0f}", end='\r')

def detect_silence(audio_data, threshold=500):
    """Detect silence in audio"""
    level = np.abs(audio_data).mean()
    if level < threshold:
        print("SILENCE DETECTED                        ", end='\r')


if __name__ == "__main__":
    import sys
    
    # List devices if requested, without creating an instance
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        SimpleAudio.list_available_devices()
        sys.exit(0)
        
    # Create audio system
    audio = SimpleAudio()
    
    # Add callbacks
    audio.add_callback(print_audio_level)
    
    # Interactive setup
    audio.select_devices()
    
    # Start the audio system
    audio.start()
    
    # Main control loop
    print("\nCommands: [r]ecord, [s]top, [p]lay/pause monitoring, [w]rite to file, [q]uit")
    running = True
    while running:
        try:
            cmd = input("> ").lower().strip()
            
            if cmd == 'r':
                audio.start_recording()
            elif cmd == 's':
                audio.stop_recording()
            elif cmd == 'p':
                audio.toggle_monitoring()
            elif cmd == 'w':
                filename = input("Enter filename (default: recording.wav): ").strip()
                if not filename:
                    filename = "recording.wav"
                audio.save_recording(filename)
            elif cmd == 'q':
                running = False
            else:
                print("Unknown command")
                
        except KeyboardInterrupt:
            running = False
        except Exception as e:
            print(f"Error: {e}")
    
    # Clean up
    audio.stop()
    print("Audio system shutdown complete")