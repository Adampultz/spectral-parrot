from pythonosc import udp_client, dispatcher, osc_server
import threading
import logging
import signal
import sys
import time
import numpy as np
import queue

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class OSCHandler:
    def __init__(self):
        # OSC Client setup
        self.client = udp_client.SimpleUDPClient("127.0.0.1", 2222, allow_broadcast=True)
        
        # OSC Server setup
        self.disp = dispatcher.Dispatcher()
        self.disp.map("/test", self.message_handler)  # From SC, this is sent as /test
        
        # Server configuration
        self.server = osc_server.ThreadingOSCUDPServer(
            ("127.0.0.1", 3333),
            self.disp
        )
        
        # Threading setup
        self.server_thread = None
        self.running = True
        
        # Data storage for visualization
        self.data_queue = queue.Queue()
        # TODO: Make self.numBands and argument to the function
        self.num_bands = 40  # Based on your SC code: ~numMelBands = 40
        
        # Callback storage
        self.callbacks = []

        self.audio_loaded = False
        self.num_training_audios = 0
        self.current_audio_index = 0 
        self.audio_switch_confirmed = False
        self.audio_load_error = None
        
        # Add OSC message handlers
        self.disp.map("/training_audio_loaded", self._handle_audio_loaded)
        self.disp.map("/training_audio_switched", self._handle_audio_switched)
        self.disp.map("/num_training_audios", self._handle_num_audios)

    def load_training_audio_folder(self, folder_path: str, rotation_count: int = 0):
        """Send folder path to SuperCollider to load all audio files"""
        logger.info(f"Sending training audio folder to SuperCollider: {folder_path}")
        logger.info(f"Rotation count: {rotation_count}")
        self.audio_loaded = False
        self.audio_load_error = None
        self.client.send_message("/load_training_audio_path", [folder_path, rotation_count])
    
    def switch_training_audio(self, buffer_index: int):
        """Switch to a specific buffer index"""
        logger.info(f"Switching to training audio buffer #{buffer_index}")
        self.audio_switch_confirmed = False
        self.client.send_message("/switch_training_audio", buffer_index)
    
    def get_num_training_audios(self):
        """Query number of loaded buffers"""
        self.client.send_message("/get_num_training_audios", 1)
    
    def wait_for_audio_load(self, timeout: float = 10.0) -> bool:
        """Wait for audio files to finish loading"""
        start_time = time.time()
        while not self.audio_loaded and (time.time() - start_time) < timeout:
            if self.audio_load_error:
                logger.error(f"Audio loading failed: {self.audio_load_error}")
                return False
            time.sleep(0.1)
        return self.audio_loaded
    
    def wait_for_audio_switch(self, timeout: float = 2.0) -> bool:
        """Wait for confirmation that audio has been switched"""
        start_time = time.time()
        while not self.audio_switch_confirmed and (time.time() - start_time) < timeout:
            time.sleep(0.05)
        return self.audio_switch_confirmed
    
    # Handler methods
    def _handle_audio_loaded(self, address, num_buffers, start_index, status):
        """Handle confirmation of audio loading"""
        if status == "success":
            self.audio_loaded = True
            self.num_training_audios = num_buffers
            self.current_audio_index = start_index
            logger.info(f"✓ SuperCollider loaded {num_buffers} audio buffers")
            logger.info(f"✓ Starting at buffer index {start_index}")
        else:
            self.audio_loaded = False
            self.audio_load_error = status
            logger.error(f"✗ Audio loading failed: {status}")
    
    def _handle_audio_switched(self, address, index, success, message):
        """Handle confirmation of audio buffer switch"""
        if success == 1:
            self.audio_switch_confirmed = True
            logger.info(f"✓ Audio switch confirmed: buffer #{index}")
        else:
            self.audio_switch_confirmed = False
            logger.error(f"✗ Audio switch failed for buffer #{index}: {message}")
    
    def _handle_num_audios(self, address, count):
        """Handle response with number of loaded audio buffers"""
        self.num_training_audios = count
        logger.info(f"SuperCollider has {count} training audio buffers loaded")
    
    def register_callback(self, callback):
        """Register a callback function that will be called with new data"""
        self.callbacks.append(callback)
        
    def message_handler(self, address, *args):
        """Handle incoming OSC messages"""
        try:
            # The OSC message format is: [num_bands, target_data..., input_data...]
            values = np.array(args)
            
            # Split the array in half (target and input)
            middle = len(values) // 2
            target_data = values[:middle]
            input_data = values[middle:]
            
            # Normalize the data for visualization
            # This step may need adjustment based on your data ranges
            target_data_norm = np.clip(target_data, 0, 1)
            input_data_norm = np.clip(input_data, 0, 1)
            
            # Call all registered callbacks with the new data
            for callback in self.callbacks:
                callback(target_data_norm, input_data_norm)
                
        except Exception as e:
            logger.error(f"Error processing OSC message: {e}")
    
    def start(self):
        """Start the OSC server"""
        logger.info("Starting OSC server on 127.0.0.1:3333")
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
    
    def stop_sc_analysis(self):
        """Send a message to stop the SuperCollider analysis"""
        self.client.send_message("/osc_from_python", False)
        
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        self.running = False
        
        # Stop SuperCollider analysis
        self.stop_sc_analysis()
        
        if self.server:
            self.server.shutdown()
        if self.server_thread:
            self.server_thread.join(timeout=1.0)

def setup_signal_handlers(osc_handler):
    def signal_handler(signum, frame):
        """Handle cleanup on signal"""
        logger.info(f"Received signal {signum}, shutting down...")
        osc_handler.cleanup()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    try:
        # Create OSC handler
        osc_handler = OSCHandler()
        setup_signal_handlers(osc_handler)
        
        # Start OSC server
        osc_handler.start()
        logger.info("OSC Handler running. Press Ctrl+C to stop.")
        
        # Simple command loop
        while osc_handler.running:
            command = input("Enter command (start/stop/quit): ").strip().lower()
            if command == "start":
                logger.info("Sending start command to SuperCollider")
                osc_handler.client.send_message("/osc_from_python", True)
            elif command == "stop":
                logger.info("Sending stop command to SuperCollider")
                osc_handler.client.send_message("/osc_from_python", False)
            elif command == "quit":
                break
            else:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        if 'osc_handler' in locals():
            osc_handler.cleanup()
        logger.info("OSC Handler stopped")