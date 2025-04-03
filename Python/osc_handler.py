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
        self.num_bands = 40  # Based on your SC code: ~numMelBands = 40
        
        # Callback storage
        self.callbacks = []
    
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
        
        # Send initial trigger to start SuperCollider analysis
        logger.info("Sending initial trigger to SuperCollider")
        self.client.send_message("/osc_from_python", True)
    
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