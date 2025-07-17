import serial
import time
import argparse
import threading

class DualESP32StepperController:
    def __init__(self, port1, port2, baudrate=115200, debug=False, motor_speed=2, motor_steps=100):
        """
        Initialize controller for two ESP32s
        
        Args:
            port1 (str): Serial port for first ESP32 (controlling odd motors)
            port2 (str): Serial port for second ESP32 (controlling even motors)
            baudrate (int): Baud rate for serial communication
            debug (bool): Enable debug output
        """
        self.port1 = port1  # Odd motors (1, 3, 5, 7)
        self.port2 = port2  # Even motors (2, 4, 6, 8)
        self.baudrate = baudrate
        self.debug = debug
        self.motor_speed = motor_speed
        self.motor_steps = motor_steps
        
        self.ser1 = None
        self.ser2 = None
        
        self.connected1 = False
        self.connected2 = False
        
        # Map motor numbers to ESP32 and internal motor index
        # Motor number (1-8) -> (ESP32 index, internal motor number)
        self.motor_map = {
            1: (0, 1),  # ESP32 #1, Motor 1
            2: (1, 1),  # ESP32 #2, Motor 1
            3: (0, 2),  # ESP32 #1, Motor 2
            4: (1, 2),  # ESP32 #2, Motor 2
            5: (0, 3),  # ESP32 #1, Motor 3
            6: (1, 3),  # ESP32 #2, Motor 3
            7: (0, 4),  # ESP32 #1, Motor 4
            8: (1, 4)   # ESP32 #2, Motor 4
        }
        
        # Response queues for each ESP32
        self.responses1 = []
        self.responses2 = []
        
        # Listener threads
        self.listeners = []
        self.listening = False
    
    def connect(self):
        """Connect to both ESP32s"""
        result = True
        
        # Connect to first ESP32 (odd motors)
        try:
            self.ser1 = serial.Serial(
                port=self.port1,
                baudrate=self.baudrate,
                timeout=1,
                write_timeout=1
            )
            time.sleep(2)  # Wait for ESP32 to initialize
            self.connected1 = True
            print(f"Connected to ESP32 #1 (odd motors) on {self.port1}")
        except Exception as e:
            print(f"Error connecting to ESP32 #1: {e}")
            self.connected1 = False
            result = False
        
        # Connect to second ESP32 (even motors)
        try:
            self.ser2 = serial.Serial(
                port=self.port2,
                baudrate=self.baudrate,
                timeout=1,
                write_timeout=1
            )
            time.sleep(2)  # Wait for ESP32 to initialize
            self.connected2 = True
            print(f"Connected to ESP32 #2 (even motors) on {self.port2}")
        except Exception as e:
            print(f"Error connecting to ESP32 #2: {e}")
            self.connected2 = False
            result = False
        
        # Start listener threads
        if self.connected1 or self.connected2:
            self.start_listeners()
        
        return result
    
    def disconnect(self):
        """Disconnect from both ESP32s"""
        # Stop listeners
        self.stop_listeners()
        
        # Disconnect from first ESP32
        if self.ser1 and self.ser1.is_open:
            self.ser1.close()
            print(f"Disconnected from ESP32 #1 on {self.port1}")
            self.connected1 = False
        
        # Disconnect from second ESP32
        if self.ser2 and self.ser2.is_open:
            self.ser2.close()
            print(f"Disconnected from ESP32 #2 on {self.port2}")
            self.connected2 = False
    
    def start_listeners(self):
        """Start listener threads for both ESP32s"""
        self.listening = True
        
        # Start listener for first ESP32 if connected
        if self.connected1:
            listener1 = threading.Thread(
                target=self._serial_listener,
                args=(self.ser1, 1, self.responses1),
                daemon=True
            )
            listener1.start()
            self.listeners.append(listener1)
        
        # Start listener for second ESP32 if connected
        if self.connected2:
            listener2 = threading.Thread(
                target=self._serial_listener,
                args=(self.ser2, 2, self.responses2),
                daemon=True
            )
            listener2.start()
            self.listeners.append(listener2)
    
    def stop_listeners(self):
        """Stop all listener threads"""
        self.listening = False
        
        # Wait for all listeners to finish
        for thread in self.listeners:
            thread.join(timeout=1.0)
        
        self.listeners = []
    
    def _serial_listener(self, ser, esp_num, response_queue):
        """Listener thread for serial responses"""
        current_line = ""
        
        while self.listening and ser and ser.is_open:
            try:
                if ser.in_waiting:
                    byte = ser.read(1)
                    if byte:
                        try:
                            char = byte.decode('ascii')
                            if char == '\n':
                                if current_line:
                                    message = current_line.strip()
                                    response_queue.append(message)
                                    if self.debug:
                                        print(f"ESP32 #{esp_num}: {message}")
                                current_line = ""
                            else:
                                current_line += char
                        except UnicodeDecodeError:
                            # Handle non-ASCII data
                            current_line += '?'
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"Error in listener for ESP32 #{esp_num}: {e}")
                time.sleep(0.1)
    
    def get_responses(self, esp_num=None, clear=True):
        """Get responses from the ESP32s"""
        if esp_num == 1:
            responses = self.responses1.copy()
            if clear:
                self.responses1.clear()
            return responses
        elif esp_num == 2:
            responses = self.responses2.copy()
            if clear:
                self.responses2.clear()
            return responses
        else:
            # Get responses from both ESP32s
            responses = {
                1: self.responses1.copy(),
                2: self.responses2.copy()
            }
            if clear:
                self.responses1.clear()
                self.responses2.clear()
            return responses
    
    def send_command(self, motor_number, command_type, value):
        """
        Send a command to a specific motor
        
        Args:
            motor_number (int): Motor number (1-8, or 0 for all motors)
            command_type (str): Command type (SPEED, DIR, STEPS, STOP)
            value (int): Command value
            
        Returns:
            bool: True if command was sent successfully
        """
        if motor_number == 0:
            # Command all motors
            success1 = self._send_to_esp32(1, 0, command_type, value)
            success2 = self._send_to_esp32(2, 0, command_type, value)
            return success1 and success2
        
        if motor_number < 1 or motor_number > 8:
            print(f"Invalid motor number: {motor_number}. Must be 1-8 or 0 for all")
            return False
        
        # Get ESP32 index and internal motor number
        esp_idx, internal_motor = self.motor_map[motor_number]
        
        # Send to appropriate ESP32
        return self._send_to_esp32(esp_idx + 1, internal_motor, command_type, value)
    
    def _send_to_esp32(self, esp_num, internal_motor, command_type, value):
        """Send command to specific ESP32"""
        ser = self.ser1 if esp_num == 1 else self.ser2
        connected = self.connected1 if esp_num == 1 else self.connected2
        
        if not connected:
            print(f"ESP32 #{esp_num} not connected")
            return False
        
        try:
            # Format command
            command = f"MOTOR:{internal_motor} {command_type}:{value}\n"
            
            if self.debug:
                print(f"Sending to ESP32 #{esp_num}: {command.strip()}")
            
            # Send command
            ser.write(command.encode('ascii'))
            return True
            
        except Exception as e:
            print(f"Error sending to ESP32 #{esp_num}: {e}")
            return False
    
    def set_speed(self, motor_number, speed):
        """Set motor speed"""
        return self.send_command(motor_number, "SPEED", speed)
    
    def set_direction(self, motor_number, direction):
        """Set motor direction (1=CW, 0=CCW)"""
        return self.send_command(motor_number, "DIR", direction)
    
    def move_steps(self, motor_number, steps):
        """Move motor by specified number of steps"""
        return self.send_command(motor_number, "STEPS", steps)
    
    def stop_motor(self, motor_number):
        """Stop the motor"""
        return self.send_command(motor_number, "STOP", 1)


def interactive_mode(controller):
    """Interactive mode for controlling motors"""
    print("\n=== Interactive Dual ESP32 Stepper Motor Control ===")
    print("Commands:")
    print("  speed <motor> <value>   - Set motor speed")
    print("  dir <motor> <value>     - Set motor direction (1=CW, 0=CCW)")
    print("  steps <motor> <value>   - Move motor by steps")
    print("  stop <motor>            - Stop motor")
    print("  all speed <value>       - Set all motors' speed")
    print("  all dir <value>         - Set all motors' direction")
    print("  all steps <value>       - Move all motors by steps")
    print("  all stop                - Stop all motors")
    print("  resp                    - Show recent responses")
    print("  exit                    - Exit interactive mode")
    print("Motor can be 1-8, or 'all' for all motors")
    print("Note: ESP32 #1 controls odd motors (1,3,5,7)")
    print("      ESP32 #2 controls even motors (2,4,6,8)")
    
    while True:
        try:
            command = input("\nCommand: ").strip().lower()
            
            if command == "exit":
                break
                
            if command == "resp":
                responses = controller.get_responses()
                print("\nResponses from ESP32 #1 (odd motors):")
                for resp in responses[1]:
                    print(f"  {resp}")
                print("\nResponses from ESP32 #2 (even motors):")
                for resp in responses[2]:
                    print(f"  {resp}")
                continue
                
            parts = command.split()
            
            if len(parts) < 2:
                print("Invalid command format")
                continue
                
            # Check for 'all' commands
            if parts[0] == "all" and len(parts) >= 3:
                cmd_type = parts[1]
                value = int(parts[2])
                
                if cmd_type == "speed":
                    controller.set_speed(0, value)
                elif cmd_type == "dir":
                    controller.set_direction(0, value)
                elif cmd_type == "steps":
                    controller.move_steps(0, value)
                elif cmd_type == "stop":
                    controller.stop_motor(0)
                else:
                    print(f"Unknown command type: {cmd_type}")
                continue
                
            # Regular commands for individual motors
            cmd_type = parts[0]
            
            try:
                if cmd_type not in ["speed", "dir", "steps", "stop"]:
                    print(f"Unknown command type: {cmd_type}")
                    continue
                    
                motor_num = int(parts[1])
                if motor_num < 1 or motor_num > 8:
                    print("Motor number must be 1-8")
                    continue
                
                if cmd_type == "stop":
                    # Stop the specified motor
                    controller.stop_motor(motor_num)
                else:
                    if len(parts) < 3:
                        print(f"{cmd_type} command requires a value")
                        continue
                        
                    value = int(parts[2])
                    
                    if cmd_type == "speed":
                        controller.set_speed(motor_num, value)
                    elif cmd_type == "dir":
                        controller.set_direction(motor_num, value)
                    elif cmd_type == "steps":
                        controller.move_steps(motor_num, value)
                
            except ValueError:
                print("Invalid number format")
                        
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_test_sequence(controller):
    """Run a test sequence for all motors"""
    print("\n=== Running 8-Motor Test Sequence ===")
    
    # Set speeds for all motors
    print("Setting speeds...")
    for i in range(1, 9):
        controller.set_speed(i, controller.motor_speed + (i * 5))
        time.sleep(0.1)
    
    # Set directions (odd CW, even CCW)
    print("Setting directions...")
    for i in range(1, 9):
        direction = 1 if i % 2 == 1 else 0  # Odd motors CW, even motors CCW
        controller.set_direction(i, direction)
        time.sleep(0.1)
    
    # Move all motors with staggered start
    print("Moving motors in sequence...")
    for i in range(1, 9):
        controller.move_steps(i, controller.motor_steps)
        print(f"Started motor {i}")
        time.sleep(0.5)
    
    # Wait for movements to complete
    print("Waiting for movements to complete...")
    time.sleep(10)
    
    # Get responses
    responses = controller.get_responses()
    print("\nResponses from ESP32 #1 (odd motors):")
    for resp in responses[1]:
        print(f"  {resp}")
    print("\nResponses from ESP32 #2 (even motors):")
    for resp in responses[2]:
        print(f"  {resp}")
    
    print("Test sequence complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Control 8 stepper motors via dual ESP32s')
    parser.add_argument('--port1', type=str, required=True, help='Serial port for ESP32 #1 (odd motors)')
    parser.add_argument('--port2', type=str, required=True, help='Serial port for ESP32 #2 (even motors)')
    parser.add_argument('--baudrate', type=int, default=115200, help='Baud rate')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--test', action='store_true', help='Run test sequence')
    
    args = parser.parse_args()
    
    # Create the controller
    controller = DualESP32StepperController(args.port1, args.port2, args.baudrate, args.debug, args.motor_speed, args.motor_steps)
    
    try:
        # Connect to both ESP32s
        if controller.connect():
            if args.test:
                # Run test sequence
                run_test_sequence(controller)
            else:
                # Interactive mode
                interactive_mode(controller)
        
    except KeyboardInterrupt:
        print("\nExiting...")
    
    finally:
        # Always disconnect
        controller.disconnect()