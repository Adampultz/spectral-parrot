
import argparse
import time
import signal
import sys
from Stepper_Control import DualESP32StepperController

# Global variable to hold controller reference for signal handler
motor_controller = None
motor_speed = 200
motor_steps = 500
completion_time = 20

def emergency_shutdown(signum=None, frame=None):
    """Emergency shutdown procedure for motors"""
    global motor_controller
    
    print("\n\nEMERGENCY SHUTDOWN SEQUENCE")
    if motor_controller:
        try:
            print("Stopping all motors...")
            motor_controller.stop_motor(0)
            time.sleep(0.5)
            motor_controller.disconnect()
            
        except Exception as e:
            print(f"Error during emergency shutdown: {e}")
    
    if signum is not None:
        sys.exit(0)

def listen_for_completion(controller, duration=completion_time):
    """
    Just listen for any motor completion messages for a specific duration.
    
    Args:
        controller: The motor controller instance
        duration: How long to listen for messages in seconds
    """
    print(f"\nListening for ANY motor completion messages for {duration} seconds...")
    
    # Clear existing messages
    controller.get_responses(clear=True)
    
    start_time = time.time()
    completion_messages = []
    
    while time.time() - start_time < duration:
        responses = controller.get_responses()
        
        for esp_num in [1, 2]:
            for response in responses[esp_num]:
                print(f"ESP32 #{esp_num}: {response}")
                
                if "movement completed" in response.lower():
                    msg = f"Motor completion message from ESP32 #{esp_num}: {response}"
                    completion_messages.append(msg)
                    print(f"✅ {msg}")
        
        time.sleep(0.1)
    
    print("\nSummary:")
    if completion_messages:
        print(f"Detected {len(completion_messages)} completion messages:")
        for msg in completion_messages:
            print(f"  - {msg}")
    else:
        print("❌ No completion messages detected during the listening period")

def test_completion_for_motor(controller, motor_num, steps=motor_steps):
    """
    Test completion messages for a specific motor.
    
    Args:
        controller: The motor controller instance
        motor_num: Which motor to test (1-8)
        steps: How many steps to move
    """
    print(f"\nTesting completion messages for motor {motor_num}...")
    
    # Determine which ESP32 controls this motor and what internal motor number it is
    esp_num = 1 if motor_num % 2 == 1 else 2
    # Calculate internal motor number
    if esp_num == 1:  # Odd motors: 1->1, 3->2, 5->3, 7->4
        internal_motor = ((motor_num - 1) // 2) + 1
    else:  # Even motors: 2->1, 4->2, 6->3, 8->4
        internal_motor = (motor_num // 2)
    
    print(f"Motor {motor_num} is controlled by ESP32 #{esp_num} as internal motor {internal_motor}")
    
    # Clear existing messages
    controller.get_responses(clear=True)
    
    # Set moderate speed
    controller.set_speed(motor_num, motor_speed)  # Or use a faster speed like 50 for quicker testing
    time.sleep(0.1)
    
    # Move the motor
    print(f"Moving motor {motor_num} clockwise by {steps} steps...")
    controller.set_direction(motor_num, 1)  # CW
    controller.move_steps(motor_num, steps)
    
    # Listen for completion messages
    print("Waiting for completion messages (10 second timeout)...")
    start_time = time.time()
    completion_detected = False
    
    while time.time() - start_time < 10.0:
        responses = controller.get_responses()
        
        for resp_esp_num in [1, 2]:
            for response in responses[resp_esp_num]:
                print(f"ESP32 #{resp_esp_num}: {response}")
                
                # Check for MOTOR_COMPLETE format
                if "MOTOR_COMPLETE:" in response:
                    try:
                        reported_internal_motor = int(response.split(":")[1].strip())
                        print(f"  -> Found MOTOR_COMPLETE for internal motor {reported_internal_motor}")
                        
                        # Check if this matches our expected internal motor
                        if resp_esp_num == esp_num and reported_internal_motor == internal_motor:
                            print(f"✅ Detected completion message for motor {motor_num}!")
                            completion_detected = True
                    except Exception as e:
                        print(f"Error parsing: {e}")
                
                # Also check verbose format
                elif "movement completed" in response.lower():
                    parts = response.split()
                    if len(parts) > 1 and parts[0].lower() == "motor":
                        try:
                            reported_internal_motor = int(parts[1])
                            print(f"  -> Found movement completed for internal motor {reported_internal_motor}")
                            
                            if resp_esp_num == esp_num and reported_internal_motor == internal_motor:
                                print(f"✅ Detected completion message for motor {motor_num}!")
                                completion_detected = True
                        except:
                            pass
        
        if completion_detected:
            break
            
        time.sleep(0.1)
    
    if not completion_detected:
        print(f"❌ No completion message detected for motor {motor_num} after 10 seconds")
        print(f"   Expected: Internal motor {internal_motor} from ESP32 #{esp_num}")
    
    # Stop the motor just to be safe
    controller.stop_motor(motor_num)
    time.sleep(0.5)
    
    return completion_detected

def interactive_motor_positioning(port1, port2, baudrate=115200):
    """
    Interactive utility to position motors and test completion messages.
    """
    global motor_controller
    
    print("\n=== Motor Positioning and Completion Message Testing ===")
    
    # Register signal handlers for emergency shutdown
    signal.signal(signal.SIGINT, emergency_shutdown)
    signal.signal(signal.SIGTERM, emergency_shutdown)
    
    # Create and connect to the motor controller
    motor_controller = DualESP32StepperController(port1, port2, baudrate, debug=True)
    if not motor_controller.connect():
        print("Failed to connect to one or both ESP32s.")
        return

    # Reset ESP32s to see initialization messages
    print("Connected, resetting ESP32s to see initialization...")

    # Reset both ESP32s by toggling DTR
    if motor_controller.ser1:
        motor_controller.ser1.dtr = False
        time.sleep(0.1)
        motor_controller.ser1.dtr = True
        
    if motor_controller.ser2:
        motor_controller.ser2.dtr = False
        time.sleep(0.1)
        motor_controller.ser2.dtr = True

    # Wait for initialization messages
    print("Waiting for initialization...")
    time.sleep(3)

    # Get and display initialization messages
    responses = motor_controller.get_responses()
    print("\n=== ESP32 #1 Initialization ===")
    for resp in responses[1]:
        print(resp)
    print("\n=== ESP32 #2 Initialization ===")
    for resp in responses[2]:
        print(resp)
    print("\n=== End of Initialization ===\n")
    
    # Set initial speeds for all motors
    print("Setting initial speeds...")
    for motor in range(1, 9):
        motor_controller.set_speed(motor, motor_speed)
        time.sleep(0.1)
    
    print("\nCommands:")
    print("  <motor> cw <steps>   - Move motor clockwise")
    print("  <motor> ccw <steps>  - Move motor counter-clockwise")
    print("  <motor> speed <val>  - Set motor speed")
    print("  <motor> stop         - Stop motor")
    print("  <motor> test         - Test completion messages")
    print("  test all             - Test completion messages for all motors")
    print("  listen <seconds>     - Just listen for any completion messages")
    print("  all stop             - Stop all motors")
    print("  quit                 - Exit the utility with safe shutdown")
    
    try:
        while True:
            cmd = input("\nCommand: ").strip().lower()
            
            if cmd in ["exit", "quit"]:
                print("Performing safe shutdown sequence...")
                emergency_shutdown()
                break
                
            if cmd == "all stop":
                motor_controller.stop_motor(0)
                print("Stopped all motors")
                continue
                
            if cmd.startswith("listen"):
                parts = cmd.split()
                duration = completion_time  # default
                if len(parts) > 1:
                    try:
                        duration = float(parts[1])
                    except:
                        pass
                listen_for_completion(motor_controller, duration)
                continue
                
            if cmd == "test all":
                results = []
                for motor in range(1, 9):
                    result = test_completion_for_motor(motor_controller, motor)
                    results.append(result)
                
                print("\nTest Results Summary:")
                for motor in range(1, 9):
                    status = "✅ PASS" if results[motor-1] else "❌ FAIL"
                    print(f"Motor {motor}: {status}")
                continue
            
            parts = cmd.split()
            if len(parts) < 2:
                print("Invalid command format")
                continue
            
            try:
                # Handle 'test' command differently
                if parts[1] == "test":
                    motor_num = int(parts[0])
                    if motor_num < 1 or motor_num > 8:
                        print("Motor number must be 1-8")
                        continue
                    
                    test_completion_for_motor(motor_controller, motor_num)
                    continue
                
                # Parse motor number
                motor = int(parts[0])
                if motor < 1 or motor > 8:
                    print("Motor number must be 1-8")
                    continue
                
                # Parse command
                action = parts[1]
                
                if action == "stop":
                    motor_controller.stop_motor(motor)
                    print(f"Stopped motor {motor}")
                    
                elif action == "cw":
                    if len(parts) < 3:
                        print("Missing steps value")
                        continue
                    steps = int(parts[2])
                    motor_controller.set_direction(motor, 1)  # 1 = CW
                    motor_controller.move_steps(motor, steps)
                    print(f"Moving motor {motor} clockwise by {steps} steps")
                    
                elif action == "ccw":
                    if len(parts) < 3:
                        print("Missing steps value")
                        continue
                    steps = int(parts[2])
                    motor_controller.set_direction(motor, 0)  # 0 = CCW
                    motor_controller.move_steps(motor, steps)
                    print(f"Moving motor {motor} counter-clockwise by {steps} steps")
                    
                elif action == "speed":
                    if len(parts) < 3:
                        print("Missing speed value")
                        continue
                    speed = int(parts[2])
                    if speed < 0:
                        print("Speed should be positive")
                        continue
                    motor_controller.set_speed(motor, speed)
                    print(f"Set motor {motor} speed to {speed}")
                    
                else:
                    print(f"Unknown action: {action}")
                    
            except ValueError:
                print("Invalid number format")
            except Exception as e:
                print(f"Error: {e}")
    
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Always ensure motors are stopped
        emergency_shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Position motors and test completion messages")
    parser.add_argument("--port1", type=str, required=True, help="Port for first ESP32 (odd motors)")
    parser.add_argument("--port2", type=str, required=True, help="Port for second ESP32 (even motors)")
    parser.add_argument("--baudrate", type=int, default=115200, help="Baud rate")
    
    args = parser.parse_args()
    interactive_motor_positioning(args.port1, args.port2, args.baudrate)