#!/usr/bin/env python3
import argparse
import time
import signal
import sys
from Stepper_Control import DualESP32StepperController

# Global variable to hold controller reference for signal handler
motor_controller = None

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

def listen_for_completion(controller, duration=10.0):
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

def test_completion_for_motor(controller, motor_num, steps=50):
    """
    Test completion messages for a specific motor.
    
    Args:
        controller: The motor controller instance
        motor_num: Which motor to test (1-8)
        steps: How many steps to move
    """
    print(f"\nTesting completion messages for motor {motor_num}...")
    
    # Clear existing messages
    controller.get_responses(clear=True)
    
    # Set moderate speed
    controller.set_speed(motor_num, 2)
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
        
        for esp_num in [1, 2]:
            for response in responses[esp_num]:
                print(f"ESP32 #{esp_num}: {response}")
                
                if "movement completed" in response.lower():
                    parts = response.split()
                    motor_in_msg = None
                    
                    try:
                        if len(parts) > 1 and parts[0] == "Motor":
                            motor_in_msg = int(parts[1])
                    except:
                        pass
                        
                    if motor_in_msg == motor_num:
                        print(f"✅ Detected completion message for motor {motor_num}!")
                        completion_detected = True
        
        if completion_detected:
            break
            
        time.sleep(0.1)
    
    if not completion_detected:
        print(f"❌ No completion message detected for motor {motor_num} after 10 seconds")
    
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
    
    # Set initial speeds for all motors
    print("Setting initial speeds...")
    for motor in range(1, 9):
        motor_controller.set_speed(motor, 2)
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
                duration = 10.0  # default
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