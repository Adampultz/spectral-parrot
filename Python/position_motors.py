#!/usr/bin/env python3
import argparse
import time
import signal
import sys
from Stepper_Control import DualESP32StepperController

# Global variable to hold controller reference for signal handler
motor_controller = None

motor_speed_max = 500

def emergency_shutdown(signum=None, frame=None):
    """Emergency shutdown procedure for motors"""
    global motor_controller
    
    print("\n\nEMERGENCY SHUTDOWN SEQUENCE")
    if motor_controller:
        try:
            # Multi-strategy approach to ensure motors stop
            print("1. Stopping all motors...")
            motor_controller.stop_motor(0)
            time.sleep(0.2)
            
            print("2. Stopping each motor individually...")
            for motor in range(1, 9):
                motor_controller.stop_motor(motor)
                time.sleep(0.05)
            
            print("3. Setting all speeds to 0...")
            for motor in range(1, 9):
                motor_controller.set_speed(motor, 0)
                time.sleep(0.05)
            
            print("4. Sending 0-step move commands...")
            for motor in range(1, 9):
                motor_controller.move_steps(motor, 0)
                time.sleep(0.05)
                
            # Extra safety: send explicit direction+stop commands
            print("5. Explicit direction stops...")
            for motor in range(1, 9):
                motor_controller.set_direction(motor, 1)  # Clockwise
                motor_controller.stop_motor(motor)
                time.sleep(0.05)
                motor_controller.set_direction(motor, 0)  # Counter-clockwise
                motor_controller.stop_motor(motor)
                time.sleep(0.05)
            
            time.sleep(0.5)
            print("Disconnecting from controllers...")
            motor_controller.disconnect()
            
        except Exception as e:
            print(f"Error during emergency shutdown: {e}")
    
    if signum is not None:
        sys.exit(0)

def interactive_motor_positioning(port1, port2, baudrate=115200):
    """
    Interactive utility to position motors before training.
    """
    global motor_controller
    
    print("\n=== Motor Positioning Utility ===")
    
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
        motor_controller.set_speed(motor, 100)
        time.sleep(0.1)
    
    print("\nMotor positioning commands:")
    print("  <motor> cw <steps>   - Move motor clockwise")
    print("  <motor> ccw <steps>  - Move motor counter-clockwise")
    print("  <motor> speed <val>  - Set motor speed")
    print("  <motor> stop         - Stop motor")
    print("  all stop             - Stop all motors")
    print("  quit                 - Exit the utility with safe shutdown")
    print("\nExample: '3 cw 50' moves motor 3 clockwise by 50 steps")
    print("Press Ctrl+C at any time for emergency shutdown")
    
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
            
            parts = cmd.split()
            if len(parts) < 2:
                print("Invalid command format")
                continue
            
            try:
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
                    if speed < 0 or speed > motor_speed_max:
                        print(f"Speed should be between 0 and {motor_speed_max}")
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
    parser = argparse.ArgumentParser(description="Position motors before training")
    parser.add_argument("--port1", type=str, required=True, help="Port for first ESP32 (odd motors)")
    parser.add_argument("--port2", type=str, required=True, help="Port for second ESP32 (even motors)")
    parser.add_argument("--baudrate", type=int, default=115200, help="Baud rate")
    
    args = parser.parse_args()
    interactive_motor_positioning(args.port1, args.port2, args.baudrate)