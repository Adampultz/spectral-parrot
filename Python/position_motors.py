
import argparse
import time
import signal
import sys
import logging
import random
from Stepper_Control import DualESP32StepperController

logger = logging.getLogger(__name__)

# Global variable to hold controller reference for signal handler
motor_controller = None
motor_speed = 200
motor_steps = 500
completion_time = 20
manual_calibration = False

def map_internal_to_global_motor( esp_num, internal_motor_num):
        """
        Map internal motor number (1-4) from ESP32 to global motor number (1-8).
        
        ESP32 #1 controls ODD motors: 
            internal 1 -> global 1
            internal 2 -> global 3
            internal 3 -> global 5
            internal 4 -> global 7
            
        ESP32 #2 controls EVEN motors:
            internal 1 -> global 2
            internal 2 -> global 4
            internal 3 -> global 6
            internal 4 -> global 8
        """
        if internal_motor_num < 1 or internal_motor_num > 4:
            logger.error(f"Invalid internal motor number: {internal_motor_num}")
            return None
            
        if esp_num == 1:  # Odd motors
            return (internal_motor_num * 2) - 1
        elif esp_num == 2:  # Even motors  
            return internal_motor_num * 2
        else:
            logger.error(f"Invalid ESP32 number: {esp_num}")
            return None

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
                # print(f"ESP32 #{esp_num}: {response}")
                
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

def calibrate_find_cw_limits(controller, speed=200):
    """
    Find the clockwise limit for all motors using StallGuard.
    Motors will move CW until StallGuard triggers emergency stop.
    
    Args:
        controller: The motor controller instance
        speed: Speed setting for calibration (slower is safer)
    """
    print("\n" + "="*60)
    print("CALIBRATION: Finding CW Limits (StallGuard)")
    print("="*60)
    print("\n⚠️  WARNING: This will move all motors clockwise until they hit")
    print("their mechanical limits and trigger StallGuard emergency stops.")
    print("\nMake sure:")
    print("  1. Strings are installed and at reasonable tension")
    print("  2. StallGuard thresholds are properly configured")
    print("  3. You're ready to manually stop if needed (Ctrl+C)")
    print(f"\nCalibration speed: {speed} (slow for safety)")

    if manual_calibration:
        response = input("\nProceed with CW limit calibration? (y/n): ").strip().lower()
        if response != 'y':
            print("Calibration cancelled")
            return
    
    print("\n" + "-"*60)
    
    # Clear any pending responses
    controller.get_responses(clear=True)
    
    print("Starting CW movement for all motors...")
    print("Motors will stop when StallGuard detects mechanical limit\n")
    
    # Start all motors moving CW with large step count
    calibration_steps = 60000
    for motor in range(1, 9):
        controller.set_direction(motor, 1)  # 1 = CW
        controller.move_steps(motor, calibration_steps)
        print(f"  Motor {motor}: Moving CW {calibration_steps} steps")
        time.sleep(7)  # Small delay between motor starts
    
    print("\n" + "-"*60)
    print("Monitoring for StallGuard stops...")
    print("(This may take a while depending on initial positions)")
    print("-"*60)
    
    # Monitor for completion messages
    motors_stopped = set()
    start_time = time.time()
    timeout = 120  # 2 minutes timeout
    
    while len(motors_stopped) < 8 and (time.time() - start_time < timeout):
        responses = controller.get_responses()
        
        for esp_num in [1, 2]:
            for response in responses[esp_num]:
                # print(f"ESP32 #{esp_num}: {response}")
                
                # Look for StallGuard detection
                if "STALL DETECTED" in response:
                    # Extract motor number
                    parts = response.split()
                    for i, part in enumerate(parts):
                        if part.lower() == "motor" and i + 1 < len(parts):
                            try:
                                internal_motor_num = int(response.split(":")[1])
                                global_motor_num = map_internal_to_global_motor(esp_num, internal_motor_num)
                                if global_motor_num:
                                    motors_stopped.add(global_motor_num)
                                    print(f"\n✓ Motor {global_motor_num} reached CW limit (StallGuard triggered)")
                            except:
                                pass
                
                # Also check for completion messages
                elif "MOTOR_COMPLETE:" in response:
                    try:
                        internal_motor_num = int(response.split(":")[1])
                        global_motor_num = map_internal_to_global_motor(esp_num, internal_motor_num)
                        if global_motor_num:
                            motors_stopped.add(global_motor_num)
                        print(f"✓ Motor {global_motor_num} stopped")
                    except:
                        pass
        
        # Show progress
        if len(motors_stopped) > 0:
            print(f"\rProgress: {len(motors_stopped)}/8 motors at CW limit", end='')
        
        time.sleep(0.5)
    
    print("\n\n" + "="*60)
    
    if len(motors_stopped) == 8:
        print("✅ CALIBRATION SUCCESSFUL")
        print("All motors have found their CW limits")
    else:
        print("⚠️  CALIBRATION INCOMPLETE")
        print(f"Only {len(motors_stopped)}/8 motors reached their limits")
        missing = set(range(1, 9)) - motors_stopped
        if missing:
            print(f"Motors not calibrated: {sorted(missing)}")
            print("\nTroubleshooting:")
            print("  - Check StallGuard threshold settings")
            print("  - Verify mechanical connections")
            print("  - Try individual motor calibration")
    
    print("="*60)
    
    # Stop any motors still moving
    print("\nStopping all motors...")
    controller.stop_motor(0)  # 0 = all motors
    time.sleep(0.5)
    
    return motors_stopped

def calibrate_find_ccw_position(controller, target_position=-4000, speed=200):
    """
    Move all motors to a specific CCW position from their current (limit) position.
    FIXED to properly track motors that stop early due to StallGuard vs completing full movement.
    """
    print("\n" + "="*60)
    print(f"CALIBRATION: Moving to CCW Position ({target_position} steps from CW limit)")
    print("="*60)
    print("\nThis assumes motors are currently at their CW limits.")
    print(f"Will move all motors {abs(target_position)} steps counter-clockwise.")
    
    if manual_calibration:
        response = input(f"\nMove all motors to position {target_position}? (y/n): ").strip().lower()
        if response != 'y':
            print("Positioning cancelled")
            return
    
    # Set speed
    print(f"Setting speed to {speed}...")
    for motor in range(1, 9):
        controller.set_speed(motor, speed)
        time.sleep(0.05)
    
    # Move all motors CCW
    steps_to_move = abs(target_position)
    print(f"Moving all motors {steps_to_move} steps CCW...")
    
    for motor in range(1, 9):
        controller.send_command(motor, "INHIBIT_SG", 1)  # Disable stallguard triggering
        controller.set_direction(motor, 0)  # 0 = CCW
        controller.move_steps(motor, steps_to_move)
        print(f"  Motor {motor}: Moving CCW {steps_to_move} steps")
        logger.info(f"Inhibited StallGuard triggering for motor {motor}")
        time.sleep(7.0)
    
    # Wait for completion
    print("\nWaiting for motors to reach position...")
    time.sleep(steps_to_move / (speed * 256) + 2)  # Rough estimate + buffer
    
    # Monitor completion with PROPER tracking
    start_time = time.time()
    timeout = 60
    motors_completed_full = set()      # Motors that completed the full movement
    motors_stopped_early = set()       # Motors that stopped early (StallGuard)
    
    while len(motors_completed_full) + len(motors_stopped_early) < 8 and (time.time() - start_time < timeout):
        responses = controller.get_responses()
        
        for esp_num in [1, 2]:
            for response in responses[esp_num]:
                if not response.strip():
                    continue
                
                # Handle STALL DETECTED (motor hit CCW limit)
                if "STALL DETECTED" in response:
                    parts = response.split()
                    for i, part in enumerate(parts):
                        if part.lower() == "motor" and i + 1 < len(parts):
                            try:
                                internal_motor_num = int(parts[i + 1])
                                global_motor_num = map_internal_to_global_motor(esp_num, internal_motor_num)
                                
                                if global_motor_num and global_motor_num not in motors_stopped_early and global_motor_num not in motors_completed_full:
                                    motors_stopped_early.add(global_motor_num)
                                    print(f"⚠️  Motor {global_motor_num} stopped early (hit CCW limit)")
                                break
                            except (ValueError, IndexError):
                                pass
                
                # Handle MOTOR_COMPLETE (motor completed full movement)
                elif "MOTOR_COMPLETE:" in response:
                    try:
                        internal_motor_num = int(response.split(":")[1].strip())
                        global_motor_num = map_internal_to_global_motor(esp_num, internal_motor_num)
                        
                        if global_motor_num and global_motor_num not in motors_completed_full and global_motor_num not in motors_stopped_early:
                            motors_completed_full.add(global_motor_num)
                            print(f"✓ Motor {global_motor_num} completed full movement")
                            controller.send_command(global_motor_num, "ALLOW_SG", 1)
                            logger.info(f"Allowed StallGuard triggering for motor {global_motor_num}")
                    except (ValueError, IndexError):
                        pass
        
        # Show progress
        total_done = len(motors_completed_full) + len(motors_stopped_early)
        if total_done > 0:
            print(f"\rProgress: {total_done}/8 motors done ({len(motors_completed_full)} full, {len(motors_stopped_early)} early)", end='')
        
        time.sleep(0.2)

    print("\nRe-enabling StallGuard for all motors...")
    for motor in range(1, 9):
        controller.send_command(motor, "ALLOW_SG", 1)
        time.sleep(0.05)
    
    print("\n\n" + "="*60)
    print("POSITIONING RESULTS")
    print("="*60)
    
    if len(motors_completed_full) == 8:
        print("✅ ALL MOTORS positioned correctly")
        print(f"All motors moved {abs(target_position)} steps CCW to center position")
    
    elif len(motors_completed_full) + len(motors_stopped_early) == 8:
        print("⚠️  MIXED RESULTS - Some motors stopped early")
        print(f"Motors positioned correctly: {sorted(motors_completed_full)} ({len(motors_completed_full)}/8)")
        print(f"Motors stopped early (hit CCW limit): {sorted(motors_stopped_early)} ({len(motors_stopped_early)}/8)")
        
        if motors_stopped_early:
            print(f"\n💡 Motors {sorted(motors_stopped_early)} may need:")
            print(f"   - Different StallGuard threshold (currently ~150)")
            print(f"   - Smaller center offset (try -3000 or -2000 instead of {target_position})")
            print(f"   - Check if they were actually at CW limits before centering")
    
    else:
        missing = set(range(1, 9)) - motors_completed_full - motors_stopped_early
        print(f"❌ INCOMPLETE - Motors didn't respond: {sorted(missing)}")
    
    print("="*60)
    
    return {
        'completed_full': motors_completed_full,
        'stopped_early': motors_stopped_early,
        'success': len(motors_completed_full) == 8
    }

def calibrate_random_position(controller, center_position=-4000, random_range=2000, speed=200, manual_calibration=False):
    """
    Move all motors to random positions around center from their current CW limit position.
    Each motor gets an individual random offset of ±random_range from center_position.
    
    Args:
        controller: The motor controller instance
        center_position: Center position in steps from CW limit (default: -4000)
        random_range: Random offset range in steps (default: ±2000)
        speed: Movement speed (default: 200)
        manual_calibration: Whether to ask for user confirmation
    """
    # Generate random offsets for each motor
    random_offsets = [random.randint(-random_range, random_range) for _ in range(8)]
    target_positions = [center_position + offset for offset in random_offsets]
    
    print("\n" + "="*60)
    print("CALIBRATION: Moving to Random Positions")
    print("="*60)
    print(f"\nCenter position: {center_position} steps from CW limit")
    print(f"Random range: ±{random_range} steps")
    print("\nRandom target positions for each motor:")
    for i, (offset, target) in enumerate(zip(random_offsets, target_positions), 1):
        print(f"  Motor {i}: {center_position} + ({offset:+d}) = {target} steps from CW limit")
    
    # Safety check - ensure no position goes beyond reasonable limits
    min_safe_position = -8000  # Don't go more than 8000 steps CCW from CW limit
    max_safe_position = -1000  # Don't go closer than 1000 steps to CW limit
    
    safe_positions = []
    warnings = []
    for i, target in enumerate(target_positions, 1):
        if target < min_safe_position:
            safe_target = min_safe_position
            warnings.append(f"Motor {i}: Limited to {safe_target} (was {target})")
        elif target > max_safe_position:
            safe_target = max_safe_position
            warnings.append(f"Motor {i}: Limited to {safe_target} (was {target})")
        else:
            safe_target = target
        safe_positions.append(safe_target)
    
    if warnings:
        print(f"\n⚠️  Safety limits applied:")
        for warning in warnings:
            print(f"  {warning}")
    
    print(f"\nThis assumes motors are currently at their CW limits.")
    
    if manual_calibration:
        response = input(f"\nProceed with random positioning? (y/n): ").strip().lower()
        if response != 'y':
            print("Random positioning cancelled")
            return
    
    # Set speed for all motors
    print(f"Setting speed to {speed}...")
    for motor in range(1, 9):
        controller.set_speed(motor, speed)
        time.sleep(0.05)
    
    # Move each motor to its individual random position
    print(f"Moving motors to random positions...")
    
    for motor_num in range(1, 9):
        target_pos = safe_positions[motor_num - 1]
        steps_to_move = abs(target_pos)
        
        controller.send_command(motor_num, "INHIBIT_SG", 1)  # Disable stallguard triggering
        controller.set_direction(motor_num, 0)  # 0 = CCW
        controller.move_steps(motor_num, steps_to_move)
        print(f"  Motor {motor_num}: Moving CCW {steps_to_move} steps to position {target_pos}")
        logger.info(f"Inhibited StallGuard triggering for motor {motor_num}")
        time.sleep(7.0)  # Delay between motor starts
    
    # Wait for completion
    max_steps = max(abs(pos) for pos in safe_positions)
    estimated_time = max_steps / (speed * 256) + 2
    print(f"\nWaiting for motors to reach random positions...")
    print(f"Estimated completion time: {estimated_time:.1f} seconds")
    time.sleep(estimated_time)
    
    # Monitor completion with proper tracking
    start_time = time.time()
    timeout = 60
    motors_completed_full = set()      # Motors that completed the full movement
    motors_stopped_early = set()       # Motors that stopped early (StallGuard)
    
    while len(motors_completed_full) + len(motors_stopped_early) < 8 and (time.time() - start_time < timeout):
        responses = controller.get_responses()
        
        for esp_num in [1, 2]:
            for response in responses[esp_num]:
                if not response.strip():
                    continue
                
                # Handle STALL DETECTED (motor hit CCW limit)
                if "STALL DETECTED" in response:
                    parts = response.split()
                    for i, part in enumerate(parts):
                        if part.lower() == "motor" and i + 1 < len(parts):
                            try:
                                internal_motor_num = int(parts[i + 1])
                                global_motor_num = map_internal_to_global_motor(esp_num, internal_motor_num)
                                
                                if global_motor_num and global_motor_num not in motors_stopped_early and global_motor_num not in motors_completed_full:
                                    motors_stopped_early.add(global_motor_num)
                                    print(f"⚠️  Motor {global_motor_num} stopped early (hit CCW limit)")
                                break
                            except (ValueError, IndexError):
                                pass
                
                # Handle MOTOR_COMPLETE (motor completed full movement)
                elif "MOTOR_COMPLETE:" in response:
                    try:
                        internal_motor_num = int(response.split(":")[1].strip())
                        global_motor_num = map_internal_to_global_motor(esp_num, internal_motor_num)
                        
                        if global_motor_num and global_motor_num not in motors_completed_full and global_motor_num not in motors_stopped_early:
                            motors_completed_full.add(global_motor_num)
                            print(f"✅ Motor {global_motor_num} reached random position")
                            controller.send_command(global_motor_num, "ALLOW_SG", 1)
                            logger.info(f"Allowed StallGuard triggering for motor {global_motor_num}")
                    except (ValueError, IndexError):
                        pass
        
        # Show progress
        total_done = len(motors_completed_full) + len(motors_stopped_early)
        if total_done > 0:
            print(f"\rProgress: {total_done}/8 motors positioned ({len(motors_completed_full)} full, {len(motors_stopped_early)} early)", end='')
        
        time.sleep(0.2)

    print("\nRe-enabling StallGuard for all motors...")
    for motor in range(1, 9):
        controller.send_command(motor, "ALLOW_SG", 1)
        time.sleep(0.05)
    
    print("\n\n" + "="*60)
    print("RANDOM POSITIONING RESULTS")
    print("="*60)
    
    if len(motors_completed_full) == 8:
        print("✅ ALL MOTORS positioned correctly")
        print("All motors successfully moved to their random positions")
    
    elif len(motors_completed_full) + len(motors_stopped_early) == 8:
        print("⚠️  MIXED RESULTS - Some motors stopped early")
        print(f"Motors positioned correctly: {sorted(motors_completed_full)} ({len(motors_completed_full)}/8)")
        print(f"Motors stopped early (hit CCW limit): {sorted(motors_stopped_early)} ({len(motors_stopped_early)}/8)")
        
        if motors_stopped_early:
            print(f"\n💡 Motors {sorted(motors_stopped_early)} may need:")
            print(f"   - Smaller random range (try ±1000 instead of ±{random_range})")
            print(f"   - Different center position (try -3000 instead of {center_position})")
    
    else:
        missing = set(range(1, 9)) - motors_completed_full - motors_stopped_early
        print(f"❌ INCOMPLETE - Motors didn't respond: {sorted(missing)}")
    
    print("="*60)
    
    # Print final positions summary
    print("\nFinal Random Positions Summary:")
    for i, (offset, target, safe_pos) in enumerate(zip(random_offsets, target_positions, safe_positions), 1):
        status = "✅" if i in motors_completed_full else ("⚠️" if i in motors_stopped_early else "❌")
        if target != safe_pos:
            print(f"  Motor {i}: {status} {safe_pos} steps (limited from {target})")
        else:
            print(f"  Motor {i}: {status} {safe_pos} steps")
    
    return {
        'completed_full': motors_completed_full,
        'stopped_early': motors_stopped_early,
        'success': len(motors_completed_full) == 8,
        'positions': safe_positions,
        'offsets': random_offsets
    }


def calibrate_full_sequence(controller, manual_calibration=False):
    """
    Full calibration sequence:
    1. Find CW limits using StallGuard
    2. Move to center position
    """
    print("\n" + "="*70)
    print("FULL CALIBRATION SEQUENCE")
    print("="*70)
    print("\nThis will:")
    print("  1. Move all motors CW until StallGuard triggers (find limits)")
    print("  2. Move all motors CCW to center position")
    print("  3. Set this as the calibrated center/home position")

    if manual_calibration:
        response = input("\nStart full calibration? (y/n): ").strip().lower()
        if response != 'y':
            print("Calibration cancelled")
            return
    
    # Step 1: Find CW limits
    print("\n--- STEP 1: Finding CW Limits ---")
    motors_calibrated = calibrate_find_cw_limits(controller, speed=motor_speed)
    
    if len(motors_calibrated) != 8:
        print("\n⚠️  Not all motors found their limits.")

        if manual_calibration:
            response = input("Continue anyway? (y/n): ").strip().lower()
            if response != 'y':
                return
    
    time.sleep(2)
    
    # Step 2: Move to center
    print("\n--- STEP 2: Moving to Center Position ---")
    center_offset = -4000  # 4000 steps CCW from CW limit
    calibrate_find_ccw_position(controller, target_position=center_offset, speed=motor_speed)
    
    print("\n" + "="*70)
    print("✅ CALIBRATION COMPLETE")
    print("="*70)
    print("\nMotors are now at calibrated center position.")
    print(f"Position: {center_offset} steps from CW limit")
    print("\nYou can now:")
    print("  1. Run your training script")
    print("  2. Fine-tune individual motor positions")
    print("  3. Save this as your standard starting position")

def calibrate_full_sequence_with_random(controller, manual_calibration=False):
    """
    Full calibration sequence with random positioning:
    1. Find CW limits using StallGuard
    2. Move to random positions around center
    """
    print("\n" + "="*70)
    print("FULL CALIBRATION SEQUENCE (WITH RANDOM POSITIONING)")
    print("="*70)
    print("\nThis will:")
    print("  1. Move all motors CW until StallGuard triggers (find limits)")
    print("  2. Move all motors CCW to random positions around center")
    print("  3. Set these as the randomized starting positions")

    if manual_calibration:
        response = input("\nStart full calibration with random positioning? (y/n): ").strip().lower()
        if response != 'y':
            print("Calibration cancelled")
            return
    
    # Step 1: Find CW limits
    print("\n--- STEP 1: Finding CW Limits ---")
    motors_calibrated = calibrate_find_cw_limits(controller, speed=motor_speed)
    
    if len(motors_calibrated) != 8:
        print("\n⚠️  Not all motors found their limits.")

        if manual_calibration:
            response = input("Continue anyway? (y/n): ").strip().lower()
            if response != 'y':
                return
    
    time.sleep(2)
    
    # Step 2: Move to random positions
    print("\n--- STEP 2: Moving to Random Positions ---")
    result = calibrate_random_position(controller, speed=motor_speed, manual_calibration=False)
    
    print("\n" + "="*70)
    print("✅ RANDOMIZED CALIBRATION COMPLETE")
    print("="*70)
    print("\nMotors are now at randomized positions around center.")
    print(f"Random offsets applied: {result.get('offsets', 'N/A')}")
    print("\nYou can now:")
    print("  1. Run your training script")
    print("  2. Re-randomize positions with 'cal random'")
    print("  3. Move to standard center with 'cal center'")
    
    return result 

def calibrate_individual_motor(controller, motor_num):
    """
    Calibrate a single motor to find its CW limit.
    Fixed to properly handle internal-to-global motor number mapping.
    """
    print(f"\n=== Calibrating Motor {motor_num} ===")
    
    # Clear responses
    controller.get_responses(clear=True)
    
    # Set slow speed
    controller.set_speed(motor_num, 3)
    time.sleep(0.1)
    
    # Move CW until StallGuard
    print(f"Moving motor {motor_num} CW until StallGuard triggers...")
    controller.set_direction(motor_num, 1)
    controller.move_steps(motor_num, 30000)
    
    # Monitor for stop
    start_time = time.time()
    timeout = 60
    stopped = False
    
    while not stopped and (time.time() - start_time < timeout):
        responses = controller.get_responses()
        esp_num = 1 if motor_num % 2 == 1 else 2
        
        for response in responses[esp_num]:
            # print(f"ESP32 #{esp_num}: {response}")
            
            if "STALL DETECTED" in response or "MOTOR_COMPLETE" in response:
                try:
                    # Extract internal motor number from response
                    internal_motor_num = None
                    
                    if "MOTOR_COMPLETE:" in response:
                        # Format: "MOTOR_COMPLETE:3"
                        internal_motor_num = int(response.split(":")[1].strip())
                    elif "STALL DETECTED" in response:
                        # Format: "!!! Motor 3 STALL DETECTED (SG: 148) - Emergency stop"
                        parts = response.split()
                        for i, part in enumerate(parts):
                            if part.lower() == "motor" and i + 1 < len(parts):
                                internal_motor_num = int(parts[i + 1])
                                break
                    
                    # Map internal motor number to global motor number
                    if internal_motor_num is not None:
                        global_motor_num = map_internal_to_global_motor(esp_num, internal_motor_num)
                        
                        # Check if this is the motor we're calibrating
                        if global_motor_num == motor_num:
                            stopped = True
                            print(f"✓ Motor {motor_num} calibrated (ESP32 #{esp_num} internal motor {internal_motor_num})")
                            break
                        else:
                            print(f"  -> Different motor completed: global {global_motor_num} (internal {internal_motor_num})")
                
                except (ValueError, IndexError) as e:
                    print(f"  -> Error parsing response: {e}")
        
        time.sleep(0.5)
    
    if not stopped:
        print(f"⚠️  Motor {motor_num} calibration timeout")
        controller.stop_motor(motor_num)

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
                # print(f"ESP32 #{resp_esp_num}: {response}")
                
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

    print("\nCalibration Commands:")
    print("  cal limits           - Find CW limits for all motors")
    print("  cal center           - Move to center from current position")
    print("  cal random           - Move to random positions around center") 
    print("  cal full             - Full calibration sequence")
    print("  cal motor <num>      - Calibrate individual motor")

    print("  quit                 - Exit the utility with safe shutdown")
    
    try:
        while True:
            cmd = input("\nCommand: ").strip().lower()
            
            if cmd in ["exit", "quit"]:
                print("Performing safe shutdown sequence...")
                emergency_shutdown()
                break

            elif cmd.startswith("cal"):
                parts = cmd.split()
                if len(parts) < 2:
                    print("Specify calibration type: limits, center, full, or motor <num>")
                    continue
                
                cal_type = parts[1]
                
                if cal_type == "limits":
                    calibrate_find_cw_limits(motor_controller)
                elif cal_type == "center":
                    calibrate_find_ccw_position(motor_controller)
                elif cal_type == "full":
                    calibrate_full_sequence(motor_controller)
                elif cal_type == "motor" and len(parts) > 2:
                    try:
                        motor_num = int(parts[2])
                        if 1 <= motor_num <= 8:
                            calibrate_individual_motor(motor_controller, motor_num)
                        else:
                            print("Motor number must be 1-8")
                    except ValueError:
                        print("Invalid motor number")
                elif cal_type == "random":
                    calibrate_random_position(motor_controller, manual_calibration=manual_calibration)
                else:
                    print("Unknown calibration command")
                
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