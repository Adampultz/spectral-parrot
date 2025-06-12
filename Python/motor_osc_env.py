# motor_osc_env.py
import logging
import numpy as np
import time
from simplified_discrete_osc_env import DiscreteOSCEnvironment  # Import the base class

logger = logging.getLogger(__name__)

class DiscreteOSCAndMotorEnvironment(DiscreteOSCEnvironment):
    """
    Extended environment that sends commands to both SuperCollider (via OSC)
    and physical motors (via dual ESP32 controllers).
    """
    
    def __init__(self, spectral_processor, osc_client, 
                 motor_controller=None, port1=None, port2=None, baudrate=115200,
                 num_oscillators=8, amp_range=(0.0, 1.0), 
                 step_wait_time=1.0, reset_wait_time=0.3,
                 reward_scale=0.1, early_stopping_threshold=0.02,
                 use_motors=False, motor_speed=2, motor_reset_speed=2, motor_steps=100):
        """
        Initialize the environment with support for both OSC and motor control.
        """
        # Call parent's init method
        super(DiscreteOSCAndMotorEnvironment, self).__init__(
            spectral_processor=spectral_processor,
            osc_client=osc_client,
            num_oscillators=num_oscillators,
            amp_range=amp_range,
            step_wait_time=step_wait_time,
            reset_wait_time=reset_wait_time,
            reward_scale=reward_scale,
            early_stopping_threshold=early_stopping_threshold
        )
        
        # Additional motor control properties
        self.use_motors = use_motors
        self.motor_controller = motor_controller
        self.motor_speed = motor_speed
        self.motor_reset_speed = motor_reset_speed
        self.motor_steps = motor_steps
        
        # Initialize motor controller if needed
        if self.use_motors and self.motor_controller is None:
            if port1 is None or port2 is None:
                raise ValueError("Serial ports must be specified when use_motors is True")
            
            from dual_esp32_steppers import DualESP32StepperController
            self.motor_controller = DualESP32StepperController(port1, port2, baudrate, debug=False)
            connected = self.motor_controller.connect()
            if not connected:
                logger.warning("Failed to connect to one or both ESP32s. Motor control may be limited.")
    
    def step(self, actions):
        """
        Override step to also control motors.
        """
        freq_commands, amp_values = self._process_actions(actions)
    
        # Track which motors will be moving
        motors_moving = set()
        for i in range(self.num_oscillators):
            motor_num = i + 1
            freq_cmd = freq_commands[i]
            if freq_cmd != 1:  # If not "maintain" (i.e., motor will move)
                motors_moving.add(motor_num)
        
        # Send commands to SuperCollider
        all_params = np.concatenate([freq_commands, amp_values])
        self.send_parameters_to_sc(all_params)
        
        # Send commands to motors if enabled
        if self.use_motors and self.motor_controller:
            self.send_parameters_to_motors(freq_commands, amp_values)
            
            # Wait for all motors to complete their movements, then wait for stabilization
            if motors_moving:
                completion_info = self.wait_for_motors_completion(
                    motors_moving, 
                    timeout=30.0, 
                    stabilization_time=self.step_wait_time  # Use your configured wait time for stabilization
                )
                logger.info(f"Movement and stabilization complete. Motors moved: {len(motors_moving)}")
            else:
                logger.info("No motors moved, proceeding immediately")
        else:
            # If not using motors, just use the standard wait time
            logger.debug(f"Waiting {self.step_wait_time}s after step")
            time.sleep(self.step_wait_time)
        
        # Get new observation
        observation = self.spectral_processor.get_observation().cpu().numpy().flatten()
        
        # Calculate reward based on spectral difference
        reward, current_mse = self.spectral_processor.calculate_reward(
            reward_scale=self.reward_scale, 
            previous_mse=self.previous_mse
        )
        
        # Update previous MSE for next step
        self.previous_mse = current_mse
        
        # Calculate spectral distance (positive value)
        spectral_distance = current_mse
        
        # Check for early stopping (very good match)
        terminated = False
        if spectral_distance < self.early_stopping_threshold:
            logger.info(f"Early stopping: Reached excellent spectral match ({spectral_distance:.6f})")
            terminated = True
            # Add a bonus reward for finding an excellent match
            reward += 2.0 * self.reward_scale
        
        # Check for lack of improvement
        if spectral_distance < self.best_spectral_distance:
            self.best_spectral_distance = spectral_distance
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

        truncated = False
        if self.steps_without_improvement >= self.max_steps_without_improvement:
            logger.info(f"Truncating episode: No improvement for {self.steps_without_improvement} steps")
            truncated = True
        
        # Additional info
        info = {
            'frequency_commands': freq_commands.copy(),
            'amplitudes': amp_values.copy(),
            'spectral_distance': spectral_distance,
            'steps_without_improvement': self.steps_without_improvement,
            'best_spectral_distance': self.best_spectral_distance
        }
        
        return observation, reward, terminated, truncated, info
        
    def _map_internal_to_global_motor(self, esp_num, internal_motor_num):
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

    def wait_for_motors_completion(self, motors_moving, timeout=30.0, stabilization_time=2.0):
        """
        Wait for motors to complete movement with improved tracking and verification.
        """
        if not self.motor_controller:
            return
        
        logger.info(f"Phase 1: Waiting for motors {sorted(motors_moving)} to complete movement")
        
        # Clear any pending responses first
        self.motor_controller.get_responses(clear=True)
        time.sleep(0.1)  # Brief pause to ensure clean slate
        
        # Track completion by global motor number
        motors_pending = set(motors_moving)
        start_time = time.time()
        movement_completion_time = None
        
        # Track which ESP32 we expect responses from
        esp1_motors = {m for m in motors_moving if m % 2 == 1}  # Odd motors
        esp2_motors = {m for m in motors_moving if m % 2 == 0}  # Even motors
        
        logger.info(f"ESP32 #1 should report on motors: {sorted(esp1_motors)}")
        logger.info(f"ESP32 #2 should report on motors: {sorted(esp2_motors)}")
        
        # Keep track of what we've seen
        completion_messages = {}
        
        while motors_pending and (time.time() - start_time < timeout):
            responses = self.motor_controller.get_responses()
            
            for esp_num in [1, 2]:
                for response in responses[esp_num]:
                    logger.debug(f"ESP32 #{esp_num}: {response}")
                    
                    # Parse completion messages
                    completed_motor = None
                    
                    # Try to parse MOTOR_COMPLETE format
                    if "MOTOR_COMPLETE:" in response:
                        try:
                            internal_motor = int(response.split(":")[1].strip())
                            completed_motor = self._map_internal_to_global_motor(esp_num, internal_motor)
                            logger.debug(f"Parsed MOTOR_COMPLETE: internal {internal_motor} -> global {completed_motor}")
                        except Exception as e:
                            logger.error(f"Error parsing MOTOR_COMPLETE: {e}")
                    
                    # Try to parse verbose format
                    elif "movement completed" in response.lower():
                        try:
                            parts = response.split()
                            for i, part in enumerate(parts):
                                if part.lower() == "motor" and i + 1 < len(parts):
                                    internal_motor = int(parts[i + 1])
                                    completed_motor = self._map_internal_to_global_motor(esp_num, internal_motor)
                                    logger.debug(f"Parsed verbose: internal {internal_motor} -> global {completed_motor}")
                                    break
                        except Exception as e:
                            logger.error(f"Error parsing verbose completion: {e}")
                    
                    # Process completed motor
                    if completed_motor is not None:
                        # Verify this motor was expected from this ESP32
                        expected_esp = 1 if completed_motor % 2 == 1 else 2
                        if esp_num != expected_esp:
                            logger.warning(f"Motor {completed_motor} reported by ESP32 #{esp_num} but should be from ESP32 #{expected_esp}!")
                            continue
                        
                        if completed_motor in motors_pending:
                            motors_pending.remove(completed_motor)
                            completion_messages[completed_motor] = response
                            logger.info(f"✓ Motor {completed_motor} completed (ESP32 #{esp_num})")
                            
                            if not motors_pending:
                                movement_completion_time = time.time()
                                logger.info("All motors reported completion!")
                                break
                        else:
                            logger.warning(f"Unexpected completion for motor {completed_motor} - not in pending list")
            
            # Early exit if all done
            if not motors_pending:
                break
                
            time.sleep(0.05)  # Shorter sleep for more responsive checking
        
        # Verification phase - double-check motors have actually stopped
        if movement_completion_time:
            logger.info("Verifying all motors have actually stopped...")
            time.sleep(0.5)  # Wait a bit for any late messages
            
            # Send individual stop commands to ensure motors are stopped
            for motor_num in motors_moving:
                try:
                    self.motor_controller.stop_motor(motor_num)
                    logger.debug(f"Sent verification stop to motor {motor_num}")
                except Exception as e:
                    logger.error(f"Error sending verification stop to motor {motor_num}: {e}")
            
            time.sleep(0.2)  # Brief pause after stops
        
        # Handle timeout case
        if motors_pending:
            logger.warning(f"TIMEOUT: Motors {sorted(motors_pending)} did not report completion!")
            logger.warning("Forcing stop on all pending motors...")
            
            # Try multiple stop strategies
            for motor_num in list(motors_pending):
                try:
                    # Strategy 1: Individual stop
                    self.motor_controller.stop_motor(motor_num)
                    time.sleep(0.05)
                    
                    # Strategy 2: Set speed to 0
                    self.motor_controller.set_speed(motor_num, 0)
                    time.sleep(0.05)
                    
                    # Strategy 3: Move 0 steps
                    self.motor_controller.move_steps(motor_num, 0)
                    
                    logger.info(f"Forced stop for motor {motor_num}")
                except Exception as e:
                    logger.error(f"Error stopping motor {motor_num}: {e}")
            
            # Global stop as last resort
            try:
                self.motor_controller.stop_motor(0)  # Stop all
                logger.info("Sent global stop command")
            except:
                pass
                
            movement_completion_time = time.time()
        
        # Log final summary
        movement_phase_duration = time.time() - start_time
        logger.info(f"Movement phase completed in {movement_phase_duration:.2f}s")
        
        # Detailed summary
        for motor_num in sorted(motors_moving):
            esp_num = 1 if motor_num % 2 == 1 else 2
            status = "✓" if motor_num not in motors_pending else "✗"
            logger.info(f"  Motor {motor_num} (ESP32 #{esp_num}): {status}")
        
        # Phase 2: Stabilization
        logger.info(f"Phase 2: Waiting {stabilization_time}s for string stabilization")
        time.sleep(stabilization_time)
        
        # Final safety stop
        logger.debug("Final safety: stopping all motors")
        try:
            self.motor_controller.stop_motor(0)  # Global stop
            time.sleep(0.1)
            # Individual stops for moving motors
            for motor_num in motors_moving:
                self.motor_controller.stop_motor(motor_num)
                time.sleep(0.05)
        except Exception as e:
            logger.error(f"Error in final stop: {e}")
        
        total_time = time.time() - start_time
        logger.info(f"Total sequence completed in {total_time:.2f}s")
        
        return {
            'motors_completed': list(motors_moving - motors_pending),
            'motors_pending': list(motors_pending),
            'movement_phase_duration': movement_phase_duration,
            'total_duration': total_time,
            'all_completed': len(motors_pending) == 0
        }

    def send_parameters_to_motors(self, freq_commands, amp_values):
        """
        Send only frequency commands to control the motors.
        """
        try:
            if not self.motor_controller:
                logger.warning("Motor controller not available")
                return
                
            # Process each motor
            for i in range(self.num_oscillators):
                motor_num = i + 1  # Convert to 1-indexed for motor controller
                
                # Get frequency command (0=decrease, 1=maintain, 2=increase)
                freq_cmd = freq_commands[i]
                
                if freq_cmd == 1:
                    # Maintain frequency - don't move motor
                    continue
                    
                # Determine which ESP32 controls this motor
                # ODD motors (1,3,5,7) -> ESP32 #1
                # EVEN motors (2,4,6,8) -> ESP32 #2
                esp_num = 1 if motor_num % 2 == 1 else 2
                
                if freq_cmd == 0:
                    # Decrease frequency - move counter-clockwise
                    self.motor_controller.set_speed(motor_num, self.motor_speed)
                    self.motor_controller.set_direction(motor_num, 0)  # 0 = CCW
                    self.motor_controller.move_steps(motor_num, self.motor_steps)
                    logger.debug(f"Motor {motor_num} (ESP32 #{esp_num}): Decrease frequency (CCW) at speed {self.motor_speed}")
                    
                elif freq_cmd == 2:
                    # Increase frequency - move clockwise
                    self.motor_controller.set_speed(motor_num, self.motor_speed)
                    self.motor_controller.set_direction(motor_num, 1)  # 1 = CW
                    self.motor_controller.move_steps(motor_num, self.motor_steps)
                    logger.debug(f"Motor {motor_num} (ESP32 #{esp_num}): Increase frequency (CW) at speed {self.motor_speed}")
                    
        except Exception as e:
            logger.error(f"Error sending parameters to motors: {e}")
        
    def reset(self):
        """
        Override reset to also reset motors.
        """
        # Reset tracking variables
        self.previous_mse = None
        self.best_spectral_distance = float('inf')
        self.steps_without_improvement = 0
        
        # Send reset message to SuperCollider
        logger.info("Sending reset message to SuperCollider")
        self.osc_client.send_message("/reset_oscillators", True)
        
        # Reset motors if using them
        if self.use_motors and self.motor_controller:
            logger.info(f"Resetting motors with speed {self.motor_reset_speed}")
            # Stop all motors
            self.motor_controller.stop_motor(0)
            # Set base speed for all motors
            for i in range(1, self.num_oscillators + 1):
                self.motor_controller.set_speed(i, self.motor_reset_speed)  
        
        # Allow more time for reset with physical components
        time.sleep(self.reset_wait_time)
        
        # Wait for the spectral processor to receive updated data after reset
        start_time = time.time()
        timeout = 2.0  # seconds
        
        while time.time() - start_time < timeout:
            # Check if we have fresh spectral data
            observation = self.spectral_processor.get_observation().cpu().numpy().flatten()
            if np.any(observation != 0):  # If data is non-zero, we have some signal
                break
            time.sleep(0.1)
            
        if time.time() - start_time >= timeout:
            logger.warning("Timeout waiting for spectral data after reset")
        
        # Get the initial observation after reset
        observation = self.spectral_processor.get_observation().cpu().numpy().flatten()
        logger.debug("Reset complete, initial observation shape: %s", observation.shape)
        
        # Return initial observation
        return observation
    
    def close(self):
        """
        Override close to properly clean up motor controller.
        Ensures motors are stopped before disconnecting.
        """
        # Stop all motors with multiple strategies
        if self.use_motors and self.motor_controller:
            logger.info("Performing multi-strategy motor shutdown")
            try:
                # Strategy 1: Send STOP command to all motors
                self.motor_controller.stop_motor(0)
                time.sleep(0.2)
                
                # Strategy 2: Send STOP to each motor individually
                for motor_num in range(1, self.num_oscillators + 1):
                    self.motor_controller.stop_motor(motor_num)
                    time.sleep(0.05)
                
                # Strategy 3: Set speed to 0 for all motors
                for motor_num in range(1, self.num_oscillators + 1):
                    self.motor_controller.set_speed(motor_num, 0)
                    time.sleep(0.05)
                
                # Strategy 4: Send 0 steps to "move" (effectively stop)
                for motor_num in range(1, self.num_oscillators + 1):
                    self.motor_controller.move_steps(motor_num, 0)
                    time.sleep(0.05)
                
                # Final wait to ensure all commands took effect
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error during multi-strategy motor shutdown: {e}")
        
        # Call parent's close method
        super(DiscreteOSCAndMotorEnvironment, self).close()
        
        # Now disconnect from controllers
        if self.use_motors and self.motor_controller:
            self.motor_controller.disconnect()

    def test_motor_completion_detection(self):
        """
        Test function to verify motor completion detection is working correctly.
        """
        logger.info("=== Testing Motor Completion Detection ===")
        
        # Test each motor individually
        for motor_num in range(1, 9):
            esp_num = 1 if motor_num % 2 == 1 else 2
            logger.info(f"\nTesting motor {motor_num} (ESP32 #{esp_num})...")
            
            # Clear responses
            self.motor_controller.get_responses(clear=True)
            
            # Move motor
            self.motor_controller.set_speed(motor_num, 2)
            self.motor_controller.set_direction(motor_num, 1)
            self.motor_controller.move_steps(motor_num, 50)
            
            # Wait and check responses
            time.sleep(5)  # Should be enough time
            
            responses = self.motor_controller.get_responses()
            found_completion = False
            
            for resp_esp in [1, 2]:
                logger.info(f"Responses from ESP32 #{resp_esp}: {len(responses[resp_esp])}")
                for resp in responses[resp_esp]:
                    logger.info(f"  - {resp}")
                    if "complete" in resp.lower():
                        found_completion = True
            
            if not found_completion:
                logger.warning(f"No completion message received for motor {motor_num}!")
            
            # Stop motor
            self.motor_controller.stop_motor(motor_num)
            time.sleep(1)