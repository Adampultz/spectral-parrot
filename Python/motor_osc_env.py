# motor_osc_env.py
import logging
import numpy as np
import time
from simplified_discrete_osc_env import DiscreteOSCEnvironment

logger = logging.getLogger(__name__)

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
                 use_motors=False):
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
        # Process actions to get the commands and amplitude values
        freq_commands, amp_values = self._process_actions(actions)
        
        # Send commands to SuperCollider
        all_params = np.concatenate([freq_commands, amp_values])
        self.send_parameters_to_sc(all_params)
        
        # Send commands to motors if enabled
        if self.use_motors and self.motor_controller:
            self.send_parameters_to_motors(freq_commands, amp_values)
        
        # Continue with parent's step method to handle the SuperCollider side
        # Allow fixed time for oscillators/motors to stabilize
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
                    
                elif freq_cmd == 0:
                    # Decrease frequency - move counter-clockwise
                    self.motor_controller.set_direction(motor_num, 0)  # 0 = CCW
                    self.motor_controller.move_steps(motor_num, 100)
                    # Log which ESP32 is handling this motor
                    esp_num = 2 if motor_num % 2 == 0 else 1
                    logger.debug(f"Motor {motor_num} (ESP32 #{esp_num}): Decrease frequency (CCW)")
                    
                elif freq_cmd == 2:
                    # Increase frequency - move clockwise
                    self.motor_controller.set_direction(motor_num, 1)  # 1 = CW
                    self.motor_controller.move_steps(motor_num, 100)
                    # Log which ESP32 is handling this motor
                    esp_num = 2 if motor_num % 2 == 0 else 1
                    logger.debug(f"Motor {motor_num} (ESP32 #{esp_num}): Increase frequency (CW)")
                
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
            logger.info("Resetting motors")
            # Stop all motors
            self.motor_controller.stop_motor(0)
            # Set base speed for all motors
            for i in range(1, self.num_oscillators + 1):
                self.motor_controller.set_speed(i, 50)
        
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