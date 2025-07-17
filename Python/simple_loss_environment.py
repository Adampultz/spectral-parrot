# simple_loss_environment.py
import numpy as np
import torch
import logging
from pythonosc import udp_client
import time
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)

class SimpleLossEnvironment(gym.Env):
    """
    A custom Gym environment that uses the multi-scale spectral loss 
    from stft_audio.py as a single observation value.
    """
    
    def __init__(self, loss_processor, osc_client, num_oscillators=8, 
                 amp_range=(0.0, 1.0), step_wait_time=1.0, reset_wait_time=0.3,
                 reward_scale=0.1, early_stopping_threshold=0.02):
        """
        Initialize the environment.
        
        Args:
            loss_processor: SimpleLossProcessor instance that gets loss from stft_audio.py
            osc_client: OSC client to send messages to SuperCollider
            num_oscillators: Number of oscillators to control
            amp_range: Range of amplitudes (min, max) for the oscillators
            step_wait_time: Time to wait after each step (in seconds)
            reset_wait_time: Time to wait after reset (in seconds)
            reward_scale: Scale factor for rewards
            early_stopping_threshold: Loss threshold for early stopping
        """
        super(SimpleLossEnvironment, self).__init__()
        
        self.loss_processor = loss_processor
        self.osc_client = osc_client
        self.num_oscillators = num_oscillators
        self.amp_range = amp_range
        self.amp_range_db = (-70, 0)
        self.step_wait_time = step_wait_time
        self.reset_wait_time = reset_wait_time
        self.reward_scale = reward_scale
        self.early_stopping_threshold = early_stopping_threshold
        
        # Track previous loss for reward calculation
        self.previous_loss = None
    
        # Track best performance for early stopping
        self.best_spectral_loss = float('inf')
        self.steps_without_improvement = 0
        self.max_steps_without_improvement = 100 
        
        # Discrete action space for frequencies (3 actions per oscillator: decrease, maintain, increase)
        self.freq_action_space = spaces.MultiDiscrete([3] * num_oscillators)
        
        # Continuous action space for amplitudes
        self.amp_action_space = spaces.Box(
            low=-3.0, high=3.0,  # Using unbounded values (will be passed through sigmoid)
            shape=(num_oscillators,), 
            dtype=np.float32
        )
        
        # Combined action space
        self.action_space = spaces.Dict({
            'freq_actions': self.freq_action_space,
            'amp_actions': self.amp_action_space
        })
        
        # Observation space is now a single loss value
        self.observation_space = spaces.Box(
            low=0.0, high=100.0,  # Reasonable range for spectral loss values
            shape=(1,),           # Single value
            dtype=np.float32
        )
        
        # Current oscillator parameters (used for tracking only)
        self.current_amps = np.zeros(num_oscillators)
        
        logger.info(f"SimpleLossEnvironment initialized")
        logger.info(f"Observation space: {self.observation_space}")
        logger.info(f"Will receive loss values from stft_audio.py via SimpleLossProcessor")
    
    def amp_to_db(self, amplitude):
        """Convert amplitude to decibels"""
        return 20 * np.log10(np.maximum(amplitude, 1e-10))
    
    def linlin(self, value, in_min, in_max, out_min, out_max, clip=False):
        """Convert a value from input range to output range."""
        if clip:
            value = np.clip(value, in_min, in_max)
        return out_min + (value - in_min) / (in_max - in_min) * (out_max - out_min)
    
    def _process_actions(self, actions):
        """
        Process the discrete frequency actions and continuous amplitude actions.
        
        Args:
            actions: Dictionary with 'freq_actions' and 'amp_actions'
            
        Returns:
            freq_commands: Discrete frequency commands (0=decrease, 1=maintain, 2=increase)
            amp_values: Amplitude values (0-1)
        """
        # Extract actions
        freq_actions = actions['freq_actions']
        amp_actions = actions['amp_actions']
        
        # Process amplitude actions (continuous values through sigmoid)
        amp_actions_norm = 1.0 / (1.0 + np.exp(-amp_actions))
        amp_actions_db = self.amp_to_db(amp_actions_norm)
        amp_values = self.linlin(amp_actions_db, self.amp_range_db[0], self.amp_range_db[1], 0.0, 1.0, True)
        
        # Store current amplitudes for tracking
        self.current_amps = amp_values
        
        return freq_actions, amp_values

    def reset(self, seed=None, options=None):
        """
        Reset the environment.
        
        Returns:
            observation: Initial observation (single loss value)
            info: Additional info dict
        """
        # Reset tracking variables
        self.previous_loss = None
        self.best_spectral_loss = float('inf')
        self.steps_without_improvement = 0
        
        # Send reset message to SuperCollider
        logger.info("Sending reset message to SuperCollider")
        self.osc_client.send_message("/reset_oscillators", True)
        
        # Reset loss processor statistics
        self.loss_processor.reset_stats()
        
        # Allow time for SuperCollider to process the reset
        time.sleep(self.reset_wait_time)
        
        # Wait for the loss processor to receive updated data after reset
        start_time = time.time()
        timeout = 3.0  # seconds
        
        while time.time() - start_time < timeout:
            if self.loss_processor.is_ready():
                break
            time.sleep(0.1)
            
        if time.time() - start_time >= timeout:
            logger.warning("Timeout waiting for loss data after reset")
        
        # Get the initial observation after reset (single loss value)
        observation = self.loss_processor.get_observation().cpu().numpy().flatten()
        logger.debug("Reset complete, initial loss value: %.6f", observation[0])
        
        return observation, {}
    
    def step(self, actions):
        """
        Take actions in the environment.
        
        Args:
            actions: Dictionary with 'freq_actions' and 'amp_actions'
            
        Returns:
            observation: Next state observation (single loss value)
            reward: Reward for the action
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Process actions to get the commands and amplitude values
        freq_commands, amp_values = self._process_actions(actions)
        
        # Send commands to SuperCollider
        all_params = np.concatenate([freq_commands, amp_values])
        self.send_parameters_to_sc(all_params)
        
        # Allow time for oscillators to stabilize and for loss to be calculated
        logger.debug(f"Waiting {self.step_wait_time}s after step")
        time.sleep(self.step_wait_time)
        
        # Get new observation (single loss value from the loss processor)
        observation = self.loss_processor.get_observation().cpu().numpy().flatten()
        
        # Calculate reward based on spectral loss
        reward, current_loss = self.loss_processor.calculate_reward(
            reward_scale=self.reward_scale, 
            previous_loss=self.previous_loss
        )
        
        # Update previous loss for next step
        self.previous_loss = current_loss
        
        # The current loss is available directly
        spectral_loss = current_loss
        
        # Check for early stopping (very good match - low loss)
        terminated = False
        if spectral_loss < self.early_stopping_threshold:
            logger.info(f"Early stopping: Reached excellent spectral match (loss: {spectral_loss:.6f})")
            terminated = True
            # Add a bonus reward for finding an excellent match
            reward += 2.0 * self.reward_scale
        
        # Check for lack of improvement
        if spectral_loss < self.best_spectral_loss:
            self.best_spectral_loss = spectral_loss
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
            'spectral_loss': spectral_loss,
            'steps_without_improvement': self.steps_without_improvement,
            'best_spectral_loss': self.best_spectral_loss
        }
        
        return observation, reward, terminated, truncated, info
    
    def send_parameters_to_sc(self, params):
        """
        Send current oscillator parameters to SuperCollider.
        """
        try:
            # Extract frequency commands and amplitudes
            freq_commands = params[:self.num_oscillators]
            amp_values = params[self.num_oscillators:]
            
            # Convert frequency commands from [0,1,2] to [-1,0,1]
            freq_commands_converted = freq_commands - 1
            
            # Recombine
            converted_params = np.concatenate([freq_commands_converted, amp_values])
            
            # Send to SuperCollider
            self.osc_client.send_message("/set_osc_params", converted_params.tolist())
            
            # Log occasionally to avoid flooding
            if np.random.random() < 0.05:  # 5% chance to log
                logger.debug(f"Sent parameters to SC: {converted_params}")
        except Exception as e:
            logger.error(f"Error sending parameters to SuperCollider: {e}")

    def close(self):
        """
        Clean up resources.
        """
        pass


class SimpleLossMotorEnvironment(SimpleLossEnvironment):
    """
    Extended environment that also controls motors.
    """
    
    def __init__(self, loss_processor, osc_client, 
                 motor_controller=None, port1=None, port2=None, baudrate=115200,
                 num_oscillators=8, amp_range=(0.0, 1.0), 
                 step_wait_time=1.0, reset_wait_time=0.3,
                 reward_scale=0.1, early_stopping_threshold=0.02,
                 use_motors=False, motor_speed=2, motor_reset_speed=2, motor_steps=100):
        """
        Initialize the environment with motor control support.
        """
        # Call parent's init method
        super(SimpleLossMotorEnvironment, self).__init__(
            loss_processor=loss_processor,
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
            
            from Stepper_Control import DualESP32StepperController
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
            
            # Wait for all motors to complete their movements
            if motors_moving:
                completion_info = self.wait_for_motors_completion(
                    motors_moving, 
                    timeout=30.0, 
                    stabilization_time=self.step_wait_time
                )
                logger.info(f"Movement and stabilization complete. Motors moved: {len(motors_moving)}")
            else:
                logger.info("No motors moved, proceeding immediately")
        else:
            # If not using motors, just use the standard wait time
            logger.debug(f"Waiting {self.step_wait_time}s after step")
            time.sleep(self.step_wait_time)
        
        # Get new observation (single loss value from the loss processor)
        observation = self.loss_processor.get_observation().cpu().numpy().flatten()
        
        # Calculate reward based on spectral loss
        reward, current_loss = self.loss_processor.calculate_reward(
            reward_scale=self.reward_scale, 
            previous_loss=self.previous_loss
        )
        
        # Update previous loss for next step
        self.previous_loss = current_loss
        
        # The current loss is available directly
        spectral_loss = current_loss
        
        # Check for early stopping (very good match - low loss)
        terminated = False
        if spectral_loss < self.early_stopping_threshold:
            logger.info(f"Early stopping: Reached excellent spectral match (loss: {spectral_loss:.6f})")
            terminated = True
            # Add a bonus reward for finding an excellent match
            reward += 2.0 * self.reward_scale
        
        # Check for lack of improvement
        if spectral_loss < self.best_spectral_loss:
            self.best_spectral_loss = spectral_loss
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
            'spectral_loss': spectral_loss,
            'steps_without_improvement': self.steps_without_improvement,
            'best_spectral_loss': self.best_spectral_loss
        }
        
        return observation, reward, terminated, truncated, info
        
    def send_parameters_to_motors(self, freq_commands, amp_values):
        """Send only frequency commands to control the motors."""
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
                esp_num = 1 if motor_num % 2 == 1 else 2
                
                if freq_cmd == 0:
                    # Decrease frequency - move counter-clockwise
                    self.motor_controller.set_speed(motor_num, self.motor_speed)
                    self.motor_controller.set_direction(motor_num, 0)  # 0 = CCW
                    self.motor_controller.move_steps(motor_num, self.motor_steps)
                    logger.debug(f"Motor {motor_num} (ESP32 #{esp_num}): Decrease frequency (CCW)")
                    
                elif freq_cmd == 2:
                    # Increase frequency - move clockwise
                    self.motor_controller.set_speed(motor_num, self.motor_speed)
                    self.motor_controller.set_direction(motor_num, 1)  # 1 = CW
                    self.motor_controller.move_steps(motor_num, self.motor_steps)
                    logger.debug(f"Motor {motor_num} (ESP32 #{esp_num}): Increase frequency (CW)")
                    
        except Exception as e:
            logger.error(f"Error sending parameters to motors: {e}")
        
    def wait_for_motors_completion(self, motors_moving, timeout=30.0, stabilization_time=2.0):
        """Wait for motors to complete movement with improved tracking and verification."""
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
        
        # Handle timeout case
        if motors_pending:
            logger.warning(f"TIMEOUT: Motors {sorted(motors_pending)} did not report completion!")
            logger.warning("Forcing stop on all pending motors...")
            
            for motor_num in list(motors_pending):
                try:
                    self.motor_controller.stop_motor(motor_num)
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
            'total_duration': total_time,
            'all_completed': len(motors_pending) == 0
        }
    
    def _map_internal_to_global_motor(self, esp_num, internal_motor_num):
        """Map internal motor number (1-4) from ESP32 to global motor number (1-8)."""
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
    
    def reset(self, seed=None, options=None):
        """Override reset to also reset motors."""
        # Reset tracking variables
        self.previous_loss = None
        self.best_spectral_loss = float('inf')
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
        
        # Reset loss processor statistics
        self.loss_processor.reset_stats()
        
        # Allow more time for reset with physical components
        time.sleep(self.reset_wait_time)
        
        # Wait for the loss processor to receive updated data after reset
        start_time = time.time()
        timeout = 3.0  # seconds
        
        while time.time() - start_time < timeout:
            if self.loss_processor.is_ready():
                break
            time.sleep(0.1)
            
        if time.time() - start_time >= timeout:
            logger.warning("Timeout waiting for loss data after reset")
        
        # Get the initial observation after reset (single loss value)
        observation = self.loss_processor.get_observation().cpu().numpy().flatten()
        logger.debug("Reset complete, initial loss value: %.6f", observation[0])
        
        return observation, {}
    
    def close(self):
        """Override close to properly clean up motor controller."""
        # Stop all motors
        if self.use_motors and self.motor_controller:
            logger.info("Performing motor shutdown")
            try:
                self.motor_controller.stop_motor(0)
                time.sleep(0.5)
                
                for motor_num in range(1, self.num_oscillators + 1):
                    self.motor_controller.stop_motor(motor_num)
                    time.sleep(0.05)
                
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error during motor shutdown: {e}")
        
        # Call parent's close method
        super(SimpleLossMotorEnvironment, self).close()
        
        # Disconnect from controllers
        if self.use_motors and self.motor_controller:
            self.motor_controller.disconnect()