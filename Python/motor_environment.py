# motor_environment.py - Simplified environment for motor control

import numpy as np
import torch
import logging
import time
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Set, Dict, Tuple
from collections import deque

logger = logging.getLogger(__name__)

class MotorEnvironment(gym.Env):
    """
    Simplified Gym environment for motor control based on MSSL feedback.
    Actions are discrete: CCW (0), HOLD (1), CW (2) for each motor.
    Observation is the spectral loss value from MSSL.
    """
    
    def __init__(self, 
                 loss_processor,
                 motor_controller=None,
                 num_motors=8,
                 step_wait_time=3.5,
                 reset_wait_time=0.3,
                 reward_scale=1.0,
                 early_stopping_threshold=0.02,
                 max_steps_without_improvement=100,
                 use_motors=True,
                 motor_speed=200,
                 motor_reset_speed=200,
                 motor_steps=1000):
        """
        Initialize the motor environment.
        
        Args:
            loss_processor: SimpleLossProcessor for MSSL feedback
            motor_controller: DualESP32StepperController instance
            num_motors: Number of motors to control
            step_wait_time: Time to wait after motor movement
            reset_wait_time: Time to wait after reset
            early_stopping_threshold: Loss threshold for episode termination
            max_steps_without_improvement: Steps before truncation
            use_motors: Whether to actually control motors
            motor_speed: Speed for motor movements
            motor_reset_speed: Speed for reset movements
            motor_steps: Steps per motor movement
        """
        super(MotorEnvironment, self).__init__()
        
        # Core components
        self.loss_processor = loss_processor
        self.motor_controller = motor_controller
        self.num_motors = num_motors
        self.use_motors = use_motors
        
        # Timing parameters
        self.step_wait_time = step_wait_time
        self.reset_wait_time = reset_wait_time
        
        # Motor parameters
        self.motor_speed = motor_speed
        self.motor_reset_speed = motor_reset_speed
        self.motor_steps = motor_steps
        
        # Reward configuration
        self.reward_scale = reward_scale
        self.early_stopping_threshold = early_stopping_threshold
        self.max_steps_without_improvement = max_steps_without_improvement
        
        # Define action and observation spaces
        # Actions: 3 discrete choices per motor
        self.action_space = spaces.MultiDiscrete([3] * num_motors)
        
        # Observation: Single loss value from MSSL
        self.observation_space = spaces.Box(
            low=0.0, 
            high=100.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.previous_loss = None
        self.best_loss = float('inf')
        self.steps_without_improvement = 0
        self.episode_steps = 0
        
        # Motor position tracking (for visualization/debugging)
        self.motor_positions = np.zeros(num_motors)  # Cumulative steps
        self.motor_movement_history = deque(maxlen=1000)
        
        # History tracking for sophisticated reward calculation
        self.improvement_history = deque(maxlen=5)
        self.loss_history = deque(maxlen=20)
        
        logger.info(f"MotorEnvironment initialized with {num_motors} motors")
        logger.info(f"Action space: {self.action_space}")
        logger.info(f"Observation space: {self.observation_space}")
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment for a new episode."""
        # Reset episode tracking
        self.previous_loss = None
        self.best_loss = float('inf')
        self.steps_without_improvement = 0
        self.episode_steps = 0
        self.motor_movement_history = []
        
        # Clear history for new episode
        self.improvement_history.clear()
        self.loss_history.clear()
        
        # Reset motors
        if self.use_motors and self.motor_controller:
            logger.info("Resetting motors to initial positions")
            # Stop all motors first
            self.motor_controller.stop_motor(0)  # 0 = all motors
            
            # Set reset speed
            for motor_num in range(1, self.num_motors + 1):
                self.motor_controller.set_speed(motor_num, self.motor_reset_speed)
            
            # Optional: Move motors to home position
            # This depends on your physical setup
            # self._move_motors_to_home()
        
        # Reset loss processor statistics
        self.loss_processor.reset_stats()
        
        # Wait for system to stabilize
        time.sleep(self.reset_wait_time)
        
        # Wait for loss processor to be ready
        start_time = time.time()
        timeout = 3.0
        
        while time.time() - start_time < timeout:
            if self.loss_processor.is_ready():
                break
            time.sleep(0.1)
        
        if not self.loss_processor.is_ready():
            logger.warning("Loss processor not ready after reset timeout")
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            'motor_positions': self.motor_positions.copy(),
            'episode_steps': 0
        }
        
        return observation, info
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take actions in the environment.
        
        Args:
            actions: Array of motor actions [0=CCW, 1=HOLD, 2=CW]
            
        Returns:
            observation: Current loss value
            reward: Reward for this step
            terminated: Whether episode ended (success)
            truncated: Whether episode was cut short
            info: Additional information
        """
        self.episode_steps += 1
        
        # Validate actions
        if len(actions) != self.num_motors:
            raise ValueError(f"Expected {self.num_motors} actions, got {len(actions)}")
        
        # Track which motors are moving
        motors_moving = set()
        movement_commands = {}
        
        for i, action in enumerate(actions):
            motor_num = i + 1  # Motors are 1-indexed
            
            if action == 0:  # CCW
                motors_moving.add(motor_num)
                movement_commands[motor_num] = 'CCW'
                self.motor_positions[i] -= self.motor_steps
            elif action == 2:  # CW
                motors_moving.add(motor_num)
                movement_commands[motor_num] = 'CW'
                self.motor_positions[i] += self.motor_steps
            # action == 1 (HOLD) - no movement
        
        # Record movement for analysis
        self.motor_movement_history.append({
            'step': self.episode_steps,
            'actions': actions.copy(),
            'motors_moved': list(motors_moving)
        })
        
        # Execute motor movements
        if self.use_motors and self.motor_controller and motors_moving:
            self._execute_motor_movements(actions)
            
            # Wait for motors to complete and system to stabilize
            completion_info = self._wait_for_motors_completion(
                motors_moving,
                timeout=30.0,
                stabilization_time=self.step_wait_time
            )
            
            logger.debug(f"Step {self.episode_steps}: Moved motors {sorted(motors_moving)}")
            logger.info(f"Step {self.episode_steps}")
        else:
            # Just wait if no motors moved
            if not motors_moving:
                logger.debug(f"Step {self.episode_steps}: No motors moved (all HOLD)")
            time.sleep(self.step_wait_time)
        
        # Get new observation
        observation = self._get_observation()
        current_loss = float(observation[0])
        
        # Update loss history
        self.loss_history.append(current_loss)
        
        # Calculate reward using the sophisticated reward function
        reward = self._calculate_sophisticated_reward(current_loss)
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Early stopping for good performance
        if current_loss < self.early_stopping_threshold:
            logger.info(f"Episode terminated: Achieved target loss {current_loss:.6f}")
            terminated = True
            reward += 10.0  # Success bonus
        
        # Track improvement
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1
        
        # Truncate if no improvement
        if self.steps_without_improvement >= self.max_steps_without_improvement:
            logger.info(f"Episode truncated: No improvement for {self.steps_without_improvement} steps")
            truncated = True
        
        # Update state
        self.previous_loss = current_loss
        
        # Compile info
        info = {
            'spectral_loss': current_loss,
            'best_loss': self.best_loss,
            'steps_without_improvement': self.steps_without_improvement,
            'motors_moved': list(motors_moving),
            'motor_positions': self.motor_positions.copy(),
            'episode_steps': self.episode_steps
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (loss value)."""
        return self.loss_processor.get_observation().cpu().numpy().flatten()
    
    def _execute_motor_movements(self, actions: np.ndarray):
        """Execute motor movements based on actions."""
        for i, action in enumerate(actions):
            motor_num = i + 1
            
            if action == 0:  # CCW
                self.motor_controller.set_speed(motor_num, self.motor_speed)
                self.motor_controller.set_direction(motor_num, 0)  # 0 = CCW
                self.motor_controller.move_steps(motor_num, self.motor_steps)
                logger.debug(f"Motor {motor_num}: CCW {self.motor_steps} steps")
                
            elif action == 2:  # CW
                self.motor_controller.set_speed(motor_num, self.motor_speed)
                self.motor_controller.set_direction(motor_num, 1)  # 1 = CW
                self.motor_controller.move_steps(motor_num, self.motor_steps)
                logger.debug(f"Motor {motor_num}: CW {self.motor_steps} steps")
            
            # action == 1 (HOLD) - no command sent

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
    
    def _wait_for_motors_completion(self, motors_moving, timeout=30.0, stabilization_time=2.0):
        """
        Wait for motors to complete movement with improved tracking and verification.
        """
        if not self.motor_controller:
            return
        
        # logger.info(f"Phase 1: Waiting for motors {sorted(motors_moving)} to complete movement")
        
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
        
        # logger.info(f"ESP32 #1 should report on motors: {sorted(esp1_motors)}")
        # logger.info(f"ESP32 #2 should report on motors: {sorted(esp2_motors)}")
        
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
                            # logger.info(f"✓ Motor {completed_motor} completed (ESP32 #{esp_num})")
                            
                            if not motors_pending:
                                movement_completion_time = time.time()
                                # logger.info("All motors reported completion!")
                                break
                        else:
                            logger.warning(f"Unexpected completion for motor {completed_motor} - not in pending list")
            
            # Early exit if all done
            if not motors_pending:
                break
                
            time.sleep(0.05)  # Shorter sleep for more responsive checking
        
        # Verification phase - double-check motors have actually stopped
        if movement_completion_time:
            logger.debug("Verifying all motors have actually stopped...")
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
        logger.debug(f"Movement phase completed in {movement_phase_duration:.2f}s")
        
        # Detailed summary
        for motor_num in sorted(motors_moving):
            esp_num = 1 if motor_num % 2 == 1 else 2
            status = "✓" if motor_num not in motors_pending else "✗"
            logger.debug(f"  Motor {motor_num} (ESP32 #{esp_num}): {status}")
        
        # Phase 2: Stabilization
        logger.debug(f"Phase 2: Waiting {stabilization_time}s for string stabilization")
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
        logger.debug(f"Total sequence completed in {total_time:.2f}s")
        
        return {
            'motors_completed': list(motors_moving - motors_pending),
            'motors_pending': list(motors_pending),
            'movement_phase_duration': movement_phase_duration,
            'total_duration': total_time,
            'all_completed': len(motors_pending) == 0
        }
    
    def _calculate_sophisticated_reward(self, current_loss: float) -> float:
        """
        Calculate reward based on spectral loss with improved scaling and shaping.
        
        Args:
            current_loss: Current spectral loss value
            
        Returns:
            reward: Calculated reward
        """
        reward_scale = self.reward_scale
        
        # 1. Exponential reward shaping (more sensitive to small improvements)
        # This gives stronger signal when loss is already low
        base_reward = -np.log(current_loss + 1.0)  # log(1) = 0 for perfect match
        
        # 2. Improvement bonus with momentum consideration
        improvement_bonus = 0
        consistency_bonus = 0
        
        if self.previous_loss is not None:
            improvement = self.previous_loss - current_loss
            
            # Add to improvement history
            self.improvement_history.append(improvement)
            
            # Average recent improvements
            avg_improvement = np.mean(list(self.improvement_history))
            
            # Bonus for improvement, penalty for worsening
            improvement_bonus = improvement * 10.0
            
            # Extra bonus for consistent improvement
            if avg_improvement > 0 and improvement > 0:
                consistency_bonus = min(avg_improvement * 5.0, 1.0)
        
        # 3. Proximity bonus (extra reward when very close to target)
        if current_loss < 0.1:  # Very close
            proximity_bonus = (0.1 - current_loss) * 50.0
        elif current_loss < 0.5:  # Getting close
            proximity_bonus = (0.5 - current_loss) * 10.0
        else:
            proximity_bonus = 0
        
        # 4. Penalty for stagnation
        stagnation_penalty = 0
        if len(self.loss_history) > 10:
            recent_std = np.std(list(self.loss_history)[-10:])
            if recent_std < 1e-4:  # Very little change
                stagnation_penalty = -0.5
        
        # Combine all components
        reward = (base_reward + improvement_bonus + consistency_bonus + 
                 proximity_bonus + stagnation_penalty) * reward_scale
        
        # # Clip to reasonable range
        # reward = np.clip(reward, -10.0 * reward_scale, 10.0 * reward_scale)
        
        # Log detailed breakdown occasionally
        if np.random.random() < 0.02:  # 2% of the time
            logger.debug(f"Reward breakdown: loss={current_loss:.4f}, "
                        f"base={base_reward:.4f}, improve={improvement_bonus:.4f}, "
                        f"consist={consistency_bonus:.4f}, prox={proximity_bonus:.4f}, "
                        f"stag={stagnation_penalty:.4f}, total={reward:.4f}")
        
        return reward
    
    def render(self) -> None:
        """Render the environment (text-based for motors)."""
        print(f"\n=== Step {self.episode_steps} ===")
        print(f"Current loss: {self.previous_loss:.6f}" if self.previous_loss else "Current loss: N/A")
        print(f"Best loss: {self.best_loss:.6f}")
        print("\nMotor positions:")
        
        for i in range(self.num_motors):
            pos = self.motor_positions[i]
            bar_length = int(abs(pos) / 100)
            if pos < 0:
                bar = '◄' + '=' * bar_length
                print(f"Motor {i+1}: {bar:>20} {pos:6.0f}")
            else:
                bar = '=' * bar_length + '►'
                print(f"Motor {i+1}: {pos:6.0f} {bar:<20}")
    
    def close(self):
        """Clean up resources."""
        if self.use_motors and self.motor_controller:
            logger.info("Shutting down motors")
            # Stop all motors
            self.motor_controller.stop_motor(0)
            time.sleep(0.5)
            
            # Individual stops for safety
            for motor_num in range(1, self.num_motors + 1):
                self.motor_controller.stop_motor(motor_num)
                time.sleep(0.05)
            
            # Disconnect
            self.motor_controller.disconnect()
    
    def get_action_distribution_stats(self) -> Dict:
        """Get statistics about action distribution."""
        if not self.motor_movement_history:
            return {}
        
        action_counts = np.zeros((self.num_motors, 3))
        
        for entry in self.motor_movement_history:
            for motor_idx, action in enumerate(entry['actions']):
                action_counts[motor_idx, action] += 1
        
        total_steps = len(self.motor_movement_history)
        action_percentages = action_counts / total_steps * 100
        
        return {
            'action_counts': action_counts,
            'action_percentages': action_percentages,
            'total_steps': total_steps
        }