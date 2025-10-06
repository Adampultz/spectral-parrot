# motor_environment.py - environment for motor control

import numpy as np
import torch
import logging
import time
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Set, List, Dict, Tuple
from collections import deque
from position_motors import calibrate_full_sequence, calibrate_full_sequence_with_random, map_internal_to_global_motor

logger = logging.getLogger(__name__)

class SoftLimitHandler:
    """Soft penalty zones using existing max_ccw_steps and max_cw_steps."""
    
    def __init__(self, max_ccw_steps, max_cw_steps, num_motors=8, 
                 danger_zone_ratio=0.1, critical_zone_ratio=0.02):
        self.max_ccw_steps = max_ccw_steps  # List from config
        self.max_cw_steps = max_cw_steps    # List from config
        self.num_motors = num_motors
        
        # Soft boundary zones (percentage from limits)
        self.danger_zone_ratio = danger_zone_ratio   # Start penalties at 20% from limit
        self.critical_zone_ratio = critical_zone_ratio # Severe penalties at 5% from limit
     
    def get_limit_penalty(self, motor_idx, position):
        """
        Calculate smooth penalty that increases exponentially near limits.
        motor_idx: 0-based index (0-7)
        position: current motor position in steps
        """
        # Get limits for this motor
        min_limit = -self.max_ccw_steps[motor_idx]
        max_limit = self.max_cw_steps[motor_idx]
        range_size = max_limit - min_limit
        
        # Calculate distances to limits
        dist_to_min = position - min_limit
        dist_to_max = max_limit - position
        
        # Convert to ratios
        dist_to_min_ratio = dist_to_min / range_size
        dist_to_max_ratio = dist_to_max / range_size
        
        # Find closest boundary
        if dist_to_min_ratio < dist_to_max_ratio:
            # Closer to CCW limit
            dist_ratio = dist_to_min_ratio
        else:
            # Closer to CW limit
            dist_ratio = dist_to_max_ratio
        
        # No penalty in safe zone
        if dist_ratio > self.danger_zone_ratio:
            return 0.0
        
        # Exponential penalty in danger zone
        if dist_ratio > self.critical_zone_ratio:
            # Smooth exponential from 0 to 5
            x = (self.danger_zone_ratio - dist_ratio) / (self.danger_zone_ratio - self.critical_zone_ratio)
            return 5.0 * (np.exp(2 * x) - 1) / (np.exp(2) - 1)
        
        # Severe penalty in critical zone
        x = (self.critical_zone_ratio - dist_ratio) / self.critical_zone_ratio
        return 5.0 + 15.0 * x  # 5 to 20 penalty

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
                motor_steps=100,
                max_ccw_steps: List[int] = None,
                max_cw_steps: List[int] = None,
                cw_limit_position: int = 5000,  # Position when at CW limit (StallGuard message from ESP32)
                auto_recalibrate: bool = True,   # Auto-recalibrate on StallGuard
                limit_penalty=5.0,
                adaptive_hold_bias=1.0,
                manual_calibration=True,
                reset_calibration=0,
                target_loss=15.0,
                initial_movement_penalty=1.0,
                final_movement_penalty=0.1,
                penalty_decay_episodes=50,
                danger_zone_ratio=0.1,
                critical_zone_ratio=0.02,
                ccw_safety_margin=200,
                stagnation_threshold=0.5,
                stagnation_window=20,
                motor_completion_timeout=30.0,
                stabilization_time=2.0,
                loss_clip_max=50.0,
                use_improvement_bonus=True,
                use_consistency_bonus=False,
                use_breakthrough_bonus=True,
                use_movement_penalty=False,
                use_stagnation_penalty=False,
                use_efficiency_bonus=False,
                use_proximity_bonus=False,
                improvement_bonus_weight=1.0,
                consistency_bonus_weight=5.0,
                breakthrough_bonus_weight=10.0,
                movement_penalty_weight=1.0,
                efficiency_bonus_weight=0.5,
                proximity_threshold_close=0.5,
                proximity_threshold_very_close=0.1,
                proximity_bonus_close=10.0,
                proximity_bonus_very_close=50.0,
                episode_steps_before_breakthrough=30,
                motors_for_movement_penalty=2,
                observation_space_loss_max=100.0,
                observation_space_loss_min=0.0,
                use_early_stopping=True,              
                use_truncation=True,                
                min_episode_steps=1,                
                reward_threshold_for_early_stop=None, 
                log_motor_details=False):       
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

        self.adaptive_hold_bias = adaptive_hold_bias
        self.stagnation_threshold = 0.5
        self.stagnation_window = 20
        self.manual_calibration = manual_calibration
        self.reset_calibration = reset_calibration
        
        # Reward configuration
        self.reward_scale = reward_scale
        self.early_stopping_threshold = early_stopping_threshold
        self.max_steps_without_improvement = max_steps_without_improvement
        
        # Define action and observation spaces
        # Actions: 3 discrete choices per motor
        self.action_space = spaces.MultiDiscrete([3] * num_motors)
        
        # Update observation space with config
        self.observation_space = spaces.Box(
            low=np.array([observation_space_loss_min, -1.0], dtype=np.float32),
            high=np.array([observation_space_loss_max, 1.0], dtype=np.float32),
            shape=(2,),
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

        self.initial_movement_penalty = 1.0  # Strong penalty early in training
        self.final_movement_penalty = 0.1    # Weak penalty late in training
        self.penalty_decay_episodes = 50     # Episodes to decay penalty over
        self.current_episode = 0
        
        # History tracking for sophisticated reward calculation
        self.improvement_history = deque(maxlen=5)
        self.loss_history = deque(maxlen=20)

        self.max_ccw_steps = max_ccw_steps
        self.max_cw_steps = max_cw_steps
        self.cw_limit_position = cw_limit_position
        self.ccw_limit_reached = np.zeros(num_motors, dtype=bool)
        self.auto_recalibrate = auto_recalibrate
        self.limit_penalty = limit_penalty
        self.limit_violations = np.zeros(num_motors, dtype=int)

        self.target_loss = target_loss
        self.initial_movement_penalty = initial_movement_penalty
        self.final_movement_penalty = final_movement_penalty
        self.penalty_decay_episodes = penalty_decay_episodes
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_window = stagnation_window
        self.motor_completion_timeout = motor_completion_timeout
        self.stabilization_time = stabilization_time
        self.ccw_safety_margin = ccw_safety_margin

        # Store reward function config
        self.use_improvement_bonus = use_improvement_bonus
        self.use_consistency_bonus = use_consistency_bonus
        self.use_breakthrough_bonus = use_breakthrough_bonus
        self.use_movement_penalty = use_movement_penalty
        self.use_stagnation_penalty = use_stagnation_penalty
        self.use_efficiency_bonus = use_efficiency_bonus
        self.use_proximity_bonus = use_proximity_bonus
        
        self.improvement_bonus_weight = improvement_bonus_weight
        self.consistency_bonus_weight = consistency_bonus_weight
        self.breakthrough_bonus_weight = breakthrough_bonus_weight
        self.movement_penalty_weight = movement_penalty_weight
        self.efficiency_bonus_weight = efficiency_bonus_weight
        
        self.proximity_threshold_close = proximity_threshold_close
        self.proximity_threshold_very_close = proximity_threshold_very_close
        self.proximity_bonus_close = proximity_bonus_close
        self.proximity_bonus_very_close = proximity_bonus_very_close
        
        self.episode_steps_before_breakthrough = episode_steps_before_breakthrough
        self.motors_for_movement_penalty = motors_for_movement_penalty

        self.use_early_stopping = use_early_stopping
        self.use_truncation = use_truncation
        self.min_episode_steps = min_episode_steps
        self.reward_threshold_for_early_stop = reward_threshold_for_early_stop
        self.log_motor_details = log_motor_details

        self.limit_handler = SoftLimitHandler(
            self.max_ccw_steps, 
            self.max_cw_steps, 
            self.num_motors,
            danger_zone_ratio=danger_zone_ratio,
            critical_zone_ratio=critical_zone_ratio
        )

        if self.use_motors and self.motor_controller:
            logger.info("Enabling motor drivers...")
            try:
                self.motor_controller.send_command(0, "ENABLE", 1)
                time.sleep(0.5)
                logger.info("All motor drivers enabled and ready")
            except Exception as e:
                logger.error(f"Failed to enable motors: {e}")
        
        logger.info(f"MotorEnvironment initialized with {num_motors} motors")
        logger.info(f"Action space: {self.action_space}")
        logger.info(f"Observation space: {self.observation_space}")

    def _get_adaptive_movement_penalty(self):
        """Calculate current movement penalty based on training progress."""
        if self.current_episode >= self.penalty_decay_episodes:
            return self.final_movement_penalty
        
        # Linear decay from initial to final penalty
        progress = self.current_episode / self.penalty_decay_episodes
        penalty = self.initial_movement_penalty * (1 - progress) + self.final_movement_penalty * progress
        
        return penalty
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment for a new episode."""
        self.current_episode += 1
        # Reset episode tracking
        self.previous_loss = None
        self.best_loss = float('inf')
        self.steps_without_improvement = 0
        self.episode_steps = 0
        self.motor_movement_history.clear()
        
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
            if self.reset_calibration == 0:
                logger.info("Calibrating motors to center position")
                calibrate_full_sequence(self.motor_controller, self.manual_calibration)
            elif self.reset_calibration == 1:
                logger.info("Calibrating motors to random positions")
                calibrate_full_sequence_with_random(self.motor_controller, self.manual_calibration)
            elif self.reset_calibration == 2:
                logger.info("Skipping motor calibration - using current positions")
                # No calibration - motors stay at current positions
            else:
                logger.warning(f"Unknown reset_calibration value: {self.reset_calibration}. Skipping calibration.")

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
    
    def is_stagnant(self):
        """Check if training is stagnant based on recent loss history."""
        if len(self.loss_history) >= self.stagnation_window:
            recent_losses = list(self.loss_history)[-self.stagnation_window:]
            loss_std = np.std(recent_losses)
            return loss_std < self.stagnation_threshold
        return False
    
    def get_recommended_hold_bias(self):
        """Recommend hold bias based on current training state."""
        if self.is_stagnant():
            return 0.8  # Exploration mode
        return 1.7  # Normal mode
    
    def _move_motors_to_home(self):
        """Move all motors to their home positions."""
        if not self.motor_controller:
            return

        logger.info("Moving motors to home positions...")
        
        # Phase 1: Move to CW limits - all motors together
        logger.info("Phase 1: Moving to CW limits")
        motors_moving = set()
        movement_commands = {}
        
        # Start all motors (both ESP32s)
        for motor_num in range(1, self.num_motors + 1):
            self.motor_controller.set_speed(motor_num, self.motor_reset_speed)
            self.motor_controller.set_direction(motor_num, 1)  # 1 = CW
            self.motor_controller.move_steps(motor_num, 30000)
            motors_moving.add(motor_num)
            movement_commands[motor_num] = ('CW', 30000)
            time.sleep(5)
            logger.debug(f"Started CW homing for motor {motor_num}")

        # Wait for all motors
        completion_info = self._wait_for_motors_completion(
            motors_moving, movement_commands, 
            timeout=self.motor_completion_timeout,
            stabilization_time=self.stabilization_time
        )
        time.sleep(1.0)

        # Phase 2: Center from CW limits - all motors together
        center_offset = 4000
        logger.info(f"Phase 2: Centering from CW limits ({center_offset} steps CCW)")
        motors_moving = set()
        movement_commands = {}
        
        # Start all motors centering
        for motor_num in range(1, self.num_motors + 1):
            self.motor_controller.set_speed(motor_num, self.motor_reset_speed)
            self.motor_controller.set_direction(motor_num, 0)  # 0 = CCW
            self.motor_controller.move_steps(motor_num, center_offset)
            motors_moving.add(motor_num)
            movement_commands[motor_num] = ('CCW', center_offset)
            time.sleep(5)
            logger.debug(f"Started centering for motor {motor_num}")

        # Wait for all motors to complete centering
        completion_info = self._wait_for_motors_completion(
            motors_moving,
            movement_commands,
            timeout=100.0,
            stabilization_time=1.0
        )
        
        # Update all positions
        for motor_num in range(1, self.num_motors + 1):
            self.motor_positions[motor_num - 1] = -center_offset
            logger.info(f"Motor {motor_num} centered at position {-center_offset}")

        logger.info("Motors homed successfully")
    
    def step(self, actions):

        self.episode_steps += 1
        
        if len(actions) != self.num_motors:
            raise ValueError(f"Expected {self.num_motors} actions, got {len(actions)}")
        
        motors_moving = set()
        movement_commands = {}
        total_limit_penalty = 0
        
        for i, action in enumerate(actions):
            motor_num = i + 1
            current_pos = self.motor_positions[i]

            soft_penalty = self.limit_handler.get_limit_penalty(i, current_pos)
            if soft_penalty > 0:
                total_limit_penalty += soft_penalty * 0.1  # Scale down
                if soft_penalty > 1.0:  # Only log significant penalties
                    logger.debug(f"Motor {motor_num} soft penalty: {soft_penalty:.2f} at pos {current_pos}")
                
            if action == 1:  # HOLD
                continue
                
            elif action == 0:  # CCW
                # Check if movement would exceed limit
                proposed_pos = current_pos - self.motor_steps
                
                if proposed_pos <= -self.max_ccw_steps[i]:
                    # BLOCK ENTIRELY - treat as HOLD
                    logger.warning(f"Motor {motor_num} blocked at CCW limit (pos: {current_pos})")
                    self.limit_violations[i] += 1
                    total_limit_penalty += self.limit_penalty
                    # Don't add to movement_commands - motor stays still
                    
                else:
                    # Full movement allowed
                    movement_commands[motor_num] = ('CCW', self.motor_steps)
                    # self.motor_positions[i] -= self.motor_steps # Keeping this for now, remove if nothing breaks
                    motors_moving.add(motor_num)
                    
            elif action == 2:  # CW
                # Check if movement would exceed limit
                proposed_pos = current_pos + self.motor_steps
                
                if proposed_pos > self.max_cw_steps[i]:
                    # BLOCK ENTIRELY - treat as HOLD
                    logger.warning(f"Motor {motor_num} blocked at CW limit (pos: {current_pos} → {proposed_pos}, limit: {self.max_cw_steps[i]})")
                    self.limit_violations[i] += 1
                    total_limit_penalty += self.limit_penalty
                    # Don't add to movement_commands - motor stays still
                    
                else:
                    # Full movement allowed
                    movement_commands[motor_num] = ('CW', self.motor_steps)
                    # self.motor_positions[i] += self.motor_steps # Keeping this for now, remove if nothing breaks
                    motors_moving.add(motor_num)
        
        # Execute motor movements (only motors not blocked by limits)
        if self.use_motors and self.motor_controller and motors_moving:
            logger.debug(f"Moving motors: {sorted(motors_moving)}")
            
            # Simple execution - all movements are full motor_steps
            for motor_num, (direction, steps) in movement_commands.items():
                self.motor_controller.set_speed(motor_num, self.motor_speed)
                self.motor_controller.set_direction(motor_num, 0 if direction == 'CCW' else 1)
                self.motor_controller.move_steps(motor_num, steps)
                logger.debug(f"Motor {motor_num}: {direction} {steps} steps")
            
            # Wait for completion
            completion_info = self._wait_for_motors_completion(
                motors_moving, movement_commands, timeout=30.0, stabilization_time=self.step_wait_time
            )

            # NOW update positions (after movement confirmed)
            for motor_num, (direction, steps) in movement_commands.items():
                motor_idx = motor_num - 1
                if direction == 'CCW':
                    self.motor_positions[motor_idx] -= steps
                else:  # CW
                    self.motor_positions[motor_idx] += steps
                logger.debug(f"Motor {motor_num} position updated: {self.motor_positions[motor_idx]}")
        else:
            time.sleep(self.step_wait_time)
        
        # Get observation and calculate reward
        observation = self._get_observation()
        current_loss = float(observation[0])
        self.loss_history.append(current_loss)
        
        # Calculate reward with penalties
        reward = self._calculate_sophisticated_reward(current_loss, len(motors_moving))
        # reward -= total_limit_penalty
        
        # Check termination
        terminated = False
        if self.use_early_stopping and current_loss < self.early_stopping_threshold:
            terminated = True
            logger.info(f"Early stopping: loss {current_loss:.4f} < threshold {self.early_stopping_threshold}")
        
        # Check reward threshold if configured
        if self.reward_threshold_for_early_stop is not None:
            if reward >= self.reward_threshold_for_early_stop:
                terminated = True
                logger.info(f"Early stopping: reward {reward:.2f} >= threshold {self.reward_threshold_for_early_stop}")
        
        # Check truncation with config
        truncated = False
        if self.use_truncation and self.episode_steps >= self.min_episode_steps:
            if self.steps_without_improvement >= self.max_steps_without_improvement:
                truncated = True
        
        # Update tracking
        self.previous_loss = current_loss
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1
        
        # Info dict
        info = {
            'spectral_loss': current_loss,
            'best_loss': self.best_loss,
            'steps_without_improvement': self.steps_without_improvement,
            'motors_moved': list(motors_moving),
            'motor_positions': self.motor_positions.copy(),
            'limit_violations': self.limit_violations.copy(),
            'episode_steps': self.episode_steps,
            'is_stagnant': self.is_stagnant(),
            'recommended_hold_bias': self.get_recommended_hold_bias()
        }
        
        return observation, reward, terminated, truncated, info

    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (loss value).
        
        Returns:
            np.ndarray: Flattened array of observation values
            
        Raises:
            RuntimeError: If loss processor fails to provide observation
        """
        try:
            observation = self.loss_processor.get_observation()
            return observation.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Failed to get observation: {e}")
            raise RuntimeError("Failed to get observation") from e
    
    def _execute_motor_movements(self, movement_commands):
        """Execute motor movements - all are full motor_steps."""
        for motor_num, (direction, steps) in movement_commands.items():
            self.motor_controller.set_speed(motor_num, self.motor_speed)
            self.motor_controller.set_direction(motor_num, 0 if direction == 'CCW' else 1)
            self.motor_controller.move_steps(motor_num, self.motor_steps)
    
    def _wait_for_motors_completion(self, motors_moving, movement_commands, timeout=30.0, stabilization_time=2.0):
        """
        Wait for motors to complete movement with improved tracking and verification.
        """
        if not self.motor_controller:
            return
        
        # logger.info(f"Phase 1: Waiting for motors {sorted(motors_moving)} to complete movement")
        
        self.movement_commands = movement_commands

        # Clear any pending responses first
        self.motor_controller.get_responses(clear=True)
        time.sleep(0.1)  # Brief pause to ensure clean slate
        
        # Track completion by global motor number
        motors_pending = set(motors_moving)
        start_time = time.time()
        movement_completion_time = None
        stallguard_triggered = {}
        
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

                    if "STALL DETECTED" in response or "emergency stop" in response.lower():
                        # Extract motor number
                        for word in response.split():
                            if word.isdigit():
                                motor_num = int(word)
                                if motor_num in motors_pending:
                                    motor_idx = motor_num - 1
                                    direction= self.movement_commands[motor_num][0]
                                    
                                    if direction == 'CW':
                                        logger.info(f"✓ Motor {motor_num} hit StallGuard at CW limit")
                                        # StallGuard at CW limit - we KNOW position is 5000
                                        old_pos = self.motor_positions[motor_idx]
                                        drift_corrected = self.max_cw_steps[motor_idx] - old_pos

                                        if drift_corrected <= self.motor_steps:
                                            # Only correct for drift if motor is within the motor's step size (to avoid false calibration)
                                            self.motor_positions[motor_idx] = self.max_cw_steps[motor_idx]
                                            logger.info(f"  Position recalibrated: {old_pos} → {self.max_cw_steps[motor_idx]}")
                                        
                                            if abs(drift_corrected) > 50:
                                                logger.warning(f"  Significant drift corrected: {drift_corrected:.0f} steps")
                                    
                                    stallguard_triggered[motor_num] = True
                                    motors_pending.discard(motor_num)
                                    break
                    
                    # Try to parse MOTOR_COMPLETE format
                    elif "MOTOR_COMPLETE:" in response:
                        try:
                            internal_motor = int(response.split(":")[1].strip())
                            completed_motor = map_internal_to_global_motor(esp_num, internal_motor)
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
                                    completed_motor = map_internal_to_global_motor(esp_num, internal_motor)
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
    
    
    def _calculate_sophisticated_reward(self, current_loss, motors_moving_count=0):
        """
        Calculate reward with configurable components.
        
        Args:
            current_loss: Current spectral loss value
            motors_moving_count: Number of motors moving this step
            
        Returns:
            reward: Calculated reward with enabled components
        """

        reward_scale = self.reward_scale
        
        # 1. Base reward (exponential shaping)
        # base_reward = -np.log(current_loss + 1.0)
        # base_reward = (expected_loss - current_loss) * 2.0
        base_reward = self.target_loss - current_loss
         
        # 2. Improvement bonus
        improvement_bonus = 0
        if self.use_improvement_bonus and self.previous_loss is not None:
            improvement = self.previous_loss - current_loss
            self.improvement_history.append(improvement)
            improvement_bonus = improvement * self.improvement_bonus_weight
        
        # 3. Consistency bonus
        consistency_bonus = 0
        if self.use_consistency_bonus and len(self.improvement_history) > 0:
            avg_improvement = np.mean(list(self.improvement_history))
            if avg_improvement > 0 and improvement_bonus > 0:
                consistency_bonus = min(avg_improvement * self.consistency_bonus_weight, 1.0)
        
        # 4. Breakthrough bonus
        breakthrough_bonus = 0
        if self.use_breakthrough_bonus and self.episode_steps > self.episode_steps_before_breakthrough:
            if current_loss < self.best_loss:
                breakthrough_bonus = self.breakthrough_bonus_weight * (self.best_loss - current_loss)
                if np.isfinite(breakthrough_bonus):
                    logger.info(f"BREAKTHROUGH! New best loss: {current_loss:.4f} (bonus: {breakthrough_bonus:.2f})")
                else:
                    breakthrough_bonus = 0.0

        if self.episode_steps > 30:
            if current_loss < self.best_loss:
                breakthrough_bonus = 10.0 * (self.best_loss - current_loss)
                if not np.isfinite(breakthrough_bonus):
                    logger.warning(f"Invalid breakthrough_bonus: {breakthrough_bonus}")
                    breakthrough_bonus = 0.0
                else:
                    logger.info(f"BREAKTHROUGH! New best loss: {current_loss:.4f} (bonus: {breakthrough_bonus:.2f})")

        if self.previous_loss is not None:
            improvement = self.previous_loss - current_loss
            
            # Add to improvement history
            self.improvement_history.append(improvement)
            
            # Average recent improvements
            avg_improvement = np.mean(list(self.improvement_history))
            
            # Bonus for improvement, penalty for worsening (do we need a multiplier? Previously scaled by * 5)
            improvement_bonus = improvement
            
            # Extra bonus for consistent improvement
            if avg_improvement > 0 and improvement > 0:
                consistency_bonus = min(avg_improvement * 5.0, 1.0)
        
        # 5. Movement penalty
        movement_penalty = 0
        if self.use_movement_penalty and motors_moving_count > self.motors_for_movement_penalty:
            current_penalty_strength = self._get_adaptive_movement_penalty()
            excess_movements = motors_moving_count - self.motors_for_movement_penalty
            movement_penalty = -current_penalty_strength * excess_movements * self.movement_penalty_weight
        
        # 6. Stagnation penalty
        stagnation_penalty = 0
        if self.use_stagnation_penalty and len(self.loss_history) > 10:
            recent_std = np.std(list(self.loss_history)[-10:])
            if recent_std < 1e-4:
                stagnation_penalty = -0.5
        
        # 7. Efficiency bonus
        efficiency_bonus = 0
        if self.use_efficiency_bonus and improvement_bonus > 0:
            if motors_moving_count == 1:
                efficiency_bonus = self.efficiency_bonus_weight * self.efficiency_bonus_weight
            elif motors_moving_count == 2:
                efficiency_bonus = 0.3 * self.efficiency_bonus_weight
        
        # 8. Proximity bonus
        proximity_bonus = 0
        if self.use_proximity_bonus:
            if current_loss < self.proximity_threshold_very_close:
                proximity_bonus = (self.proximity_threshold_very_close - current_loss) * self.proximity_bonus_very_close
            elif current_loss < self.proximity_threshold_close:
                proximity_bonus = (self.proximity_threshold_close - current_loss) * self.proximity_bonus_close
        
        # FINAL REWARD CALCULATION - sum enabled components
        reward = (
            base_reward +
            improvement_bonus +
            consistency_bonus +
            breakthrough_bonus +
            movement_penalty +
            stagnation_penalty +
            efficiency_bonus +
            proximity_bonus
        )

        reward = np.clip(reward, -50, +100)
        
        # Log detailed breakdown occasionally (only if components are enabled)
        if np.random.random() < 0.02:
            components = [f"base={base_reward:.2f}"]
            if self.use_improvement_bonus and improvement_bonus != 0:
                components.append(f"improve={improvement_bonus:.2f}")
            if self.use_consistency_bonus and consistency_bonus != 0:
                components.append(f"consist={consistency_bonus:.2f}")
            if self.use_breakthrough_bonus and breakthrough_bonus != 0:
                components.append(f"breakthrough={breakthrough_bonus:.2f}")
            if self.use_movement_penalty and movement_penalty != 0:
                components.append(f"movement={movement_penalty:.2f}")
            if self.use_stagnation_penalty and stagnation_penalty != 0:
                components.append(f"stagnation={stagnation_penalty:.2f}")
            if self.use_efficiency_bonus and efficiency_bonus != 0:
                components.append(f"efficiency={efficiency_bonus:.2f}")
            if self.use_proximity_bonus and proximity_bonus != 0:
                components.append(f"proximity={proximity_bonus:.2f}")
            if self.log_motor_details and motors_moving:
                logger.debug(f"Motor movements: {movement_commands}")
                logger.debug(f"Motor positions after: {self.motor_positions}")
            
            logger.debug(f"Reward (ep {self.current_episode}): {' + '.join(components)} = {reward:.2f}")
        
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
        """Clean up resources and disable motor drivers."""
        if self.use_motors and self.motor_controller:
            logger.info("Emergency stop: disabling all motors")
            
            # Send stop commands
            self.motor_controller.stop_motor(0)
            time.sleep(0.2)
            
            # Disable all motor drivers (cuts power)
            try:
                self.motor_controller.send_command(0, "DISABLE", 1)
                logger.info("Sent DISABLE command to all motors")
                time.sleep(1.0)  # Wait for disable to take effect
            except Exception as e:
                logger.error(f"Failed to disable motors: {e}")
            
            # Individual stops as backup
            for motor_num in range(1, self.num_motors + 1):
                self.motor_controller.stop_motor(motor_num)
                time.sleep(0.05)
            
            # Final wait before disconnect
            time.sleep(1.0)
            
            self.motor_controller.disconnect()
            logger.info("Motors disabled and disconnected")
    
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