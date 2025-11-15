"""
String Change Manager for Spectral Parrot
Handles pausing training and manual motor control for string replacement
with precise position tracking and restoration
"""

import logging
import time
import threading
from typing import Optional, Dict, Tuple, List, Set
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class TrainingState(Enum):
    """Training states for the system"""
    RUNNING = "running"
    PAUSED = "paused"
    STRING_CHANGE = "string_change"
    RESUMING = "resuming"

class MotorPositionTracker:
    """Tracks and restores motor positions during string changes"""
    
    def __init__(self, num_motors: int = 8):
        self.num_motors = num_motors
        self.saved_positions: Dict[int, int] = {}  # motor_num -> original position in steps
        self.current_positions: Dict[int, int] = {}  # motor_num -> current position during string change
        self.total_slacken_steps: Dict[int, int] = {}   # motor_num -> total steps moved CCW for slackening
        self.slacken_history: Dict[int, List[int]] = {}  # motor_num -> list of individual slacken movements
        self.original_speeds: Dict[int, int] = {}  # motor_num -> original speed setting
        
    def save_position(self, motor_num: int, position: int, speed: int = 200):
        """Save motor position before string change"""
        self.saved_positions[motor_num] = position
        self.current_positions[motor_num] = position
        self.original_speeds[motor_num] = speed
        self.total_slacken_steps[motor_num] = 0
        self.slacken_history[motor_num] = []
        logger.info(f"Motor {motor_num}: Saved original position {position} steps")
        
    def record_slacken_movement(self, motor_num: int, steps: int):
        """Record an incremental slackening movement"""
        if motor_num not in self.total_slacken_steps:
            self.total_slacken_steps[motor_num] = 0
            self.slacken_history[motor_num] = []
        
        self.total_slacken_steps[motor_num] += steps
        self.slacken_history[motor_num].append(steps)
        
        if motor_num in self.current_positions:
            self.current_positions[motor_num] -= steps
            
        logger.info(f"Motor {motor_num}: Slackened {steps} steps CCW (total: {self.total_slacken_steps[motor_num]})")
        
    def get_restoration_steps(self, motor_num: int) -> Optional[int]:
        """Get total number of steps needed to restore position"""
        if motor_num in self.total_slacken_steps:
            return self.total_slacken_steps[motor_num]
        return None
    
    def get_slacken_history(self, motor_num: int) -> List[int]:
        """Get history of slackening movements for a motor"""
        return self.slacken_history.get(motor_num, [])
        
    def verify_position(self, motor_num: int, current_position: int) -> bool:
        """Verify if motor has returned to original position"""
        if motor_num not in self.saved_positions:
            return False
        
        expected = self.saved_positions[motor_num]
        difference = abs(current_position - expected)
        
        if difference == 0:
            logger.info(f"Motor {motor_num}: Position perfectly restored to {expected}")
            return True
        elif difference <= 2:  # Allow tiny tolerance for mechanical play
            logger.warning(f"Motor {motor_num}: Position restored with minor deviation "
                         f"(expected: {expected}, actual: {current_position})")
            return True
        else:
            logger.error(f"Motor {motor_num}: Position restoration failed! "
                        f"Expected: {expected}, Actual: {current_position}, "
                        f"Difference: {difference} steps")
            return False
    
    def clear(self, motor_num: Optional[int] = None):
        if motor_num is not None:
            self.saved_positions.pop(motor_num, None)
            self.current_positions.pop(motor_num, None)
            self.total_slacken_steps.pop(motor_num, None)
            self.slacken_history.pop(motor_num, None)
            self.original_speeds.pop(motor_num, None)
        else:
            self.saved_positions.clear()
            self.current_positions.clear()
            self.total_slacken_steps.clear()          
            self.slacken_history.clear()
            self.original_speeds.clear()

class StringChangeManager:
    """
    Manages string changes during training sessions.
    Allows pausing training, manual motor control, and safe string replacement
    with precise position tracking and restoration.
    """
    
    def __init__(self, motor_controller=None, environment=None, 
                 loss_processor=None, agent=None):
        """
        Initialize the string change manager.
        
        Args:
            motor_controller: DualESP32StepperController instance
            environment: MotorEnvironment instance
            loss_processor: SimpleLossProcessor instance  
            agent: MotorPPOAgent instance
        """
        self.motor_controller = motor_controller
        self.environment = environment
        self.loss_processor = loss_processor
        self.agent = agent
        
        self.state = TrainingState.RUNNING
        self.position_tracker = MotorPositionTracker()
        self.active_string_changes: Set[int] = set()  # Motors currently being serviced
        
        # Safety parameters
        self.slacken_steps_default = 500  # Default steps to slacken
        self.slacken_speed = 200          # Slower speed for controlled slackening
        self.restore_speed = 200          # Moderate speed for position restoration
        self.verification_pause = 0.5      # Pause after restoration for settling
        
        # Control flags
        self.training_paused = False
        self.audio_active = True  # Keep audio running during string changes
        
        logger.info("StringChangeManager initialized with position tracking")
    
    def pause_training(self, reason: str = "Manual pause") -> bool:
        """
        Pause training while keeping audio active.
        
        Args:
            reason: Reason for pausing
            
        Returns:
            bool: Success status
        """
        if self.state != TrainingState.RUNNING:
            logger.warning(f"Cannot pause - current state: {self.state}")
            return False
        
        self.state = TrainingState.PAUSED
        self.training_paused = True
        
        # Don't stop the loss processor - keep audio feedback active
        logger.info(f"Training paused: {reason}")
        logger.info("Audio feedback remains active")
        
        # Store current episode state if needed
        if self.agent is not None:
            # Could save agent state here if needed for safety
            pass
        
        return True
    
    def prepare_string_change(self, motor_num: int, slacken_steps: Optional[int] = None) -> bool:
        """
        Prepare a motor for string change by saving position (first time) or additional slackening.
        Can be called multiple times for incremental slackening.
        
        Args:
            motor_num: Motor number (1-8)
            slacken_steps: Steps to slacken (uses default if None)
            
        Returns:
            bool: Success status
        """
        if self.state not in [TrainingState.PAUSED, TrainingState.STRING_CHANGE]:
            logger.error("Must pause training before string changes")
            return False
        
        if not self.motor_controller:
            logger.error("No motor controller available")
            return False
        
        motor_idx = motor_num - 1
        
        # Check if this is the first slackening for this motor
        is_first_slacken = motor_num not in self.active_string_changes
        
        if self.environment:
            current_pos = self.environment.motor_positions[motor_idx]
            max_ccw = self.environment.max_ccw_steps[motor_idx]
            
            if is_first_slacken:
                # First time - save original position
                self.position_tracker.save_position(
                    motor_num, 
                    current_pos,
                    self.environment.motor_speed
                )
                logger.info(f"Motor {motor_num}: Beginning string change procedure")
            else:
                logger.info(f"Motor {motor_num}: Additional slackening requested")
            
            # Calculate safe slacken amount
            slacken = slacken_steps or self.slacken_steps_default
            safe_slacken = min(slacken, current_pos + max_ccw - 100)  # Leave safety margin
            
            if safe_slacken <= 0:
                logger.error(f"Motor {motor_num} already at CCW limit, cannot slacken further")
                return False
            
            logger.info(f"  Current position: {current_pos}")
            logger.info(f"  Will slacken by: {safe_slacken} steps CCW")
            
            # Show total slackening if this is additional
            if not is_first_slacken:
                total_so_far = self.position_tracker.total_slacken_steps.get(motor_num, 0)
                logger.info(f"  Total slackening after this: {total_so_far + safe_slacken} steps")
            
        else:
            # Fallback if no environment
            safe_slacken = slacken_steps or self.slacken_steps_default
            if is_first_slacken:
                self.position_tracker.save_position(motor_num, 0, 200)
        
        # Execute slackening movement
        self.state = TrainingState.STRING_CHANGE
        self.active_string_changes.add(motor_num)
        
        try:
            # Set slower speed for safety
            self.motor_controller.set_speed(motor_num, self.slacken_speed)
            
            # Move CCW to slacken
            self.motor_controller.set_direction(motor_num, 0)  # 0 = CCW
            self.motor_controller.move_steps(motor_num, safe_slacken)
            
            # Record the movement (accumulates if multiple slackens)
            self.position_tracker.record_slacken_movement(motor_num, safe_slacken)
            
            # Wait for completion
            logger.info(f"Motor {motor_num}: Waiting for slackening to complete...")
            time.sleep(safe_slacken / self.slacken_speed + 2)  # Rough estimate + buffer
            
            # Update environment position if available
            if self.environment:
                self.environment.motor_positions[motor_idx] -= safe_slacken
                logger.info(f"Motor {motor_num}: Position after slackening: "
                          f"{self.environment.motor_positions[motor_idx]}")
            
            # Show status
            total_slackened = self.position_tracker.total_slacken_steps.get(motor_num, 0)
            original_pos = self.position_tracker.saved_positions.get(motor_num, 0)
            
            logger.info(f"Motor {motor_num}: Ready for string replacement")
            logger.info(f"  Original position: {original_pos}")
            logger.info(f"  Total slackened: {total_slackened} steps")
            logger.info(f"  Current position: {original_pos - total_slackened}")
            
            if is_first_slacken:
                logger.info(f"*** You can now replace the string on Motor {motor_num} ***")
                logger.info(f"*** Or press 'm' to slacken more if needed ***")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to slacken motor {motor_num}: {e}")
            if is_first_slacken:
                self.active_string_changes.discard(motor_num)
            return False
    
    def slacken_motor_incremental(self, motor_num: int, steps: int) -> bool:
        """
        Incrementally slacken a motor by a specific number of steps.
        This is a convenience method that calls prepare_string_change.
        
        Args:
            motor_num: Motor number (1-8)
            steps: Number of steps to slacken
            
        Returns:
            bool: Success status
        """
        return self.prepare_string_change(motor_num, steps)
    
    def restore_motor_position(self, motor_num: int) -> bool:
        """
        Restore motor to its original position after string change.
        
        Args:
            motor_num: Motor number (1-8)
            
        Returns:
            bool: Success status (True if position restored correctly)
        """
        if motor_num not in self.active_string_changes:
            logger.warning(f"Motor {motor_num} not in active string change list")
            return False
        
        restoration_steps = self.position_tracker.get_restoration_steps(motor_num)
        if restoration_steps is None:
            logger.error(f"No restoration data for motor {motor_num}")
            return False
        
        logger.info(f"Motor {motor_num}: Beginning position restoration")
        logger.info(f"  Will move CW by {restoration_steps} steps")
        
        try:
            # Set restoration speed
            self.motor_controller.set_speed(motor_num, self.restore_speed)
            
            # Move CW to restore position
            self.motor_controller.set_direction(motor_num, 1)  # 1 = CW
            self.motor_controller.move_steps(motor_num, restoration_steps)
            
            # Wait for completion
            logger.info(f"Motor {motor_num}: Waiting for restoration to complete...")
            time.sleep(restoration_steps / self.restore_speed + 2)  # Rough estimate + buffer
            
            # Update environment position
            motor_idx = motor_num - 1
            if self.environment:
                self.environment.motor_positions[motor_idx] += restoration_steps
                
                # Verify position
                current_pos = self.environment.motor_positions[motor_idx]
                if self.position_tracker.verify_position(motor_num, current_pos):
                    logger.info(f"✓ Motor {motor_num}: Position successfully restored")
                    
                    # Restore original speed
                    original_speed = self.position_tracker.original_speeds.get(
                        motor_num, self.environment.motor_speed
                    )
                    self.motor_controller.set_speed(motor_num, original_speed)
                    
                    # Clean up
                    self.active_string_changes.discard(motor_num)
                    self.position_tracker.clear(motor_num)
                    
                    return True
                else:
                    logger.error(f"✗ Motor {motor_num}: Position verification failed!")
                    return False
            else:
                # No environment, assume success
                self.active_string_changes.discard(motor_num)
                self.position_tracker.clear(motor_num)
                return True
                
        except Exception as e:
            logger.error(f"Failed to restore motor {motor_num} position: {e}")
            return False
    
    def complete_all_string_changes(self) -> bool:
        """
        Complete all active string changes and restore positions.
        
        Returns:
            bool: True if all motors restored successfully
        """
        if not self.active_string_changes:
            logger.info("No active string changes to complete")
            return True
        
        logger.info(f"Completing string changes for motors: {sorted(self.active_string_changes)}")
        
        all_success = True
        failed_motors = []
        
        for motor_num in list(self.active_string_changes):
            if not self.restore_motor_position(motor_num):
                all_success = False
                failed_motors.append(motor_num)
        
        if all_success:
            logger.info("✓ All motors successfully restored to original positions")
        else:
            logger.error(f"✗ Failed to restore motors: {failed_motors}")
            logger.error("Manual intervention may be required to recalibrate")
        
        return all_success
    
    def resume_training(self) -> bool:
        """
        Resume training after string changes.
        
        Returns:
            bool: Success status
        """
        # Ensure all string changes are complete
        if self.active_string_changes:
            logger.warning("Completing remaining string changes before resuming...")
            if not self.complete_all_string_changes():
                logger.error("Cannot resume - position restoration failed")
                return False
        
        # Verify environment state
        if self.environment:
            logger.info("Current motor positions:")
            for i in range(self.environment.num_motors):
                logger.info(f"  Motor {i+1}: {self.environment.motor_positions[i]} steps")
        
        self.state = TrainingState.RESUMING
        time.sleep(self.verification_pause)  # Allow settling
        
        self.state = TrainingState.RUNNING
        self.training_paused = False
        
        logger.info("Training resumed successfully")
        logger.info("All motor positions have been preserved")
        
        return True
    
    def emergency_stop(self):
        """Emergency stop all motors and pause system"""
        logger.warning("EMERGENCY STOP initiated")
        
        if self.motor_controller:
            try:
                # Stop all motors immediately
                for motor in range(1, 9):
                    self.motor_controller.stop_motor(motor)
                logger.info("All motors stopped")
            except Exception as e:
                logger.error(f"Error during emergency stop: {e}")
        
        self.state = TrainingState.PAUSED
        self.training_paused = True
        
    def get_status(self) -> Dict:
        """Get current status of string change manager"""
        return {
            'state': self.state.value,
            'training_paused': self.training_paused,
            'audio_active': self.audio_active,
            'active_string_changes': list(self.active_string_changes),
            'saved_positions': dict(self.position_tracker.saved_positions),
            'slacken_steps': dict(self.position_tracker.total_slacken_steps),
            'current_positions': dict(self.position_tracker.current_positions),
            'slacken_history': dict(self.position_tracker.slacken_history)
        }