"""
Interactive Control for String Changes
Provides keyboard interface for pausing training and managing string changes
"""

import logging
import threading
import time
from typing import Optional, Callable
import sys
import select
import tty
import termios

logger = logging.getLogger(__name__)

# Thread lock for coordinated terminal output
_terminal_lock = threading.Lock()

class InteractiveController:
    """
    Interactive keyboard controller for managing string changes during training.
    """
    
    def __init__(self, string_change_manager=None):
        """
        Initialize the interactive controller.
        
        Args:
            string_change_manager: StringChangeManager instance
        """
        self.string_change_manager = string_change_manager
        self.listener_thread = None
        self.stop_listener = threading.Event()
        self.command_handlers = {}
            
        # State for multi-key commands
        self.input_mode = None  # 'motor_select', 'steps_input', etc.
        self.input_buffer = ""
        self.selected_motor = None
        
        # Register default commands
        self._register_default_commands()
        
        # Terminal settings for non-blocking input
        self.old_settings = None
        self.terminal_lock = _terminal_lock
        
    def _safe_print(self, *args, **kwargs):
        """
        Print with proper terminal handling.
        Temporarily restores terminal settings, prints, then restores raw mode.
        Thread-safe with lock.
        """
        with self.terminal_lock:
            try:
                # Restore normal terminal mode if in raw mode
                if self.old_settings:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                
                # Print normally with explicit flush
                print(*args, **kwargs, flush=True)
                
                # Return to raw mode if listener is active
                if not self.stop_listener.is_set() and self.old_settings:
                    tty.setcbreak(sys.stdin.fileno())
            except Exception as e:
                logger.error(f"Error in safe_print: {e}")
        
    def _register_default_commands(self):
        """Register default keyboard commands"""
        self.command_handlers = {
            'p': self._handle_pause,
            'r': self._handle_resume,
            's': self._handle_string_change,
            'm': self._handle_manual_slacken,  # Manual slacken with custom steps
            'a': self._handle_additional_slacken,  # Additional slackening
            'c': self._handle_complete_changes,
            'e': self._handle_emergency_stop,
            'h': self._show_help,
            'q': self._handle_status,
            '1': lambda: self._handle_motor_string_change(1),
            '2': lambda: self._handle_motor_string_change(2),
            '3': lambda: self._handle_motor_string_change(3),
            '4': lambda: self._handle_motor_string_change(4),
            '5': lambda: self._handle_motor_string_change(5),
            '6': lambda: self._handle_motor_string_change(6),
            '7': lambda: self._handle_motor_string_change(7),
            '8': lambda: self._handle_motor_string_change(8),
        }
    
    def start(self):
        """Start the keyboard listener thread."""
        if not self.listener_thread or not self.listener_thread.is_alive():
            self.stop_listener.clear()
            # Log BEFORE starting thread to avoid formatting issues
            logger.info("Interactive controller started")
            logger.info("Keyboard listener active - press 'h' for help")
            
            self.listener_thread = threading.Thread(
                target=self._keyboard_listener,
                daemon=True
            )
            self.listener_thread.start()
    
    def stop(self):
        """Stop the interactive controller"""
        self.stop_listener.set()
        if self.listener_thread:
            self.listener_thread.join(timeout=1.0)
        
        # Restore terminal settings
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        
        logger.info("Interactive controller stopped")
    
    def _keyboard_listener(self):
        """Background thread for keyboard input."""
        try:
            # Small delay to ensure any final log flushes complete
            time.sleep(0.2)
            
            # Save terminal settings
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            
            while not self.stop_listener.is_set():
                # Check for available input with timeout
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    
                    # Handle input
                    if char.lower() == 'h':
                        # Show help with proper terminal handling
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                        self._show_help()
                        tty.setcbreak(sys.stdin.fileno())
                    elif char.lower() == 'p':
                        self._handle_pause()
                    elif char.lower() == 'r':
                        self._handle_resume()
                    elif char.lower() == 's':
                        self._handle_string_change()
                    elif char.lower() == 'm':
                        self._handle_manual_slacken()
                    elif char.lower() == 'a':
                        self._handle_additional_slacken()
                    elif char.lower() == 'c':
                        self._handle_complete_changes()
                    elif char.lower() == 'e':
                        self._handle_emergency_stop()
                    elif char.lower() == 'q':
                        self._handle_status()
                    elif char in '12345678':
                        self._handle_motor_string_change(int(char))
                    elif char == '\x03':  # Ctrl+C
                        logger.info("Ctrl+C detected, initiating shutdown...")
                        break
                    elif char == '\x1b':  # ESC
                        if self.input_mode:
                            print("\nCancelled")
                            self.input_mode = None
                            self.input_buffer = ""
                            self.selected_motor = None
    
        except Exception as e:
            logger.error(f"Keyboard listener error: {e}")
            raise
        finally:
            # Restore terminal settings
            if self.old_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def _handle_steps_input(self, char):
        """Handle numeric input for step count"""
        if char == '\r' or char == '\n':  # Enter key
            if self.input_buffer and self.input_buffer.isdigit():
                steps = int(self.input_buffer)
                motor = self.selected_motor
                
                self._safe_print(f"Slackening motor {motor} by {steps} steps...")
                
                # Reset input state
                self.input_mode = None
                self.input_buffer = ""
                
                # Execute slackening
                if self.string_change_manager:
                    success = self.string_change_manager.slacken_motor_incremental(motor, steps)
                    if success:
                        self._safe_print(f"Motor {motor} slackened by {steps} steps")
                        self._safe_print("Press 'a' to slacken more, or 'c' to complete")
                    else:
                        self._safe_print(f"Failed to slacken motor {motor}")
            else:
                self._safe_print("Invalid input. Cancelled.")
                self.input_mode = None
                self.input_buffer = ""
        elif char == '\x1b':  # ESC
            self._safe_print("Cancelled")
            self.input_mode = None
            self.input_buffer = ""
            self.selected_motor = None
        elif char == '\x7f' or char == '\b':  # Backspace
            if self.input_buffer:
                self.input_buffer = self.input_buffer[:-1]
                self._safe_print(f"\rSteps to slacken: {self.input_buffer}", end='')
        elif char.isdigit():
            self.input_buffer += char
            self._safe_print(f"\rSteps to slacken: {self.input_buffer}", end='')
    
    def _handle_motor_select_for_slacken(self, char):
        """Handle motor selection for manual slackening"""
        if char in '12345678':
            motor_num = int(char)
            self.selected_motor = motor_num
            self.input_mode = 'steps_input'
            self.input_buffer = ""
            self._safe_print(f"Motor {motor_num} selected")
            self._safe_print("Enter number of steps to slacken: ", end='')
        elif char == '\x1b':  # ESC
            self._safe_print("Cancelled")
            self.input_mode = None
            self.selected_motor = None
        else:
            self._safe_print("Invalid motor number. Press 1-8 or ESC to cancel")
    
    def _handle_pause(self):
        """Handle pause command"""
        if self.string_change_manager:
            logger.info("Pause command received")
            if self.string_change_manager.state.value != "paused":
                success = self.string_change_manager.pause_training("User request via keyboard")
                if success:
                    self._safe_print("")
                    self._safe_print("=== TRAINING PAUSED ===")
                    self._safe_print("Audio feedback remains active")
                    self._safe_print("Press 's' to start string change procedure")
                    self._safe_print("Press '1-8' to change specific motor string")
                    self._safe_print("Press 'r' to resume training")
                    logger.info("Training paused successfully")
                else:
                    self._safe_print("Cannot pause at this time")
                    logger.warning("Failed to pause training")
            else:
                self._safe_print("Training is already paused")
                logger.info("Training already paused")
        else:
            self._safe_print("No string change manager available")
            logger.error("String change manager not available")
    
    def _handle_resume(self):
        """Handle resume command"""
        if self.string_change_manager:
            self._safe_print("")
            self._safe_print("=== RESUMING TRAINING ===")
            success = self.string_change_manager.resume_training()
            if success:
                self._safe_print("Training resumed successfully")
                self._safe_print("All motor positions preserved")
            else:
                self._safe_print("Failed to resume - check motor positions")
        else:
            self._safe_print("No string change manager available")
    
    def _handle_string_change(self):
        """Handle generic string change command"""
        self._safe_print("")
        self._safe_print("=== STRING CHANGE MODE ===")
        self._safe_print("Which motor needs a string change?")
        self._safe_print("Press 1-8 for the motor number, or ESC to cancel")
    
    def _handle_manual_slacken(self):
        """Handle manual slackening with custom step count"""
        self._safe_print("")
        self._safe_print("=== MANUAL SLACKENING ===")
        self._safe_print("Which motor to slacken?")
        self._safe_print("Press 1-8 for motor number, or ESC to cancel")
        self.input_mode = 'motor_select_for_slacken'
    
    def _handle_additional_slacken(self):
        """Handle additional slackening for active string changes"""
        if self.string_change_manager:
            status = self.string_change_manager.get_status()
            active_changes = status['active_string_changes']
            
            if not active_changes:
                self._safe_print("No active string changes. Use 'm' to start manual slackening.")
                return
            
            if len(active_changes) == 1:
                # Only one motor active, slacken it
                motor = active_changes[0]
                self.selected_motor = motor
                self.input_mode = 'steps_input'
                self.input_buffer = ""
                self._safe_print(f"Additional slackening for motor {motor}")
                self._safe_print("Enter number of steps to slacken: ", end='')
            else:
                # Multiple motors active, need to select
                self._safe_print(f"Active string changes: {sorted(active_changes)}")
                self._safe_print("Which motor to slacken further?")
                self.input_mode = 'motor_select_for_slacken'
    
    def _handle_motor_string_change(self, motor_num: int):
        """Handle string change for specific motor (default slackening)"""
        if self.string_change_manager:
            self._safe_print(f"=== STRING CHANGE FOR MOTOR {motor_num} ===")
            
            # First ensure we're paused
            if self.string_change_manager.state.value != "paused":
                self.string_change_manager.pause_training("String change requested")
            
            # Start string change procedure with default slackening
            self._safe_print(f"Slackening motor {motor_num} with default steps...")
            success = self.string_change_manager.prepare_string_change(motor_num)
            
            if success:
                status = self.string_change_manager.get_status()
                total_slackened = status.get('slacken_steps', {}).get(motor_num, 0)
                
                self._safe_print(f"*** MOTOR {motor_num} SLACKENED BY {total_slackened} STEPS ***")
                self._safe_print("Options:")
                self._safe_print("  'a' - Slacken more if needed") 
                self._safe_print("  'c' - String replaced, restore position")
                self._safe_print("WARNING: Motor MUST be restored before resuming!")
            else:
                self._safe_print(f"Failed to prepare motor {motor_num} for string change")
        else:
            self._safe_print("No string change manager available")
    
    def _handle_complete_changes(self):
        """Handle completion of string changes"""
        if self.string_change_manager:
            self._safe_print("")
            self._safe_print("=== COMPLETING STRING CHANGES ===")
            self._safe_print("Restoring motor positions...")
            
            success = self.string_change_manager.complete_all_string_changes()
            
            if success:
                self._safe_print("OK - All motors restored to original positions")
                self._safe_print("Ready to resume training (press 'r')")
            else:
                self._safe_print("ERROR - Some motors failed to restore!")
                self._safe_print("Manual intervention required")
        else:
            self._safe_print("No string change manager available")
    
    def _handle_emergency_stop(self):
        """Handle emergency stop"""
        if self.string_change_manager:
            self._safe_print("")
            self._safe_print("!!! EMERGENCY STOP !!!")
            self.string_change_manager.emergency_stop()
            self._safe_print("All motors stopped")
            self._safe_print("System paused")
        else:
            self._safe_print("No string change manager available")
    
    def _handle_status(self):
        """Show current status"""
        if self.string_change_manager:
            status = self.string_change_manager.get_status()
            self._safe_print("")
            self._safe_print("=== SYSTEM STATUS ===")
            self._safe_print(f"State: {status['state']}")
            self._safe_print(f"Training paused: {status['training_paused']}")
            self._safe_print(f"Audio active: {status['audio_active']}")
            
            if status['active_string_changes']:
                self._safe_print(f"Active string changes: {status['active_string_changes']}")
                
                # Show slackening details for each active motor
                for motor in status['active_string_changes']:
                    if motor in status.get('slacken_steps', {}):
                        total = status['slacken_steps'][motor]
                        original = status.get('saved_positions', {}).get(motor, 'unknown')
                        current = original - total if isinstance(original, int) else 'unknown'
                        
                        self._safe_print(f"Motor {motor}:")
                        self._safe_print(f"  Original position: {original} steps")
                        self._safe_print(f"  Total slackened: {total} steps")
                        self._safe_print(f"  Current position: {current} steps")
                        
                        # Show slacken history if available
                        if hasattr(self.string_change_manager.position_tracker, 'slacken_history'):
                            history = self.string_change_manager.position_tracker.slacken_history.get(motor, [])
                            if history:
                                self._safe_print(f"  Slacken history: {history}")
            
            elif status['saved_positions']:
                self._safe_print("No active changes, but saved positions exist:")
                for motor, pos in status['saved_positions'].items():
                    self._safe_print(f"  Motor {motor}: {pos} steps")
        else:
            self._safe_print("No string change manager available")
    
    def _show_help(self):
        """Show help with proper terminal handling."""
        # Temporarily restore terminal for multi-line output
        with self.terminal_lock:
            try:
                if self.old_settings:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                
                print("\033[2J\033[H", flush=True)  # Clear screen and move to top
                print("=== INTERACTIVE CONTROL HELP ===", flush=True)
                print("\nKeyboard commands:", flush=True)
                print("  p     - Pause training", flush=True)
                print("  r     - Resume training", flush=True)
                print("  s     - Start string change procedure", flush=True)
                print("  m     - Manual slacken (specify motor and steps)", flush=True)
                print("  a     - Additional slacken (for active string change)", flush=True)
                print("  1-8   - Quick string change (default slackening)", flush=True)
                print("  c     - Complete string changes (restore positions)", flush=True)
                print("  e     - Emergency stop", flush=True)
                print("  q     - Show status", flush=True)
                print("  h     - Show this help", flush=True)
                print("  ESC   - Cancel current operation", flush=True)
                print("  Ctrl+C - Exit program", flush=True)
                
                print("\nString Change Procedures:", flush=True)
                
                print("\n1. QUICK CHANGE (default slackening):", flush=True)
                print("   - Press 'p' to pause", flush=True)
                print("   - Press motor number (1-8)", flush=True)
                print("   - Replace string", flush=True)
                print("   - Press 'c' to restore", flush=True)
                print("   - Press 'r' to resume", flush=True)
                
                print("\n2. CUSTOM SLACKENING:", flush=True)
                print("   - Press 'p' to pause", flush=True)
                print("   - Press 'm' for manual slacken", flush=True)
                print("   - Select motor (1-8)", flush=True)
                print("   - Enter step count", flush=True)
                print("   - Replace string or press 'a' to slacken more", flush=True)
                print("   - Press 'c' to restore", flush=True)
                print("   - Press 'r' to resume", flush=True)
                
                print("\n3. INCREMENTAL SLACKENING:", flush=True)
                print("   - After initial slackening", flush=True)
                print("   - Press 'a' to slacken more", flush=True)
                print("   - Enter additional steps", flush=True)
                print("   - Repeat as needed", flush=True)
                print("   - Press 'c' when string is replaced", flush=True)
                
                print("\nIMPORTANT:", flush=True)
                print("- Motors MUST be restored ('c') before resuming", flush=True)
                print("- System tracks total slackening automatically", flush=True)
                print("- Original position is always preserved", flush=True)
                print("=" * 40, flush=True)
                
                # Wait for user acknowledgment
                print("\nPress Enter to continue...", flush=True)
                input()  # Wait for Enter key
                
                # Return to raw mode if listener is active
                if not self.stop_listener.is_set() and self.old_settings:
                    tty.setcbreak(sys.stdin.fileno())
            except Exception as e:
                logger.error(f"Error in _show_help: {e}")

def create_training_wrapper(train_function, config):
    """
    Wrapper to add string change capability to existing training function.
    
    Args:
        train_function: Original training function
        config: Training configuration
        
    Returns:
        Modified training function with string change support
    """
    def wrapped_train(*args, **kwargs):
        # Import inside function to avoid circular imports
        from string_change_manager import StringChangeManager
        
        # Extract components from kwargs or create placeholders
        motor_controller = kwargs.get('motor_controller')
        environment = kwargs.get('environment')
        loss_processor = kwargs.get('loss_processor')
        agent = kwargs.get('agent')
        
        # Create string change manager
        string_manager = StringChangeManager(
            motor_controller=motor_controller,
            environment=environment,
            loss_processor=loss_processor,
            agent=agent
        )
        
        # Create interactive controller
        controller = InteractiveController(string_manager)
        controller.start()
        
        # Add to kwargs for access in training loop
        kwargs['string_change_manager'] = string_manager
        kwargs['interactive_controller'] = controller
        
        try:
            # Run original training
            result = train_function(*args, **kwargs)
            return result
        finally:
            # Clean up
            controller.stop()
    
    return wrapped_train