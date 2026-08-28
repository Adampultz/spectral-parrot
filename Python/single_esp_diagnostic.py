"""
Single-ESP32 isolation tool for driver and motor troubleshooting.

Connects to exactly one ESP32 (the other can stay unplugged/unpowered)
so faults can be isolated to a specific board, driver, or motor without
the other ESP32 in the loop. Uses internal motor numbers 1-4, matching
the firmware's own addressing (see Spec_Parrot_Motor_control_ESP32.ino).
"""

import argparse
import signal
import sys
import time
import threading
from collections import deque

import serial

from position_motors import map_internal_to_global_motor

controller = None


class SingleESP32Controller:
    def __init__(self, port, esp_num, baudrate=115200):
        self.port = port
        self.esp_num = esp_num  # 1 (odd motors) or 2 (even motors), for display only
        self.baudrate = baudrate
        self.ser = None
        self.connected = False
        self.responses = deque()
        self.listener_thread = None
        self.listening = False

    def connect(self):
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1,
                write_timeout=1
            )
            time.sleep(2)  # let the ESP32 finish its power-on reset
            self.connected = True
            print(f"Connected to ESP32 #{self.esp_num} on {self.port}")
        except Exception as e:
            print(f"Error connecting to ESP32 #{self.esp_num}: {e}")
            self.connected = False
            return False

        self.listening = True
        self.listener_thread = threading.Thread(target=self._listen, daemon=True)
        self.listener_thread.start()
        return True

    def disconnect(self):
        self.listening = False
        if self.listener_thread:
            self.listener_thread.join(timeout=1.0)
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"Disconnected from ESP32 #{self.esp_num} on {self.port}")
        self.connected = False

    def _listen(self):
        current_line = ""
        while self.listening and self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting:
                    byte = self.ser.read(1)
                    if byte:
                        try:
                            char = byte.decode('ascii')
                        except UnicodeDecodeError:
                            char = '?'
                        if char == '\n':
                            if current_line:
                                line = current_line.strip()
                                self.responses.append(line)
                                print(f"  {line}")
                            current_line = ""
                        else:
                            current_line += char
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"Listener error: {e}")
                time.sleep(0.1)

    def drain_responses(self):
        out = []
        while True:
            try:
                out.append(self.responses.popleft())
            except IndexError:
                return out

    def reset_and_capture(self, settle_time=3.0):
        """Toggle DTR to reset the ESP32; boot/driver-detection output prints live as it arrives."""
        if not self.ser:
            return
        self.drain_responses()
        print(f"\nResetting ESP32 #{self.esp_num}...")
        self.ser.dtr = False
        time.sleep(0.1)
        self.ser.dtr = True
        time.sleep(settle_time)

    def send(self, internal_motor, command_type, value):
        """internal_motor: 1-4, or 0 for all motors on this ESP32."""
        if not self.connected:
            print(f"ESP32 #{self.esp_num} not connected")
            return False
        try:
            command = f"MOTOR:{internal_motor} {command_type}:{value}\n"
            print(f"Sending: {command.strip()}")
            self.ser.write(command.encode('ascii'))
            return True
        except Exception as e:
            print(f"Error sending command: {e}")
            return False

    def stop_all(self):
        self.send(0, "STOP", 1)


def emergency_shutdown(signum=None, frame=None):
    global controller
    print("\n\nEMERGENCY SHUTDOWN SEQUENCE")
    if controller:
        try:
            print("Stopping all motors on this ESP32...")
            controller.stop_all()
            time.sleep(0.5)
            controller.disconnect()
        except Exception as e:
            print(f"Error during emergency shutdown: {e}")
    if signum is not None:
        sys.exit(0)


def listen_for_messages(ctrl, duration):
    print(f"\nWatching for {duration} seconds (messages print live as they arrive)...")
    ctrl.drain_responses()
    time.sleep(duration)
    count = len(ctrl.drain_responses())
    print(f"({count} lines received)")


def global_label(esp_num, internal_motor):
    g = map_internal_to_global_motor(esp_num, internal_motor)
    return f"internal motor {internal_motor} (global string {g})"


def interactive(ctrl, motor_speed):
    print("\nCommands:")
    print("  <motor 1-4> cw <steps>    - Move motor clockwise")
    print("  <motor 1-4> ccw <steps>   - Move motor counter-clockwise")
    print("  <motor 1-4> speed <val>   - Set motor speed")
    print("  <motor 1-4> sg <val>      - Set StallGuard threshold for this motor")
    print("  <motor 1-4> stop          - Stop motor")
    print("  all stop                 - Stop all motors on this ESP32")
    print("  reset                    - Reboot this ESP32 and show driver-detection output")
    print("  listen <seconds>         - Listen for any messages")
    print("  quit                     - Exit with safe shutdown\n")

    for motor in range(1, 5):
        ctrl.send(motor, "SPEED", motor_speed)
        time.sleep(0.1)

    while True:
        try:
            cmd = input("\nCommand: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if cmd in ("exit", "quit"):
            print("Performing safe shutdown sequence...")
            emergency_shutdown()
            break

        if cmd == "reset":
            ctrl.reset_and_capture()
            continue

        if cmd == "all stop":
            ctrl.stop_all()
            print("Stopped all motors on this ESP32")
            continue

        if cmd.startswith("listen"):
            parts = cmd.split()
            duration = 20.0
            if len(parts) > 1:
                try:
                    duration = float(parts[1])
                except ValueError:
                    pass
            listen_for_messages(ctrl, duration)
            continue

        parts = cmd.split()
        if len(parts) < 2:
            print("Invalid command format")
            continue

        try:
            motor = int(parts[0])
            if motor < 1 or motor > 4:
                print("Motor number must be 1-4 (this ESP32 only drives 4 motors)")
                continue

            action = parts[1]
            label = global_label(ctrl.esp_num, motor)

            if action == "stop":
                ctrl.send(motor, "STOP", 1)
                print(f"Stopped {label}")

            elif action == "cw":
                if len(parts) < 3:
                    print("Missing steps value")
                    continue
                steps = int(parts[2])
                ctrl.send(motor, "DIR", 1)
                ctrl.send(motor, "STEPS", steps)
                print(f"Moving {label} clockwise by {steps} steps")

            elif action == "ccw":
                if len(parts) < 3:
                    print("Missing steps value")
                    continue
                steps = int(parts[2])
                ctrl.send(motor, "DIR", 0)
                ctrl.send(motor, "STEPS", steps)
                print(f"Moving {label} counter-clockwise by {steps} steps")

            elif action == "speed":
                if len(parts) < 3:
                    print("Missing speed value")
                    continue
                speed = int(parts[2])
                if speed < 0:
                    print("Speed should be positive")
                    continue
                ctrl.send(motor, "SPEED", speed)
                print(f"Set {label} speed to {speed}")

            elif action == "sg":
                if len(parts) < 3:
                    print("Missing StallGuard threshold value")
                    continue
                threshold = int(parts[2])
                ctrl.send(motor, "SG", threshold)
                print(f"Set {label} StallGuard threshold to {threshold}")

            else:
                print(f"Unknown action: {action}")

        except ValueError:
            print("Invalid number format")
        except Exception as e:
            print(f"Error: {e}")


def main():
    global controller

    parser = argparse.ArgumentParser(
        description="Isolate a single ESP32 for driver and motor troubleshooting"
    )
    parser.add_argument("--port", type=str, required=True, help="Serial port for the ESP32 under test")
    parser.add_argument("--esp", type=int, required=True, choices=[1, 2],
                         help="Which ESP32 this is (1=odd motors 1/3/5/7, 2=even motors 2/4/6/8) — for global string labeling only")
    parser.add_argument("--baudrate", type=int, default=115200, help="Baud rate")
    parser.add_argument("--motor-speed", type=int, default=200, help="Initial speed set on all 4 motors")
    parser.add_argument("--skip-reset", action="store_true",
                         help="Don't toggle DTR on connect; use this if you want to reset the board "
                              "yourself (power cycle / EN button) instead of the software auto-reset")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, emergency_shutdown)
    signal.signal(signal.SIGTERM, emergency_shutdown)

    controller = SingleESP32Controller(args.port, args.esp, args.baudrate)
    if not controller.connect():
        print("Failed to connect.")
        return

    print(f"\n=== Single-ESP32 Diagnostic: ESP32 #{args.esp} ===")
    if args.skip_reset:
        print("Skipping auto-reset (--skip-reset). Power-cycle or press the board's "
              "EN/RESET button now, then use 'listen 10' to watch for boot output, "
              "or 'reset' to fall back to the software DTR toggle.")
    else:
        controller.reset_and_capture()

    try:
        interactive(controller, args.motor_speed)
    finally:
        emergency_shutdown()


if __name__ == "__main__":
    main()
