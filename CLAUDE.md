# CLAUDE.md — Spectral Parrot

## Project Overview

Spectral Parrot is a deep reinforcement learning system that autonomously tunes a motorised 8-string feedback instrument. The RL agent observes a multi-scale spectral loss (MSSL) between a target audio spectrum and the instrument's live pickup signal, and outputs motor commands to tune the strings toward the target. The system operates in real hardware time: every action involves physical motor movement and acoustic settling before the next observation is valid.

This is not a simulation. Code changes can cause string breakage, motor stalls, or training instability. Understand the physical consequences of any change before proposing it.

---

## System Architecture

### Languages and Responsibilities

| Layer | Language | Role |
|---|---|---|
| Machine Learning | Python | PPO agent, environment, reward, spectral loss |
| Audio | SuperCollider | Feedback routing, pickup summing, audio playback |
| Motor Control | C (Arduino/ESP32) | TMC2209 driver commands, StallGuard detection |

### Python Module Map

| File | Purpose |
|---|---|
| `main_motor_training.py` | Top-level training loop, CLI entry point |
| `motor_environment.py` | Gymnasium environment: step, reset, reward, position tracking |
| `motor_ppo_agent.py` | PPO agent: actor/critic update, GAE, memory buffer |
| `motor_actor_network.py` | Actor and critic neural network definitions |
| `simple_spectral_loss_processor.py` | Receives MSSL values from SuperCollider via OSC |
| `stft_audio.py` | Multi-scale STFT computation (scales: 512, 1024, 2048, 4096) |
| `Stepper_Control.py` | Serial communication with both ESP32s |
| `osc_handler.py` | OSC message routing between Python and SuperCollider |
| `position_motors.py` | Motor calibration routines (CW stall → random CCW offset) |
| `string_change_manager.py` | Manages string-change state across episodes |
| `string_change_interactive_control.py` | Interactive terminal UI for manual motor control |
| `save_session_hyperparameters.py` | Persists config snapshot per training session |
| `main_MSSL.py` | Standalone MSSL computation (outside training loop) |
| `cpu_monitor.py` | System resource monitoring during training |
| `config.json` | All hyperparameters and hardware settings (single source of truth) |

### Hardware Layout

- **8 stepper motors** — one per string, driven by TMC2209 drivers
- **ESP32 #1** — controls motors 1, 3, 5, 7 (odd)
- **ESP32 #2** — controls motors 2, 4, 6, 8 (even)
- **Pickups** — Cycfi Nu Series electromagnetic pickups, one per string
- **Audio interface** — RME Fireface UFX III at 48 kHz
- **Contact exciters** — mounted on instrument body, receive SuperCollider output to sustain feedback
- **Power** — 9V rechargeable batteries (avoid mains noise)

### Signal Flow

```
Strings → Electromagnetic Pickups → RME Fireface UFX III
       → SuperCollider (summing + routing)
       → Contact Exciters → Strings  [feedback loop]

SuperCollider → OSC → Python (MSSL calculation)
Python (PPO agent) → Serial → ESP32s → TMC2209 → Motors → String tension
```

---

## Running the System

### Motor Calibration (run first)
```bash
python position_motors.py --port1 /dev/cu.usbserial-0001 --port2 /dev/cu.usbserial-2
# Then type: cal limits
# Motors drive CW to mechanical stall, establishing position zero.
# Tighten strings manually, then optionally: cal center
```

### Training
```bash
python main_motor_training.py \
  --input-device <device_number> \
  --port1 /dev/cu.usbserial-0001 \
  --port2 /dev/cu.usbserial-2
```

Use `--list-devices` to enumerate audio input devices.  
Use `--skip-calibration` only for short test runs or debugging; position tracking will drift without calibration.

SuperCollider (`Spectral_Parrot_SuperCollider.scd`) must be running before training starts.

---

## Hyperparameters

All hyperparameters live in `config.json`. Do not hardcode values in source files. Key parameters to understand before making changes:

### Motor / Physical

| Parameter | Description | Current |
|---|---|---|
| `max_cw_steps` | Per-motor CW travel limit (steps) | 4500 all |
| `max_ccw_steps` | Per-motor CCW travel limit (steps) | 4000 / 4500 |
| `step_wait_time` | Seconds to wait after motor move (acoustic settling) | 0.5 |
| `reset_wait_time` | Wait after episode reset | 2.0 |
| `motor_speed` | Steps/sec | 200 |
| `motor_random_calibration_range` | Max CCW offset from CW limit on reset | 3000 |
| `stallguard_threshold` | TMC2209 StallGuard sensitivity | 150 |

### PPO / Learning

| Parameter | Description | Current |
|---|---|---|
| `lr_actor` | Actor learning rate | 0.0001 |
| `lr_critic` | Critic learning rate | 0.0003 |
| `update_interval` | Steps between PPO updates | 512 |
| `max_ep_length` | Max steps per episode | 1024 |
| `hold_bias` | Logit bias toward HOLD action | 3.0 |
| `gamma` | Discount factor | 0.98 |
| `gae_lambda` | GAE lambda | 0.95 |
| `clip_param` | PPO clipping epsilon | 0.2 |
| `entropy_coef` | Entropy bonus coefficient | 0.01 |
| `target_loss` | MSSL value considered good performance | 7.0 |

### Reward Components (toggleable)

| Flag | Component |
|---|---|
| `use_improvement_bonus` | Reward for reducing loss |
| `use_consistency_bonus` | Reward for sustained low loss |
| `use_breakthrough_bonus` | Reward for new session-minimum loss |
| `use_movement_penalty` | Penalise excessive simultaneous motor movement |
| `use_stagnation_penalty` | Penalise no loss change over a window |

---

## Critical Constraints

### Physical Safety
- **Never move all 8 motors simultaneously.** 4–6 simultaneous movements create chaotic acoustic interactions that prevent learning. The `motors_for_movement_penalty` parameter and `hold_bias` work together to limit this.
- **Position tracking must remain accurate.** Drift accumulates across a session. Calibration at episode start (`reset_calibration: 1`) is the primary defence. Do not disable it casually.
- **CW limits are hard mechanical stops.** StallGuard detects these. CCW limits are software-enforced via `max_ccw_steps` with a `ccw_safety_margin` buffer. Exceeding either risks string breakage.
- **`step_wait_time` must not be reduced without acoustic testing.** The instrument needs time to settle before the next spectral reading is valid. Current minimum is 0.5 s; the original design assumed 1.0 s.

### StallGuard
- StallGuard is reliable for CW (tightening) limit detection.
- It is **not** reliable for short moves or during acceleration. False triggers are possible.
- `stallguard_warnings_before_stop` controls tolerance before a motor is considered stalled.
- StallGuard timing relative to move commands is sensitive; do not change serial communication timing without testing.

### Observation Space
- State dimension is 10: `[normalised_loss, motor_pos_0, ..., motor_pos_7, loss_direction]`
- The loss is normalised to [0, 1] by dividing by `loss_clip_max` (observation only — reward and termination logic use raw loss units, e.g. `target_loss`).
- Motor positions are normalised against their per-motor travel range to [-1, 1].
- `loss_direction` is the sign of the target-vs-instrument spectral energy difference: -1, 0, or +1.
- The Markov property is only approximately satisfied. Identical loss values can occur at different motor configurations.

### Reward Design
- Simple linear rewards based on deviation from `target_loss` outperform complex multi-component structures.
- The logarithmic component in MSSL can amplify floating-point differences; treat log-domain losses carefully.
- Volume-invariant normalisation is intentional — spectral shape, not amplitude, is what matters.

---

## Common Failure Modes

| Symptom | Likely Cause |
|---|---|
| Critic loss explodes (>5000) | Learning rate too high, or reward scale mismatch |
| All motors hold indefinitely | `hold_bias` too high, or entropy collapsed |
| Loss stays near 23–25 | Motors not moving at all, or calibration failed |
| Stall triggers on short moves | StallGuard threshold too low, or move too brief |
| Position tracking drifts | Calibration skipped, or serial packet loss |
| Loss oscillates without converging | Too many motors moving simultaneously |

---

## Code Conventions

- All configuration comes from `config.json` via the `TrainingConfig` dataclass in `config.py`. Do not introduce new hardcoded constants; add them to `config.json`.
- OSC is used for all Python ↔ SuperCollider communication. Do not introduce direct audio processing in Python where SuperCollider already handles it.
- Serial messages to ESP32s follow the protocol defined in `Spec_Parrot_Motor_control_ESP32.ino`. Changes to the Python serial layer must be matched in the Arduino firmware.
- Motor indices are 0-based internally. The mapping to global motor numbers (1-based, odd/even ESP32 split) is handled by `map_internal_to_global_motor` in `position_motors.py`.
- Checkpoints are saved to `./checkpoints/`, logs to `./logs/`, plots to `./results/`. These paths are configurable in `config.json`.
- Training sessions are identified by timestamp. Session hyperparameters are saved by `save_session_hyperparameters.py` at the start of each run.

---

## What Not to Change Without Discussion

- The MSSL calculation pipeline (`stft_audio.py`, `simple_spectral_loss_processor.py`) — changes here affect the fundamental reward signal.
- The calibration sequence in `position_motors.py` — this is the safety anchor for all position tracking.
- StallGuard thresholds and timing in both the Python serial layer and the ESP32 firmware simultaneously — they are tightly coupled.
- The observation normalisation scheme — the agent's learned policy is sensitive to input scaling.
