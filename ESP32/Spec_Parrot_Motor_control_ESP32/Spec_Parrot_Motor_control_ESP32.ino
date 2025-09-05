              #include <FlexyStepper.h>
              #include <TMCStepper.h>

              // Define number of motors
              #define NUM_MOTORS 4

              // Define pins for all motors
              const int DIR_PINS[NUM_MOTORS] = {14, 26, 33, 21};
              const int STEP_PINS[NUM_MOTORS] = {12, 27, 25, 19}; 

              // All motors share the same UART port
              #define SERIAL_PORT      Serial2  // HardwareSerial port pins 16 & 17
              #define RX_PIN           16      // UART RX pin
              #define TX_PIN           17      // UART TX pin

              // TMC2209 Driver addresses
              const uint8_t DRIVER_ADDRESSES[NUM_MOTORS] = {0b00, 0b01, 0b10, 0b11};

              #define R_SENSE 0.11f // Match to your driver

              // Motor phase tracking
              enum MotorPhase {
                PHASE_IDLE,
                PHASE_ACCELERATING,
                PHASE_CONSTANT_SPEED,
                PHASE_DECELERATING
              };

              // Motor status and settings
              bool shaft[NUM_MOTORS] = {false, false, false, false};
              bool isMoving[NUM_MOTORS] = {false, false, false, false};
              int motorSpeed[NUM_MOTORS] = {2, 2, 2, 2};
              int motorDirection[NUM_MOTORS] = {1, 1, 1, 1};
              int stepsToMove[NUM_MOTORS] = {600, 600, 600, 600};
              int mSteps[NUM_MOTORS] = {256, 256, 256, 256};
              int accelSpeed[NUM_MOTORS] = {400, 400, 400, 400};
              uint16_t mAmps[NUM_MOTORS] = {1200, 1200, 1200, 1200}; // Set to 1.2A (1200mA) based on our previous discussion
              long currentPosition[NUM_MOTORS] = {0, 0, 0, 0};
              long targetPosition[NUM_MOTORS] = {0, 0, 0, 0};
              bool movementCompleteSent[NUM_MOTORS] = {false, false, false, false};
              MotorPhase motorPhase[NUM_MOTORS] = {PHASE_IDLE, PHASE_IDLE, PHASE_IDLE, PHASE_IDLE};
              float maxVelocityReached[NUM_MOTORS] = {0, 0, 0, 0};
              long lastPosition[NUM_MOTORS] = {0, 0, 0, 0};
              float lastVelocity[NUM_MOTORS] = {0, 0, 0, 0};
              uint32_t phaseStartTime[NUM_MOTORS] = {0, 0, 0, 0};
              bool stallGuardEnabled[NUM_MOTORS] = {false, false, false, false};

              const int varianceSize = 10;
              const int variance_threshold = 1000;
              const int accelerationTime = 2000;
              const int sG_numWarnings = 5;

              const float VELOCITY_THRESHOLD = 0.95;  // Consider constant speed if velocity > 95% of max
              const float DECEL_THRESHOLD = 0.90;     // Start decel detection when velocity < 90% of max
              const int DECEL_POSITION_THRESHOLD = 200; // Steps from target to consider decelerating


              // Add basic StallGuard variables - only these are new
              uint16_t stallGuardResult[NUM_MOTORS] = {0, 0, 0, 0};
              int sgThreshold = 150;  // Default StallGuard threshold (0-255)

              // Create stepper and driver objects - all using the same SERIAL_PORT but different addresses
              TMC2209Stepper* drivers[NUM_MOTORS];
              FlexyStepper steppers[NUM_MOTORS];

              void setup() {
                // Initialize Serial for command input
                Serial.begin(115200);
                while (!Serial) {
                  delay(10);
                }
                delay(500);
                Serial.println(F("Serial Initialized"));

                // Initialize UART for all TMC2209 drivers
                SERIAL_PORT.begin(115200, SERIAL_8N1, RX_PIN, TX_PIN);

                // Initialize all GPIO pins first
                for (int i = 0; i < NUM_MOTORS; i++) {
                  pinMode(STEP_PINS[i], OUTPUT);
                  pinMode(DIR_PINS[i], OUTPUT);
                  digitalWrite(STEP_PINS[i], LOW);
                  digitalWrite(DIR_PINS[i], LOW);
                }
                
                // Create and initialize all drivers
                for (int i = 0; i < NUM_MOTORS; i++) {
                  drivers[i] = new TMC2209Stepper(&SERIAL_PORT, R_SENSE, DRIVER_ADDRESSES[i]);
                  
                  // Connect stepper to pins
                  steppers[i].connectToPins(STEP_PINS[i], DIR_PINS[i]);
                  
                  // Configure the TMC2209 driver
                  drivers[i]->begin();
                  drivers[i]->toff(5);
                  drivers[i]->shaft(shaft[i]);
                  drivers[i]->rms_current(1200);  
                  drivers[i]->microsteps(mSteps[i]);
                  drivers[i]->pwm_autoscale(true);
                  drivers[i]->en_spreadCycle(false);  // false = StealthChop / true = SpreadCycle

                  // Add this single line to configure StallGuard
                  drivers[i]->SGTHRS(sgThreshold);
                  
                  // Configure the stepper
                  steppers[i].setCurrentPositionInSteps(0);
                  steppers[i].setSpeedInStepsPerSecond(motorSpeed[i] * mSteps[i]);
                  steppers[i].setAccelerationInStepsPerSecondPerSecond(accelSpeed[i] * mSteps[i]);
                  
                  // Debug output
                  Serial.print("Motor ");
                  Serial.print(i + 1);
                  Serial.print(" initialized. Address: 0b");
                  Serial.print(DRIVER_ADDRESSES[i], BIN);
                  Serial.print(", STEP Pin: ");
                  Serial.print(STEP_PINS[i]);
                  Serial.print(", DIR Pin: ");
                  Serial.println(DIR_PINS[i]);
                  
                  // Test driver communication
                  uint8_t version = drivers[i]->version();
                  Serial.print("  Driver version: 0x");
                  Serial.print(version, HEX);
                  if (version == 0x21) {
                    Serial.println(" (TMC2209 detected)");
                  } else if (version == 0) {
                    Serial.println(" (NO RESPONSE - check wiring and MS pins)");
                  } else {
                    Serial.println(" (unknown driver type)");
                  }
                  
                  delay(100); // Small delay between driver initializations
                }

              delay(500);  // Give all drivers time to settle
              Serial.println("Verifying driver configurations:");

              for (int i = 0; i < NUM_MOTORS; i++) {
                uint8_t version = drivers[i]->version();
                uint8_t sgthrs = drivers[i]->SGTHRS();
                uint8_t toff = drivers[i]->toff();
                
                Serial.print("Motor ");
                Serial.print(i + 1);
                Serial.print(" - Version: 0x");
                Serial.print(version, HEX);
                Serial.print(", SGTHRS: ");
                Serial.print(sgthrs);
                Serial.print(", TOFF: ");
                Serial.println(toff);
              }
                
                // Test direct pin control
                Serial.println("Testing direct pin control...");
                for (int i = 0; i < NUM_MOTORS; i++) {
                  Serial.print("Testing motor ");
                  Serial.print(i + 1);
                  Serial.print(" (STEP pin ");
                  Serial.print(STEP_PINS[i]);
                  Serial.println(")");
                  
                  // Set DIR pin
                  digitalWrite(DIR_PINS[i], HIGH);
                  
                  // Generate 10 test pulses
                  for (int p = 0; p < 10; p++) {
                    digitalWrite(STEP_PINS[i], HIGH);
                    delayMicroseconds(500);
                    digitalWrite(STEP_PINS[i], LOW);
                    delayMicroseconds(500);
                  }
                  
                  delay(250);
                }
                
                Serial.println("Ready for commands (MOTOR:num SPEED:value, MOTOR:num DIR:value, MOTOR:num STEPS:value)");
                Serial.println("Added StallGuard commands: MOTOR:num SG:value - Set StallGuard threshold");
                Serial.println("Use MOTOR:0 to control all motors at once");
              }

              // New simple function to set StallGuard threshold
              void setStallGuardThreshold(int motorIndex, int threshold) {
                if (threshold < 0) threshold = 0;
                if (threshold > 255) threshold = 255;
                
                if (motorIndex < NUM_MOTORS) {
                  drivers[motorIndex]->SGTHRS(threshold);
                  Serial.print("Motor ");
                  Serial.print(motorIndex + 1);
                  Serial.print(" StallGuard threshold set to: ");
                  Serial.println(threshold);
                } else {
                  // Set for all motors
                  for (int i = 0; i < NUM_MOTORS; i++) {
                    drivers[i]->SGTHRS(threshold);
                  }
                  Serial.print("All motors StallGuard threshold set to: ");
                  Serial.println(threshold);
                  sgThreshold = threshold;
                }
              }

void printMotorPhases() {
    Serial.println("\n========== Motor Phase Status ==========");
    for (int i = 0; i < NUM_MOTORS; i++) {
        Serial.print("Motor ");
        Serial.print(i + 1);
        Serial.print(": ");
        
        // Print phase
        switch(motorPhase[i]) {
            case PHASE_IDLE:
                Serial.print("IDLE          ");
                break;
            case PHASE_ACCELERATING:
                Serial.print("ACCELERATING  ");
                break;
            case PHASE_CONSTANT_SPEED:
                Serial.print("CONSTANT      ");
                break;
            case PHASE_DECELERATING:
                Serial.print("DECELERATING  ");
                break;
        }
        
        // Print StallGuard status
        Serial.print(" | StallGuard: ");
        Serial.print(stallGuardEnabled[i] ? "ON " : "OFF");
        
        // Print current SG value if moving
        if (isMoving[i]) {
            Serial.print(" | SG Value: ");
            Serial.print(drivers[i]->SG_RESULT());
            Serial.print(" | Velocity: ");
            Serial.print(abs(steppers[i].getCurrentVelocityInStepsPerSecond()), 0);
            Serial.print(" | Distance: ");
            // Serial.print(abs(steppers[i].getDistanceToTargetSigned()));
        }
        
        Serial.println();
    }
    Serial.println("=========================================");
}

// Function to manually set StallGuard state (for testing)
void setStallGuardState(int motorIndex, bool enabled) {
    if (motorIndex >= 0 && motorIndex < NUM_MOTORS) {
        stallGuardEnabled[motorIndex] = enabled;
        Serial.print("Motor ");
        Serial.print(motorIndex + 1);
        Serial.print(" StallGuard manually set to: ");
        Serial.println(enabled ? "ON" : "OFF");
    }
}

void checkStallGuard() {
    static uint32_t lastCheck = 0;
    if (millis() - lastCheck < 50) {
        return;
    }
    lastCheck = millis();
    
    // Variance calculation arrays
    static uint16_t sgHistory[NUM_MOTORS][varianceSize] = {0};
    static uint8_t historyIndex[NUM_MOTORS] = {0};
    static uint8_t warningCount[NUM_MOTORS] = {0};
    
    // FIX: Add a flag to track if history buffer is valid
    static bool historyValid[NUM_MOTORS] = {false, false, false, false};
    static uint8_t historySamples[NUM_MOTORS] = {0};
    
    for (int i = 0; i < NUM_MOTORS; i++) {
        if (isMoving[i]) {
            // Get current motor state
            long currentPosition = steppers[i].getCurrentPositionInSteps();
            float currentVelocity = abs(steppers[i].getCurrentVelocityInStepsPerSecond());
            long distanceToTarget = abs(targetPosition[i] - currentPosition);
            
            // ===== PHASE DETECTION LOGIC =====
            
            if (motorPhase[i] == PHASE_IDLE) {
                // Just started moving
                motorPhase[i] = PHASE_ACCELERATING;
                phaseStartTime[i] = millis();
                maxVelocityReached[i] = 0;
                stallGuardEnabled[i] = false;
                
                // FIX: Reset history validity when starting new movement
                historyValid[i] = false;
                historySamples[i] = 0;
                historyIndex[i] = 0;
                warningCount[i] = 0;  // Also reset warning count
                
                // Serial.print("Motor ");
                // Serial.print(i + 1);
                // Serial.println(" started - ACCELERATING (StallGuard OFF)");
            }
            else if (motorPhase[i] == PHASE_ACCELERATING) {
                // Track maximum velocity reached
                if (currentVelocity > maxVelocityReached[i]) {
                    maxVelocityReached[i] = currentVelocity;
                }
                
                // Check if we've reached constant speed
                float targetVelocity = motorSpeed[i] * mSteps[i];
                bool velocityReached = currentVelocity >= (VELOCITY_THRESHOLD * targetVelocity);
                bool timeElapsed = (millis() - phaseStartTime[i]) > accelerationTime;
                
                if (velocityReached || timeElapsed) {
                    motorPhase[i] = PHASE_CONSTANT_SPEED;
                    phaseStartTime[i] = millis();
                    stallGuardEnabled[i] = true;
                    
                    // FIX: Pre-fill history buffer with current StallGuard value
                    uint16_t currentSG = drivers[i]->SG_RESULT();
                    for (int j = 0; j < varianceSize; j++) {
                        sgHistory[i][j] = currentSG;
                    }
                    historyIndex[i] = 0;
                    historySamples[i] = varianceSize;  // Mark buffer as full
                    historyValid[i] = false;  // But wait for a few real samples
                    
                    // Serial.print("Motor ");
                    // Serial.print(i + 1);
                    // Serial.print(" at CONSTANT SPEED: ");
                    // Serial.print(currentVelocity);
                    // Serial.print(" steps/s (StallGuard ON, pre-filled with SG=");
                    // Serial.print(currentSG);
                    // Serial.println(")");
                }
            }
            else if (motorPhase[i] == PHASE_CONSTANT_SPEED) {
                // Check for deceleration start
                bool velocityDropping = currentVelocity < (DECEL_THRESHOLD * maxVelocityReached[i]);
                
                // Calculate deceleration distance
                long decelDistance = 0;
                if (accelSpeed[i] > 0) {
                    decelDistance = (long)((currentVelocity * currentVelocity) / 
                                         (2.0 * accelSpeed[i] * mSteps[i]));
                }
                bool nearTarget = distanceToTarget <= (decelDistance * 1.3);
                
                if (velocityDropping || nearTarget) {
                    motorPhase[i] = PHASE_DECELERATING;
                    phaseStartTime[i] = millis();
                    stallGuardEnabled[i] = false;
                    
                    // Serial.print("Motor ");
                    // Serial.print(i + 1);
                    // Serial.print(" DECELERATING - ");
                    // Serial.print(distanceToTarget);
                    // Serial.println(" steps to target (StallGuard OFF)");
                }
            }
            else if (motorPhase[i] == PHASE_DECELERATING) {
                // Stay in deceleration until stopped
                // StallGuard remains disabled
            }
            
            // ===== STALLGUARD CHECKING (only if enabled) =====
            
            if (stallGuardEnabled[i]) {
                // Read StallGuard value
                stallGuardResult[i] = drivers[i]->SG_RESULT();
                
                // Store in history for averaging
                sgHistory[i][historyIndex[i]] = stallGuardResult[i];
                historyIndex[i] = (historyIndex[i] + 1) % varianceSize;
                
                // FIX: Track how many real samples we have after transition
                if (!historyValid[i]) {
                    historySamples[i]++;
                    // Wait for at least half the buffer to fill with real samples
                    if (historySamples[i] >= varianceSize / 2) {
                        historyValid[i] = true;
                        // Serial.print("Motor ");
                        // Serial.print(i + 1);
                        // Serial.println(" StallGuard history now valid");
                    }
                }
                
                // Calculate rolling average
                uint32_t sum = 0;
                for (int j = 0; j < varianceSize; j++) {
                    sum += sgHistory[i][j];
                }
                float mean = sum / (float)varianceSize;
                
                // Periodic debug output
                static uint32_t lastDebugOutput[NUM_MOTORS] = {0};
                // if (millis() - lastDebugOutput[i] > 500) {
                //     Serial.print("Motor ");
                //     Serial.print(i + 1);
                //     Serial.print(" SG: ");
                //     Serial.print(mean, 0);
                //     Serial.print(" (CONST-ON");
                //     if (!historyValid[i]) {
                //         Serial.print("-FILLING");
                //     }
                //     Serial.println(")");
                //     lastDebugOutput[i] = millis();
                // }
                
                // FIX: Only check for stalls if history buffer is valid
                if (historyValid[i]) {
                    if (mean < sgThreshold) {
                        warningCount[i]++;
                        
                        if (warningCount[i] >= sG_numWarnings) {
                            Serial.print("!!! Motor ");
                            Serial.print(i + 1);
                            Serial.print(" STALL DETECTED (SG: ");
                            Serial.print(mean, 0);
                            Serial.println(") - Emergency stop");
                            
                            if (!movementCompleteSent[i]) {
                                movementCompleteSent[i] = true;
                                Serial.print("MOTOR_COMPLETE:");
                                Serial.println(i + 1);
                            }
                            
                            stopMotor(i);
                            warningCount[i] = 0;
                        }
                    } else {
                        warningCount[i] = 0;
                    }
                } else {
                    // History not valid yet, don't check for stalls
                    warningCount[i] = 0;
                }
            } else {
                // StallGuard is disabled
                static uint32_t lastDisabledMsg[NUM_MOTORS] = {0};
                if (millis() - lastDisabledMsg[i] > 2000) {
                    Serial.print("Motor ");
                    Serial.print(i + 1);
                    Serial.print(" StallGuard DISABLED (Phase: ");
                    switch(motorPhase[i]) {
                        case PHASE_ACCELERATING: Serial.print("ACCELERATING"); break;
                        case PHASE_DECELERATING: Serial.print("DECELERATING"); break;
                        default: Serial.print("OTHER");
                    }
                    Serial.println(")");
                    lastDisabledMsg[i] = millis();
                }
                
                warningCount[i] = 0;
            }
            
            // Update last values
            lastPosition[i] = currentPosition;
            lastVelocity[i] = currentVelocity;
            
        } else {
            // Motor stopped - reset to idle
            if (motorPhase[i] != PHASE_IDLE) {
                motorPhase[i] = PHASE_IDLE;
                stallGuardEnabled[i] = true;
                historyValid[i] = false;  // FIX: Reset validity for next movement
                warningCount[i] = 0;
                
                Serial.print("Motor ");
                Serial.print(i + 1);
                Serial.println(" stopped - IDLE (StallGuard ready)");
            }
        }
    }
}

              // Function to set speed for a specific motor
              void setMotorSpeed(int motorIndex, int speed) {
                motorSpeed[motorIndex] = speed;
                steppers[motorIndex].setSpeedInStepsPerSecond(speed * mSteps[motorIndex]);
                
                // Serial.print("Motor ");
                // Serial.print(motorIndex + 1);
                // Serial.print(" speed set to: ");
                // Serial.println(speed);
              }

              // Function to set direction for a specific motor
              void setMotorDirection(int motorIndex, int direction) {
                motorDirection[motorIndex] = direction;
                
                // If we want to change direction while moving, we need to set a new target
                if (isMoving[motorIndex]) {
                  long currentPos = steppers[motorIndex].getCurrentPositionInSteps();
                  // Calculate remaining steps based on current position and target
                  long remainingSteps = abs(targetPosition[motorIndex] - currentPos);
                  
                  // Set new target in the opposite direction
                  if (direction == 0) {
                    targetPosition[motorIndex] = currentPos - remainingSteps;
                  } else {
                    targetPosition[motorIndex] = currentPos + remainingSteps;
                  }
                  
                  // Update the target position for the stepper
                  steppers[motorIndex].setTargetPositionInSteps(targetPosition[motorIndex]);
                }
                
                // Serial.print("Motor ");
                // Serial.print(motorIndex + 1);
                // Serial.print(" direction set to: ");
                // Serial.println(direction == 1 ? "Clockwise" : "Counter-clockwise");
              }

              // Function to move a specific motor
              void moveMotor(int motorIndex, int steps) {
                stepsToMove[motorIndex] = steps;

                movementCompleteSent[motorIndex] = false;
                
                // Get the current position directly from the stepper object
                currentPosition[motorIndex] = steppers[motorIndex].getCurrentPositionInSteps();
                
                // Calculate the target position based on direction and steps
                long relativeSteps = steps * mSteps[motorIndex];
                if (motorDirection[motorIndex] == 0) {
                  relativeSteps = -relativeSteps;
                }
                
                // Set the target position relative to current position
                targetPosition[motorIndex] = currentPosition[motorIndex] + relativeSteps;
                
                // Set the target position in the stepper object
                steppers[motorIndex].setTargetPositionInSteps(targetPosition[motorIndex]);
                isMoving[motorIndex] = true;
              }

              // Function to stop a specific motor
              void stopMotor(int motorIndex) {

                  // CRITICAL: Call setTargetPositionToStop() BEFORE setting isMoving to false
                  if (!steppers[motorIndex].motionComplete()) {
                    // if (isMoving[motorIndex]){
                      long currentPos = steppers[motorIndex].getCurrentPositionInSteps();
                      steppers[motorIndex].setTargetPositionToStop();
                      
                      // Process movement one more time to ensure the stop is registered
                      steppers[motorIndex].processMovement();

                      steppers[motorIndex].setTargetPositionInSteps(currentPos);
                      
                      // Now mark as not moving
                      isMoving[motorIndex] = false;
                      
                      // Reset the movement completed flag so it doesn't send duplicate messages
                      movementCompleteSent[motorIndex] = false;
                      
                      Serial.print("Motor ");
                      Serial.print(motorIndex + 1);
                      Serial.println(" emergency stop triggered");
                    }
                }

              void loop() {
                // Process movement for all steppers and check completion
                for (int i = 0; i < NUM_MOTORS; i++) {
                  if (isMoving[i]) {
                    // Process the movement
                    steppers[i].processMovement();

                if (steppers[i].motionComplete()) {
              // Only send completion message if not already sent
              if (!movementCompleteSent[i]) {
                  // First ensure the motor is actually stopped
                  steppers[i].setTargetPositionToStop();
                  stopMotor(i);
                  
                  // Mark as no longer moving
                  isMoving[i] = false;
                  
                  // Small delay to ensure motor has physically stopped
                  // delay(100);  // 100ms should be enough for deceleration
                  
                  // Double-check the motor isn't still processing movement
                  steppers[i].processMovement();  // One final process call
                  
                  // Now we can safely say it's complete
                  movementCompleteSent[i] = true;
                  
                  // Send ONLY the structured format (remove duplicate message)
                  Serial.print("MOTOR_COMPLETE:");
                  Serial.println(i + 1);  // Convert to 1-based index
              }
          }
                  } else {
                    // Still process movement even when not actively moving
                    // This ensures any pending commands get handled
                    steppers[i].processMovement();
                  }
                }
                
                // Check StallGuard values (just monitoring for now)
                checkStallGuard();
                
                // Check for serial commands
                if (Serial.available() > 0) {
                  // Read the incoming string
                  String incomingString = Serial.readStringUntil('\n');
                  
                  // Parse motor number first
                  int motorSeparatorIndex = incomingString.indexOf("MOTOR:");
                  int motorIndex = -1;
                  
                  if (motorSeparatorIndex != -1) {
                    int motorEndIndex = incomingString.indexOf(' ', motorSeparatorIndex);
                    if (motorEndIndex == -1) motorEndIndex = incomingString.length();
                    
                    String motorStr = incomingString.substring(motorSeparatorIndex + 6, motorEndIndex);
                    motorIndex = motorStr.toInt() - 1; // Convert to 0-based index
                    
                    // Check for control all motors
                    if (motorIndex == -1) {
                      motorIndex = NUM_MOTORS; // Special value to control all motors
                    }
                    
                    // Validate motor index
                    if (motorIndex < 0 || (motorIndex > NUM_MOTORS-1 && motorIndex != NUM_MOTORS)) {
                      Serial.println("Invalid motor number. Use 1-4 or 0 for all motors.");
                      return;
                    }
                    
                    // Find command after motor specification
                    incomingString = incomingString.substring(motorEndIndex + 1);
                  } else {
                    Serial.println("Motor number not specified. Use MOTOR:x before command.");
                    return;
                  }
                  
                  // Parse the command type and value
                  int separatorIndex = incomingString.indexOf(':');
                  
                  if (separatorIndex != -1) {
                    String commandType = incomingString.substring(0, separatorIndex);
                    String valueStr = incomingString.substring(separatorIndex + 1);
                    int value = valueStr.toInt();
                    
                    // Handle different command types
                    if (commandType == "SPEED") {
                      if (motorIndex == NUM_MOTORS) {
                        // Set speed for all motors
                        for (int i = 0; i < NUM_MOTORS; i++) {
                          setMotorSpeed(i, value);
                        }
                      } else {
                        setMotorSpeed(motorIndex, value);
                      }
                    }
                    else if (commandType == "DIR") {
                      if (motorIndex == NUM_MOTORS) {
                        // Set direction for all motors
                        for (int i = 0; i < NUM_MOTORS; i++) {
                          setMotorDirection(i, value);
                        }
                      } else {
                        setMotorDirection(motorIndex, value);
                      }
                    }
                    else if (commandType == "STEPS") {
                      if (motorIndex == NUM_MOTORS) {
                        // Move all motors
                        for (int i = 0; i < NUM_MOTORS; i++) {
                          moveMotor(i, value);
                        }
                      } else {
                        moveMotor(motorIndex, value);
                      }
                    }
                  else if (commandType == "STOP") {
                      if (motorIndex == NUM_MOTORS) {
                          // Stop all motors
                          for (int i = 0; i < NUM_MOTORS; i++) {

                              stopMotor(i);
                          }
                          Serial.println("All motors stopped");
                      } else {
                          stopMotor(motorIndex);
                      }
                  }
                    // Add new command handler for StallGuard threshold
                    else if (commandType == "SG") {
                      setStallGuardThreshold(motorIndex, value);
                    }
                    else if (commandType == "DISABLE") {
                      if (motorIndex == NUM_MOTORS) {
                        // Disable all motors
                        for (int i = 0; i < NUM_MOTORS; i++) {
                          drivers[i]->toff(0);  // Disable driver completely
                          isMoving[i] = false;
                        }
                        Serial.println("All motors disabled - drivers turned off");
                      } else {
                        // Disable specific motor
                        drivers[motorIndex]->toff(0);  // Disable driver completely
                        isMoving[motorIndex] = false;
                        Serial.print("Motor ");
                        Serial.print(motorIndex + 1);
                        Serial.println(" disabled - driver turned off");
                      }
                    }
                    else if (commandType == "ENABLE") {
                      if (motorIndex == NUM_MOTORS) {
                        // Enable all motors
                        for (int i = 0; i < NUM_MOTORS; i++) {
                          drivers[i]->toff(5);  // Re-enable driver with toff=5
                        }
                        Serial.println("All motors enabled - drivers turned on");
                      } else {
                        // Enable specific motor
                        drivers[motorIndex]->toff(5);  // Re-enable driver with toff=5
                        Serial.print("Motor ");
                        Serial.print(motorIndex + 1);
                        Serial.println(" enabled - driver turned on");
                      }
                    }
                    else if (commandType == "PHASES") {
                          // Print current phase status of all motors
                          printMotorPhases();
                      }
                      else if (commandType == "SGSTATE") {
                          // Manually control StallGuard state (for testing)
                          // Format: MOTOR:x SGSTATE:1 (1=on, 0=off)
                          if (motorIndex < NUM_MOTORS) {
                              setStallGuardState(motorIndex, value == 1);
                          } else {
                              // Set for all motors
                              for (int i = 0; i < NUM_MOTORS; i++) {
                                  setStallGuardState(i, value == 1);
                              }
                          }
                      }

                  }
                }
              }