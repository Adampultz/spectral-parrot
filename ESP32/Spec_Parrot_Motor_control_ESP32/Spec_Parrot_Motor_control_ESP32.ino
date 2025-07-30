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

              // Motor status and settings
              bool shaft[NUM_MOTORS] = {false};
              bool isMoving[NUM_MOTORS] = {false};
              int motorSpeed[NUM_MOTORS] = {2, 2, 2, 2};
              int motorDirection[NUM_MOTORS] = {1, 1, 1, 1};
              int stepsToMove[NUM_MOTORS] = {600, 600, 600, 600};
              int mSteps[NUM_MOTORS] = {256, 256, 256, 256};
              int accelSpeed[NUM_MOTORS] = {400, 400, 400, 400};
              uint16_t mAmps[NUM_MOTORS] = {1200, 1200, 1200, 1200}; // Set to 1.2A (1200mA) based on our previous discussion
              long currentPosition[NUM_MOTORS] = {0, 0, 0, 0};
              long targetPosition[NUM_MOTORS] = {0, 0, 0, 0};
              bool movementCompleteSent[NUM_MOTORS] = {false};

              const int varianceSize = 10;
              const int variance_threshold = 1000;
              const int accelerationTime = 2000;

              // Add basic StallGuard variables - only these are new
              uint16_t stallGuardResult[NUM_MOTORS] = {0};
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

              void checkStallGuard() {
      static uint32_t lastCheck = 0;
      if (millis() - lastCheck < 50) { // Check every 50ms
          return;
      }
      lastCheck = millis();
      
      // For variance calculation
      static uint16_t sgHistory[NUM_MOTORS][varianceSize] = {0};
      static uint8_t historyIndex[NUM_MOTORS] = {0};
      
      // For acceleration detection
      static uint32_t movementStartTime[NUM_MOTORS] = {0};
      static bool movementStarted[NUM_MOTORS] = {false};
      
      // Add warning flags instead of immediate stopping
      static bool stallGuardWarning[NUM_MOTORS] = {false};
      static uint8_t warningCount[NUM_MOTORS] = {0};
      
      for (int i = 0; i < NUM_MOTORS; i++) {
          // Check if motion just started
          if (isMoving[i] && !movementStarted[i]) {
              movementStarted[i] = true;
              movementStartTime[i] = millis();
              stallGuardWarning[i] = false;  // Reset warning flag
              warningCount[i] = 0;           // Reset warning counter
          }
          // Check if motion just stopped
          else if (!isMoving[i] && movementStarted[i]) {
              movementStarted[i] = false;
          }
          
          if (isMoving[i]) {
              // Read StallGuard value
              stallGuardResult[i] = drivers[i]->SG_RESULT();
              
              // Store in history array
              sgHistory[i][historyIndex[i]] = stallGuardResult[i];
              historyIndex[i] = (historyIndex[i] + 1) % varianceSize;
              
              // Only check after acceleration phase
              if (millis() - movementStartTime[i] > accelerationTime) {
                  // Calculate mean
                  uint32_t sum = 0;
                  for (int j = 0; j < varianceSize; j++) {
                      sum += sgHistory[i][j];
                  }
                  float mean = sum / varianceSize;

                  Serial.print("Motor ");
                          Serial.print(i + 1);
                          Serial.print(" SG Value ");
                          Serial.println(mean);
                  
                  // CRITICAL CHANGE: Don't stop motor, just warn and count
                  if (mean < sgThreshold) {
                      warningCount[i]++;
                      
                      // Only log occasionally to avoid spam
                      // if (warningCount[i] >= 50) {
                      //     Serial.print("Motor ");
                      //     Serial.print(i + 1);
                      //     Serial.print(" EMERGENCY STOP after ");
                      //     Serial.print(warningCount[i]);
                      //     Serial.println(" StallGuard warnings");
                          
                      //     // CRITICAL: Send completion message BEFORE stopping
                      //     if (!movementCompleteSent[i]) {
                      //         movementCompleteSent[i] = true;
                      //         Serial.print("MOTOR_COMPLETE:");
                      //         Serial.println(i + 1);
                      //     }
                          
                      //     // Now stop the motor
                      //     stopMotor(i);
                      //     warningCount[i] = 0;
                      // }
                      
                      // OPTION 1: Never auto-stop (safest for training)
                      // Let the motor complete normally and handle via Python
                      
                      // OPTION 2: Only stop after many consecutive warnings (safer)
                      if (warningCount[i] >= 25) { // 50 * 50ms = 2.5 seconds of warnings
                          Serial.print("Motor ");
                          Serial.print(i + 1);
                          Serial.print(" EMERGENCY STOP after ");
                          Serial.print(warningCount[i]);
                          Serial.println(" StallGuard warnings");

                          if (!movementCompleteSent[i]) {
                              movementCompleteSent[i] = true;
                              Serial.print("MOTOR_COMPLETE:");
                              Serial.println(i + 1);
                          }
                          
                          stopMotor(i);
                          warningCount[i] = 0;
                      }
                  } else {
                      // Reset warning count if StallGuard is normal
                      warningCount[i] = 0;
                  }
              }
          }
      }
  }

              // Function to set speed for a specific motor
              void setMotorSpeed(int motorIndex, int speed) {
                motorSpeed[motorIndex] = speed;
                steppers[motorIndex].setSpeedInStepsPerSecond(speed * mSteps[motorIndex]);
                
                Serial.print("Motor ");
                Serial.print(motorIndex + 1);
                Serial.print(" speed set to: ");
                Serial.println(speed);
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
                
                Serial.print("Motor ");
                Serial.print(motorIndex + 1);
                Serial.print(" direction set to: ");
                Serial.println(direction == 1 ? "Clockwise" : "Counter-clockwise");
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
                
                // Debug the values before setting the target
                // Serial.print("Motor ");
                // Serial.print(motorIndex + 1);
                // Serial.print(" moving steps: ");
                // Serial.println(steps);
                // Serial.print("From position: ");
                // Serial.print(currentPosition[motorIndex]);
                // Serial.print(" to position: ");
                // Serial.println(targetPosition[motorIndex]);
                
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
                  // } else {
                  //   Serial.print("Motor ");
                  //     Serial.print(motorIndex + 1);
                  //     Serial.println(" has already stopped");
                  // }
                }

              void loop() {
                // Process movement for all steppers and check completion
                for (int i = 0; i < NUM_MOTORS; i++) {
                  if (isMoving[i]) {
                    // Process the movement
                    steppers[i].processMovement();
                    
                //     // Check if movement is complete
                //      if (steppers[i].motionComplete()) {
                //   // Only send completion message if not already sent
                //   if (!movementCompleteSent[i]) {
                //     isMoving[i] = false;
                //     movementCompleteSent[i] = true;
                    
                //     // Send completion message in a standard format for reliable parsing
                //     Serial.print("MOTOR_COMPLETE:");
                //     Serial.println(i + 1);  // Convert to 1-based index
                //   }
                // }
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
        //                   else if (commandType == "TEST") {
        //     Serial.println("ESP32 is alive and responding!");
        //     // Test each driver
        //     for (int i = 0; i < NUM_MOTORS; i++) {
        //         Serial.print("Driver ");
        //         Serial.print(i + 1);
        //         Serial.print(" version: 0x");
        //         Serial.println(drivers[i]->version(), HEX);
        //     }
        // }
                  }
                }
              }