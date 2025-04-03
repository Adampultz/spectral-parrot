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
int accelSpeed[NUM_MOTORS] = {20, 20, 20, 20}; // Reduced acceleration for smoother operation
long currentPosition[NUM_MOTORS] = {0, 0, 0, 0};
long targetPosition[NUM_MOTORS] = {0, 0, 0, 0};

// StallGuard variables
uint16_t stallGuardResult[NUM_MOTORS] = {0};
uint16_t minStallGuardValue[NUM_MOTORS] = {255, 255, 255, 255}; // Track minimum values between reports
int stallGuardThreshold = 50;  // Unified StallGuard threshold
bool motorStalled[NUM_MOTORS] = {false}; // Track stall status for each motor
bool stallGuardEnabled = true; // Global enable/disable stall detection
bool motorStallGuardEnabled[NUM_MOTORS] = {true, true, true, true}; // Per-motor enable/disable
unsigned long motorStartTime[NUM_MOTORS] = {0}; // Track when each motor started moving
int stallGuardStartupDelay = 1000; // Ignore StallGuard for first 1000ms of movement

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
    drivers[i]->rms_current(1200);  // Set to 1.2A (1200mA) for 4 motors on 5A power supply
    drivers[i]->microsteps(mSteps[i]);
    drivers[i]->pwm_autoscale(true);
    drivers[i]->en_spreadCycle(false);  // false = StealthChop / true = SpreadCycle

    // Configure StallGuard
    drivers[i]->SGTHRS(stallGuardThreshold);
    drivers[i]->TCOOLTHRS(0xFFFFF); // ~20bit max - Disable coolStep while moving slowly
    
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
    
    delay(50); // Small delay between driver initializations
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
  
  Serial.println("Ready for commands:");
  Serial.println("- MOTOR:num SPEED:value");
  Serial.println("- MOTOR:num DIR:value");
  Serial.println("- MOTOR:num STEPS:value");
  Serial.println("- MOTOR:num STOP:value");
  Serial.println("- MOTOR:num SG:value");
  Serial.println("- MOTOR:num SGENABLE:0/1");
  Serial.println("- MOTOR:num SGDELAY:value");
  Serial.println("- MOTOR:num ACCEL:value");
  Serial.println("Use MOTOR:0 to control all motors at once");
}

// Function to set StallGuard threshold
void setStallGuardThreshold(int motorIndex, int threshold) {
  if (threshold < 0) threshold = 0;
  if (threshold > 255) threshold = 255;
  
  if (motorIndex < NUM_MOTORS) {
    // Set for a specific motor
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
    // Update the global threshold
    stallGuardThreshold = threshold;
    Serial.print("All motors StallGuard threshold set to: ");
    Serial.println(threshold);
  }
}

// Function to set StallGuard startup delay
void setStallGuardDelay(int motorIndex, int delayMs) {
  if (delayMs < 0) delayMs = 0;
  if (delayMs > 5000) delayMs = 5000; // Cap at 5 seconds
  
  if (motorIndex < NUM_MOTORS) {
    Serial.print("Motor ");
    Serial.print(motorIndex + 1);
    Serial.print(" StallGuard startup delay set to: ");
    Serial.print(delayMs);
    Serial.println("ms");
  } else {
    // Set for all motors
    stallGuardStartupDelay = delayMs;
    Serial.print("All motors StallGuard startup delay set to: ");
    Serial.print(delayMs);
    Serial.println("ms");
  }
}

// Function to set acceleration
void setAcceleration(int motorIndex, int acceleration) {
  if (acceleration < 1) acceleration = 1;
  
  if (motorIndex < NUM_MOTORS) {
    accelSpeed[motorIndex] = acceleration;
    steppers[motorIndex].setAccelerationInStepsPerSecondPerSecond(acceleration * mSteps[motorIndex]);
    Serial.print("Motor ");
    Serial.print(motorIndex + 1);
    Serial.print(" acceleration set to: ");
    Serial.println(acceleration);
  } else {
    // Set for all motors
    for (int i = 0; i < NUM_MOTORS; i++) {
      accelSpeed[i] = acceleration;
      steppers[i].setAccelerationInStepsPerSecondPerSecond(acceleration * mSteps[i]);
    }
    Serial.print("All motors acceleration set to: ");
    Serial.println(acceleration);
  }
}

// Function to enable/disable stall detection
void setStallGuardEnable(int motorIndex, bool enable) {
  if (motorIndex < NUM_MOTORS) {
    // Set for a specific motor
    motorStallGuardEnabled[motorIndex] = enable;
    Serial.print("Motor ");
    Serial.print(motorIndex + 1);
    Serial.print(" stall detection ");
    Serial.println(enable ? "enabled" : "disabled");
  } else {
    // Set for all motors
    stallGuardEnabled = enable;
    for (int i = 0; i < NUM_MOTORS; i++) {
      motorStallGuardEnabled[i] = enable;
    }
    Serial.print("All motors stall detection ");
    Serial.println(enable ? "enabled" : "disabled");
  }
}

// Function to check StallGuard values more frequently but report at regular intervals
void checkStallGuard() {
  static unsigned long lastCheckTime = 0;
  static unsigned long lastReportTime = 0;
  unsigned long currentTime = millis();
  
  // Sample StallGuard values more frequently (every 20ms)
  if (currentTime - lastCheckTime >= 20) {
    lastCheckTime = currentTime;
    
    // Reset min values at the start of each report cycle
    if (currentTime - lastReportTime >= 100) {
      for (int i = 0; i < NUM_MOTORS; i++) {
        minStallGuardValue[i] = 255;
      }
    }
    
    for (int i = 0; i < NUM_MOTORS; i++) {
      if (isMoving[i]) {
        // Read the StallGuard value
        uint16_t sgValue = drivers[i]->SG_RESULT();
        stallGuardResult[i] = sgValue;
        
        // Update minimum value seen
        if (sgValue < minStallGuardValue[i]) {
          minStallGuardValue[i] = sgValue;
        }
        
        // Only perform stall detection if enabled and past startup delay
        if (stallGuardEnabled && motorStallGuardEnabled[i] && 
            (currentTime - motorStartTime[i] >= stallGuardStartupDelay)) {
          
          // Check if current value falls below threshold
          if (sgValue < stallGuardThreshold) {
            if (!motorStalled[i]) {
              motorStalled[i] = true;
              
              // Report the stall with detailed information
              Serial.print("Motor ");
              Serial.print(i + 1);
              Serial.print(" STALLED! SG value: ");
              Serial.print(sgValue);
              Serial.print(" (min: ");
              Serial.print(minStallGuardValue[i]);
              Serial.print("), Threshold: ");
              Serial.println(stallGuardThreshold);
              
              // Stop the motor
              stopMotor(i);
            }
          } else if (motorStalled[i] && sgValue >= stallGuardThreshold) {
            // Reset stall status when value goes above threshold
            motorStalled[i] = false;
            Serial.print("Motor ");
            Serial.print(i + 1);
            Serial.println(" recovered from stall");
          }
        }
      }
    }
  }
  
  // Report at regular intervals (every 100ms)
  if (currentTime - lastReportTime >= 100) {
    lastReportTime = currentTime;
    
    for (int i = 0; i < NUM_MOTORS; i++) {
      if (isMoving[i]) {
        Serial.print("Motor ");
        Serial.print(i + 1);
        Serial.print(" SG: ");
        Serial.print(stallGuardResult[i]);
        if (minStallGuardValue[i] < 255 && minStallGuardValue[i] != stallGuardResult[i]) {
          Serial.print(" (min: ");
          Serial.print(minStallGuardValue[i]);
          Serial.print(")");
        }
        Serial.println();
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
  Serial.print("Motor ");
  Serial.print(motorIndex + 1);
  Serial.print(" moving steps: ");
  Serial.println(steps);
  Serial.print("From position: ");
  Serial.print(currentPosition[motorIndex]);
  Serial.print(" to position: ");
  Serial.println(targetPosition[motorIndex]);
  
  // Reset stall status before starting movement
  motorStalled[motorIndex] = false;
  
  // Reset minimum StallGuard value
  minStallGuardValue[motorIndex] = 255;
  
  // Record the start time for StallGuard delay
  motorStartTime[motorIndex] = millis();
  
  // Set the target position in the stepper object
  steppers[motorIndex].setTargetPositionInSteps(targetPosition[motorIndex]);
  isMoving[motorIndex] = true;
}

// Function to stop a specific motor
void stopMotor(int motorIndex) {
  steppers[motorIndex].setTargetPositionToStop();
  isMoving[motorIndex] = false;
  
  Serial.print("Motor ");
  Serial.print(motorIndex + 1);
  Serial.println(" emergency stop triggered");
}

void loop() {
  // Process movement for all steppers and check completion
  for (int i = 0; i < NUM_MOTORS; i++) {
    if (isMoving[i]) {
      // Process the movement
      steppers[i].processMovement();
      
      // Check if movement is complete
      if (steppers[i].motionComplete()) {
        isMoving[i] = false;
        Serial.print("Motor ");
        Serial.print(i + 1);
        Serial.println(" movement completed");
      }
    } else {
      // Still process movement even when not actively moving
      // This ensures any pending commands get handled
      steppers[i].processMovement();
    }
  }
  
  // Check StallGuard values and detect stalls
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
        } else {
          stopMotor(motorIndex);
        }
      }
      else if (commandType == "SG") {
        setStallGuardThreshold(motorIndex, value);
      }
      else if (commandType == "SGENABLE") {
        setStallGuardEnable(motorIndex, value > 0);
      }
      else if (commandType == "SGDELAY") {
        setStallGuardDelay(motorIndex, value);
      }
      else if (commandType == "ACCEL") {
        setAcceleration(motorIndex, value);
      }
      else {
        Serial.print("Unknown command type: ");
        Serial.println(commandType);
      }
    }
  }
}