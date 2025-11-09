The Spectral Parrot.

The Spectral Parrot is an self-actuating (feedback) instrument with 8 strings, each of which is controlled by an individually assignable motorized tuner. The motors are controlled by a Deep Reinforcement Learning algorithm that learns to tune the instrument's self-oscillating strings to approximate a target audio spectrum.

This repository comprises the code for Python (machine learning), SuperCollider (feedback processing and audio routing), and C (motor driver control running on two ESP32s). It is assumed that you have Python, SuperCollider, and the Arduino IDE (or similar) installed and are familiar with how these softwares work.

INSTRUCTIONS:

1. Build a parrot. Instructions for this are pending.
2. Download this repo.
3. In the repo, go to ../ESP32/Spec_Parrot_Motor_control_ESP32/Spec_Parrot_Motor_control_stallGuard/Spec_Parrot_Motor_control_stallGuard.ino and upload this code
    to both ESP32s
4. Connect both ESP32s to your laptop, one at a time. The ESP32 you connect first will control the odd-numbered motors (1, 3, 5, 7), while the even-numbered motors (2, 4, 6, 8) will be controlled by the second ESP32.
5. Open the file Spectral_Parrot_SuperCollider.scd and set your interface. Modify any ins and out channel values to accomodate to your hardware setup. Run the patch.

You will need some training audio for the Parrot to learn. Open the file Config.py, navigate to the line "training_audio_folder", and type in the adress of a folder with audio files. During training, the Parrot will train on any given file for n number of episodes before moving on to the next file. The number of episodes are defined in the line "training_audio_rotation_interval" in the Config.py file. If you want to train on just one file, set training_audio_rotation_interval to 0 or set a path to a folder with just one file.

Calibrating motors:
You want to first calibrate the motors. The following process assumes that you're using the same tuners as I, which are Göldo bass tuners. When tuned clockwise, these tuners will eventually block when fully contracted, while they simply keep on turning without further movement when turned all the way counter-clockwise. 
The reason for calibration is to make sure that the software keeps track of where the tuners are in terms of physical position, measured in motor steps. This ensures that the software will tend to block the tuners before they hit their extreme clockwise position, and similarly block them from traveling past the counter-clockwise position at which the strings would get too slack to oscillate.

First, make sure the strings are fairly slack. Then, in your terminal, run $ python position_motors.py --port1 "ESP32_1" --port2 "ESP32_2", where "ESP32_1" and "ESP32_2" should be replaced with the serial adresses of your ESP32s or any microcontroller you are using for motor communication. For example, my command often looks like this: $ python position_motors.py --port1 /dev/cu.usbserial-0001 --port2 /dev/cu.usbserial-2. You should be given a choice between a few commands in the terminal. Type $ cal limits. This command will cause the motors to turn the tuners clockwise as far as they can go, until they block and the drivers report a stallguard to the microcontroller, which in turn stops the motor. Once all motors have stalled, proceed to tighten the strings as much as you are comfortable. I have installed tuners on the end opposite the motor, allowing me to manually tighten the string. 
Once all strings are as tight as you like, you have a few options: Either you can type $ cal center to center all motors betweent their two extremes (tightest and slackest). Or you can simply proceed to training, which will usually start with a motor calibration.

Training:
Command: $ python main_motor_training.py
Required training flags:
--input-device: the number of your input device. You can check this by running $
--port1 and --port2 /dev/cu.usbserial-2: your serial ports for your microcontrollers
In your terminal, type $ python main_motor_training.py --input-device 8 --port1 "ESP32_1" --port2 /dev/cu.usbserial-2 "ESP32_2".
Once you run this command, training should begin by starting SuperCollider (you should hear the strings start feeding back) and triggering the motors to start calibration. This calibration consists in all tuners turning to their clockwise stallguard block (see the section on motor calibration above) and then position themselves at a random point between the clockwise and counter-clockwise limits. This calibration takes place after each episode in order to avoid overfitting to a single starting point and additionally avoiding any position tracking drift accumulating across episodes. You can set the flag --skip-calibration to go straight to training. This flag is only recommended for troubleshooting or short test runs, as the motor tracking will be off without calibration.




    
