import RPi.GPIO as GPIO
import time
import subprocess
import os

# GPIO Pin Setup
FIRE_SENSOR_PIN1 = 11
FIRE_SENSOR_PIN2 = 13
FIRE_SENSOR_PIN3 = 15
MOTOR_ENA = 33  # Motor A PWM
MOTOR_ENB = 32  # Motor B PWM
MOTOR_IN1 = 31  # Motor A IN1
MOTOR_IN2 = 29  # Motor A IN2
MOTOR_IN3 = 18  # Motor B IN3
MOTOR_IN4 = 16  # Motor B IN4
SERVO_PIN = 24  # Servo motor pin

# Setup GPIO mode
GPIO.setmode(GPIO.BOARD)

# Setup pins
GPIO.setup(FIRE_SENSOR_PIN1, GPIO.IN)
GPIO.setup(FIRE_SENSOR_PIN2, GPIO.IN)
GPIO.setup(FIRE_SENSOR_PIN3, GPIO.IN)
GPIO.setup(MOTOR_ENA, GPIO.OUT)
GPIO.setup(MOTOR_ENB, GPIO.OUT)
GPIO.setup(MOTOR_IN1, GPIO.OUT)
GPIO.setup(MOTOR_IN2, GPIO.OUT)
GPIO.setup(MOTOR_IN3, GPIO.OUT)
GPIO.setup(MOTOR_IN4, GPIO.OUT)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# PWM setup
servo = GPIO.PWM(SERVO_PIN, 50)  # 50Hz for servo
servo.start(0)
pwm_A = GPIO.PWM(MOTOR_ENA, 100)  # 100Hz PWM
pwm_B = GPIO.PWM(MOTOR_ENB, 100)
pwm_A.start(100)  # Start with 100% duty cycle
pwm_B.start(100)

# Functions for motor control
def move_forward():
    GPIO.output(MOTOR_IN1, GPIO.HIGH)
    GPIO.output(MOTOR_IN2, GPIO.LOW)
    GPIO.output(MOTOR_IN3, GPIO.HIGH)
    GPIO.output(MOTOR_IN4, GPIO.LOW)

def move_backward():
    GPIO.output(MOTOR_IN1, GPIO.LOW)
    GPIO.output(MOTOR_IN2, GPIO.HIGH)
    GPIO.output(MOTOR_IN3, GPIO.LOW)
    GPIO.output(MOTOR_IN4, GPIO.HIGH)

def stop_motors():
    GPIO.output(MOTOR_IN1, GPIO.LOW)
    GPIO.output(MOTOR_IN2, GPIO.LOW)
    GPIO.output(MOTOR_IN3, GPIO.LOW)
    GPIO.output(MOTOR_IN4, GPIO.LOW)

# Function for fire detection
def detect_fire():
    return (GPIO.input(FIRE_SENSOR_PIN1) == 1 or 
            GPIO.input(FIRE_SENSOR_PIN2) == 0 or 
            GPIO.input(FIRE_SENSOR_PIN3) == 1)

# Servo function to spray water
def spray_water():
    for _ in range(2):
        for angle in range(30, 151):
            set_servo_angle(angle)
        for angle in range(150, 29, -1):
            set_servo_angle(angle)
    set_servo_angle(90)

# Function to set servo angle
def set_servo_angle(angle):
    duty_cycle = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.01)
    servo.ChangeDutyCycle(0)

# Function to start the pump using subprocess
def start_pump():
    subprocess.Popen(["python3", "pump2.py"])
    print("Pump script started")

# Function to start person detection using subprocess
def start_person_detection():
    return subprocess.Popen(["python3", "signal_person.py", "--modeldir=Sample_TFlite_model"])

# Main logic
try:
    
    # Start person detection subprocess
    person_detection_process = start_person_detection()
    print("Person detection started")
    
    time.sleep(5)
    
    move_forward()
    start_time = time.time()
    pump_process = None

    while True:
        if detect_fire():
            print("Fire detected! Stopping, activating pump, and spraying water...")
            stop_motors()
            if pump_process is None or pump_process.poll() is not None:
                pump_process = start_pump()
            spray_water()
            
            # Check for fire for 5 seconds after spraying
            fire_check_start = time.time()
            while time.time() - fire_check_start < 5:
                if not detect_fire():
                    break
                time.sleep(0.5)
            
            if not detect_fire():
                print("Fire extinguished. Moving backward for 2.5 seconds.")
                if pump_process and pump_process.poll() is None:
                    pump_process.terminate()
                    print("Pump script terminated")
                move_backward()
                time.sleep(2.5)
                stop_motors()
                break  # End the program after moving backward
            else:
                print("Fire still detected. Continuing to spray.")
        
        # Stop after 30 seconds if no fire is detected initially
        if time.time() - start_time > 10:
            print("No fire detected for 10 seconds. Stopping the bot.")
            stop_motors()
            break
        
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Program interrupted by user.")

finally:
    if pump_process and pump_process.poll() is None:
        pump_process.terminate()
        print("Pump script terminated")
    if person_detection_process and person_detection_process.poll() is None:
        person_detection_process.terminate()
        print("Person detection script terminated")
    pwm_A.stop()
    pwm_B.stop()
    servo.stop()
    GPIO.cleanup()
    print("GPIO cleanup completed.")
