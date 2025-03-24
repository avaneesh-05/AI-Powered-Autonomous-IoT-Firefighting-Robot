import RPi.GPIO as GPIO
import time
import threading
import os

# Set the GPIO mode														
GPIO.setmode(GPIO.BOARD)

# Pin configuration
relay_pin = 22  # GPIO 22 is pin 15 on the Raspberry Pi

# Set up the GPIO pin as output
GPIO.setup(relay_pin, GPIO.OUT)

def motor_on():
    GPIO.output(relay_pin, GPIO.LOW)  # Activate relay to turn motor ON
    print("Motor is ON")

def terminate_program():
    GPIO.output(relay_pin, GPIO.HIGH)  # Deactivate relay to turn motor OFF
    print("Motor is OFF")
    print("Time's up! Terminating the program.")
    GPIO.cleanup()
    os._exit(1)  # Forcefully terminate the program

# Turn the motor ON and set a timer to turn it off and terminate the program after 10 seconds
try:
    motor_on()
    
    # Set a timer to automatically turn the motor off and terminate after 10 seconds
    timer = threading.Timer(12, terminate_program)
    timer.start()

    # Keep the program running until the timer finishes
    while True:
        time.sleep(1)

finally:
    # Clean up the GPIO state before exiting
    GPIO.cleanup()
    print("GPIO cleanup done")