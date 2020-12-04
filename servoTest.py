import RPi.GPIO as GPIO  # Imports the standard Raspberry Pi GPIO library
from time import sleep   # Imports sleep (aka wait or pause) into the program



def runServo():
    GPIO.setmode(GPIO.BOARD)
    #for i in range(a): 
    #i+=1
    # Set up pin 16 for PWM
    GPIO.setup(16,GPIO.OUT)  # Sets up pin 16 to an output (instead of an input)
    p = GPIO.PWM(16,50)      # Sets up pin 16 as a PWM pin
    p.start(0)               # Starts running PWM on the pin and sets it to 0
    p.ChangeDutyCycle(3)     # Changes the duty cycle to 3%  ().6 ms pulse) 0 Degrees
    sleep(1)                 # Wait 1 second
    p.ChangeDutyCycle(12.5)  # Changes the duty cycle to 12.5%  (2.5 ms pulse) 180 Degrees
    sleep(1)

    # Clean up everything
    p.stop()                 # At the end of the program, stop the PWM
    GPIO.cleanup()           # Resets the GPIO pins back to defaults
 

