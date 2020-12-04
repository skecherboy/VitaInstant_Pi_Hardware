# This Captures the user's face so the trainer can analyze it

import os #os dependent functionality 
import cv2 #OpenCV
import numpy as nmp # Math

from picamera import PiCamera #Pi Cam Library
from picamera.array import PiRGBArray #Pi images to Array Format

from time import sleep 

#Assign faceCascade var to folder containing pre-built haar cascade algorithim
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

#Assign an ID to each person
face_id = input('\n enter user id end press <return> ==>  ')
print("\n Initializing face capture. Look the camera ")

# Initialize face count
count = 0


cam    = PiCamera() # shorten function to variable cam
cam.resolution = (800,608) #Small Resolution sutiable for cam
cam.framerate = 45 # Framerate speed that the Pi-4 can handle appropriately
rawCap = PiRGBArray(cam, size=(800,608)) # Raw Image data in Array format so OPENCV can use
                         # Size is passed in to match resolution
#Sensor warmup    
sleep(1)

#Capture continuous = infinite iterative captures
#frame = each capture
for frame in cam.capture_continuous(rawCap, format="bgr", use_video_port=True):
    img = frame.array # Img to OpenCV
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # helps cascade
    faces = faceCascade.detectMultiScale(  
        gray,            # grab gray image
        scaleFactor=1.2, # Scale Pyramid 
        minNeighbors=10, # Rectangle Neighbors    
        minSize=(20, 20) # min Rectangle size for face
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # xy is small rectangle on top edge of face
        # x+w / y+w expands a rectangle to edges of face
        count += 1 # one more picture
        # Write to the directory 
        cv2.imwrite("/home/pi/faceRecogProj/faceDataset/User." + str(face_id) + '.' +
                    str(count) + ".jpg", gray[y:y+h, x:x+w])
        #Show "image" on new window on screen
        cv2.imshow('image',img)
        
    rawCap.truncate(0) #Clear the stream and prepare for next image in array
    # key press is detected and only check for last 8 bits
    key = cv2.waitKey(1) & 0xff
    # Press Q to quit program
    if key == ord('q'): # press 'q' to quit
        break
    elif count >= 200: # Take 30 face sample and stop video
        break
    
cv2.destroyAllWindows()