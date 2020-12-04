#faceRecogDataset
import os
import cv2
import numpy as nmp

from time import sleep
from picamera import PiCamera
from picamera.array import PiRGBArray

faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

cam    = PiCamera() # shorten function to variable cam
cam.resolution = (640,480) #Small Resolution sutiable for cam
cam.framerate = 40# Framerate speed that the Pi-4 can handle appropriately
rawCap = PiRGBArray(cam, size=(640,480)) # Raw Image data in Array format so OPENCV can use
                         # Size is passed in to match resolution
#Sensor warmup    
sleep(1)

for frame in cam.capture_continuous(rawCap, format="bgr", use_video_port=True):
    img = frame.array
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]  
    cv2.imshow('video',img)
    rawCap.truncate(0)
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'): # press 'q' to quit
        break
rawCap.release()
cv2.destroyAllWindows()
