import cv2
import numpy as np
import os

from picamera import PiCamera
from picamera.array import PiRGBArray

from time import sleep

# Prebuilt Prediction Algorithim
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Read the YML file created by the trainer
recognizer.read('/home/pi/faceRecogProj/trainerData/trainer.yml')
cascadePath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Carlos', 'Mike'] 
# Initialize and start realtime video capture
#cam = cv2.VideoCapture(0)
cam = PiCamera()
cam.resolution = (640,480)
cam.framerate = 40# Framerate speed that the Pi-4 can handle appropriately
rawCap = PiRGBArray(cam, size=(640,480))

# Define min window size to be recognized as a face
minW = 0.15
minH = 0.15

sleep(1)

for frame in cam.capture_continuous(rawCap, format="bgr", use_video_port=True):
    img = frame.array
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img)
    
    rawCap.truncate(0)
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
