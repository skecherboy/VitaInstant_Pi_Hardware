import cv2 # install openCV enviroment in Pi
import numpy as np #install numpy in python enviroment
import os # comes with Pi
import serial # comes with pi this is for NFC
import RPi.GPIO as GPIO # comes with pi
GPIO.setmode(GPIO.BOARD)

from servoTest import RunServo #function in servoTest file
from apiRequest import PostReq #custom functiom from apiRequest file

from picamera import PiCamera #install in pi proprietary library 
from picamera.array import PiRGBArray 

from time import sleep

import firebase_admin #pip install --upgrade firebase-admin
import google.cloud
from firebase_admin import credentials, firestore

# get json key from the console
cred = credentials.Certificate("Enter serviceaccountkey directory here")
app = firebase_admin.initialize_app(cred)

store = firestore.client()
doc_ref = store.collection(u'UserData').document(u'RnDZwSsR1IRZjL0xLHMCdtAdTjg2').collection('boolState')




recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/home/pi/faceRecogProj/trainerData/trainer.yml')
cascadePath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter
id = 0

#indicate dispense counts
carlosDispense = False
# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Carlos', 'Mike', 'SJ'] 
# Initialize and start realtime video capture
#cam = cv2.VideoCapture(0)
cam = PiCamera()
cam.resolution = (512,256)
cam.framerate = 41# Framerate speed that the Pi-4 can handle appropriately
rawCap = PiRGBArray(cam, size=(512,256))

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
        minNeighbors = 8,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # Check if confidence is less them 100 ==> "0" is perfect match 
        
        if (names[id] == "Carlos"):
            carlosCheck = 100 - confidence
            if (carlosCheck>=45):
                carlosDispense +=1
                print("Carlos Detected")
        if (names[id] == "Mike"):
            mikeCheck = 100 - confidence
            if (mikeCheck>=50):
                print("Mike Detected")
                
        if (names[id] == "SJ"):
            sjCheck = 100 - confidence
            if (sjCheck>=45):
                print("SJ detected")
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
    
    if (carlosDispense == True):
        RunServo()
        PostReq("Carlos", "Carlos")
        carlosDispense = False
        continue
    
    
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    docs = doc_ref.get()
    print(docs)
    if k == 27:
        break
    else:
        if __name__ == '__main__':
            ser = serial.Serial('/dev/ttyACM0', 9600, timeout=.25)
            ser.flush()
            line = ser.readline().decode('utf-8').rstrip()
            if line == ("C9BD57A3"):
                print(line)
                carlosDispense==True
            elif line == ("E7F0513B"):
                print(line)
            continue
        continue

# Do a bit of cleanup
GPIO.cleanup()
print("\n [INFO] Exiting Program and cleanup stuff")
cv2.destroyAllWindows()