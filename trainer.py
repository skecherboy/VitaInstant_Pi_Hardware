import cv2
import numpy as np
from PIL import Image
import os
# Path for face image database
path = '/home/pi/faceRecogProj/faceDataset'
recognizer = cv2.face.LBPHFaceRecognizer_create() # pre-built OpenCV LIB
detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml"); #Cascade from GIT
# function to get the images and label data from file path
def getImagesAndLabels(path): 
    # for images in path 
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8') # Turn image into Numpy frmt
        id = int(os.path.split(imagePath)[-1].split(".")[1]) # Get the ID number
        faces = detector.detectMultiScale(img_numpy) # Detect Face
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w]) # Numpy image array
            ids.append(id) # ID of person
    return faceSamples,ids # Return the two lists
print ("Training faces. Please wait.")
faces,ids = getImagesAndLabels(path)   # Get your returned variables
recognizer.train(faces, np.array(ids)) # Train the faces using LBPH
# Save the model into trainer/trainer.yml
recognizer.write('/home/pi/faceRecogProj/trainerData/trainer.yml') 
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))