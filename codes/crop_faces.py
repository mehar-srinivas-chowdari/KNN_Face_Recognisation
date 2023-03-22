import cv2
import os
import numpy as np
#import dlib
import sys
print(sys.version)

def faceDetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)#convert color image to grayscale
    haar_path = "G:/Final year project/PROJECT/FACE_DETECTION/haarcascade_frontalface_default.xml"
    face_haar_cascade=cv2.CascadeClassifier(haar_path)#Load haar classifier
    faces = face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=5,minSize=(1000, 1000))#detectMultiScale returns rectangles
    
    return faces


def labels_for_training_data(directory):
    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:

            if filename.startswith("."):
                print("Skipping system file")#Skipping files that startwith .
                continue

            id=os.path.basename(path)#fetching subdirectory names
            img_path=os.path.join(path,filename)#fetching image path
            print("img_path:",img_path)
            print("id:",id)
            
            test_img = cv2.imread(img_path)#loading each image one by one
            if test_img is None:
                print("Image not loaded properly")
                continue
            faces_rect = faceDetection(test_img)#Calling faceDetection function to return faces detected in particular image
            if len(faces_rect)!=1:
               continue #Since we are assuming only single person images are being fed to classifier
            (x,y,w,h)=faces_rect[0]
            print(faces_rect[0])
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            roi_color = test_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from grayscale image
            output_dir_path = "G:/Dataset/Realtime_only_faces/"+id+'/'
            status = cv2.imwrite(output_dir_path +str(x) + str(y)+str(w)+str(h) + '_face.jpg', roi_color)
            print(status)
        
        
dir = "G:\Dataset\Real_time_faces_2"

#dir = "G:\PROJECTS\Face_recignisation_CNN\Dataset\Sample"

#Uncomment THIS

labels_for_training_data(dir)