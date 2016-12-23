

"""
Usage: python example_face_detection.py -i xml/th.jpeg
"""

from opencv import face_detect
import imutils
import cv2
import time
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image")

args = vars(ap.parse_args())

frame = cv2.imread(args["image"])
# print(frame.shape)
# exit()

face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
profileface_cascade = cv2.CascadeClassifier('xml/haarcascade_profileface.xml') # set to None if you don't need profile face

list_of_faces = face_detect(frame, face_cascade, profileface_cascade, window = None)

print(list_of_faces)

for (x,y,w,h) in list_of_faces:
    cv2.rectangle(frame,(x, y),(x + w, y + h), (255,0,0), 2)
    cv2.putText(frame, "Face", (x - 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
cv2.imshow("face detection example", frame)
cv2.waitKey(0)
