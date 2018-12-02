#!/usr/bin/python

import dlib
import sys
import cv2 as cv
import numpy as np
from math import sqrt
import random

if len(sys.argv) != 3:
    print(
        "Call this program like this:\n"
        "   ./main.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat\n"
        "You can download a trained facial shape predictor and recognition model from:\n"
        "    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n"
        "    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
    exit()

predictor_path = sys.argv[1]
face_rec_model_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

def euclDistance(face1, face2):
    result = face1 - face2
    result = result ** 2
    result = np.sum(result)
    return sqrt(result)

def drawBox(shape, pred):
    global frame, colors

    if pred > 0:
        text = 'Person {}'.format(pred)
        color = colors[pred-1]
    else:
        text = 'Unknown'
        color = (0,255,0)
    box_pt1 = (shape.rect.tl_corner().x, shape.rect.tl_corner().y)
    box_pt2 = (shape.rect.br_corner().x, shape.rect.br_corner().y)
    cv.rectangle(frame, box_pt1, box_pt2, color, 2)
    
    text_origin = (box_pt1[0], box_pt2[1] + 25)
    cv.putText(frame, text, text_origin, cv.FONT_HERSHEY_DUPLEX, 1, color, 2)
    
def processFrame():
    global frame, ids
    
    dets = detector(frame, 0)
    shapes = []
    preds = []
    for d in dets:
        shape = sp(frame, d)
        shapes.append(shape)
        
        target_face = np.array(facerec.compute_face_descriptor(frame, shape))
        match = False
        for i, id in ids.items():
            if euclDistance(np.array(id), target_face) < 0.6:
                preds.append(i)
                match = True
                break
        if not match:
            preds.append(0)

    for i, box in enumerate(shapes):
        drawBox(box, preds[i])
    
    
def calibrateImage(img):
    global calibrated, ids, colors
    
    dets = detector(img, 0)
    if len(dets) > 0:
        for i, det in enumerate(dets):
            shape = sp(img, det)
            face_id = facerec.compute_face_descriptor(img, shape, 100)
            ids[i+1] = face_id
            rand_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            colors.append(rand_color)
        calibrated = True
    
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 60.0)

calibrated = False
ids = {}
colors = []

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        if not calibrated:
            cv.imshow('Press C to calibrate everyone\'s faces', frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord('c'):
                calibrateImage(frame)
                if calibrated:
                    cv.destroyWindow('Press C to calibrate everyone\'s faces')
            elif key == ord('q'):
                break
        else:
            processFrame()
            cv.imshow('Running Demo', frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord('r'):
                cv.destroyWindow('Running Demo')
                calibrated = False
                ids.clear()
                colors.clear()
            elif key == ord('q'):
                break
    else:
        break
		
cap.release()
cv.destroyAllWindows()