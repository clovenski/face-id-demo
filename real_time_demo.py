#!/usr/bin/python

from __future__ import print_function

import dlib
import sys
import cv2 as cv
import numpy as np
from math import sqrt
import random

#   You can download a trained facial shape predictor and recognition model from:
#       http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
#       http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

# adjust these accordingly if needed
predictor_path = './models/shape_predictor_5_face_landmarks.dat'
face_rec_model_path = './models/dlib_face_recognition_resnet_model_v1.dat'

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
    cv.rectangle(frame, box_pt1, box_pt2, color, thickness=2)
    
    text_origin = (box_pt1[0], box_pt2[1] + 25)
    cv.putText(frame, text, text_origin, cv.FONT_HERSHEY_DUPLEX, 1, color, thickness=2)
    
def processFrame(num_upsamples=0):
    global frame, ids
    
    dets = detector(frame, num_upsamples)
    shapes = []
    preds = []
    for d in dets:
        shape = sp(frame, d)
        shapes.append(shape)
        
        target_face = np.array(facerec.compute_face_descriptor(frame, shape))
        potential_ids = {}
        for i, id in ids.items():
            dist = euclDistance(np.array(id), target_face)
            if  dist < 0.6:
                potential_ids[i] = dist
        if len(potential_ids) == 0:
            preds.append(0)
        else:
            preds.append(min(potential_ids, key=potential_ids.get))

    for i, box in enumerate(shapes):
        drawBox(box, preds[i])
    
    
def calibrateImage(img, num_upsamples=0):
    global calibrated, ids, colors
    
    dets = detector(img, num_upsamples)
    if len(dets) > 0:
        for i, det in enumerate(dets):
            shape = sp(img, det)
            face_id = facerec.compute_face_descriptor(img, shape, 100)
            ids[i+1] = face_id
            rand_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            while rand_color in colors or rand_color == (0,255,0):
                rand_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            colors.append(rand_color)
        calibrated = True


def process_args():
    global frame

    windows_to_remove = []

    print('Calibrating given image . . . ', end='')
    try:
        init_img = dlib.load_rgb_image(sys.argv[1])
        frame = cv.cvtColor(init_img, cv.COLOR_RGB2BGR)
        calibrateImage(frame, num_upsamples=1)
        processFrame(num_upsamples=1)
        if calibrated:
            print('done')
            cv.imshow('Reference', frame)
            windows_to_remove.append('Reference')
        else:
            print('failed')
            return None
    except:
        print('failed')
        return None

    for img_fn in sys.argv[2:]:
        try:
            init_img = dlib.load_rgb_image(img_fn)
            frame = cv.cvtColor(init_img, cv.COLOR_RGB2BGR)
            processFrame(num_upsamples=1)
            cv.imshow(img_fn, frame)
            windows_to_remove.append(img_fn)
        except:
            print('Failed loading image: {}'.format(img_fn))

    return windows_to_remove
    

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 60.0)

calibrated = False
ids = {}
colors = []

if len(sys.argv) >= 2:
    window_names = process_args()
else:
    window_names = None

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        if not calibrated:
            cv.imshow('face-id-demo', frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord('c'):
                print('Calibrating image . . . ', end='')
                calibrateImage(frame)
                if calibrated:
                    print('done')
                    print('Faces found: {}'.format(len(ids)))
                else:
                    print('failed')
            elif key == ord('q'):
                break
        else:
            processFrame()
            cv.imshow('face-id-demo', frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord('r'):
                print('Calibration has been reset')
                if window_names != None:
                    for name in window_names:
                        cv.destroyWindow(name)
                    window_names = None
                else:
                    cv.destroyWindow('Reference')
                calibrated = False
                ids.clear()
                colors.clear()
            elif key == ord('q'):
                break
    else:
        break
		
cap.release()
cv.destroyAllWindows()