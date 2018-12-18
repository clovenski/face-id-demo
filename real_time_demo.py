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

# dlib's face detector
detector = dlib.get_frontal_face_detector()
# dlib's model for facial landmarks
sp = dlib.shape_predictor(predictor_path)
# dlib's model for facial recognition; outputs the 128 dim vector space
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# function to compute Euclidean distance between two numpy arrays
#   that represent the 128 dim vectors computed from dlib's face_rec model
def euclDistance(face1, face2):
    result = face1 - face2
    result = result ** 2
    result = np.sum(result)
    return sqrt(result)

# function to draw the bounding box on the given shape object coming from
#   dlib's shape predictor; if pred == 0 then face is unknown, otherwise
#   the bounding box and text will be in the color corresponding to the
#   appropriate person
def drawBox(shape, pred):
    global frame, colors

    # if this face was identified as someone
    if pred > 0:
        text = 'Person {}'.format(pred)
        color = colors[pred-1]
    else: # unknown face
        text = 'Unknown'
        color = (0,255,0)

    # get the top left and bottom right points of the bounding box
    box_pt1 = (shape.rect.tl_corner().x, shape.rect.tl_corner().y)
    box_pt2 = (shape.rect.br_corner().x, shape.rect.br_corner().y)
    # draw the bounding box
    cv.rectangle(frame, box_pt1, box_pt2, color, thickness=2)
    # draw the text under the box
    text_origin = (box_pt1[0], box_pt2[1] + 25)
    cv.putText(frame, text, text_origin, cv.FONT_HERSHEY_DUPLEX, 1, color, thickness=2)

# function to process the current frame of the webcam; num_upsamples defines how
#   many times to upsample the frame to try to detect smaller faces
def processFrame(num_upsamples=0):
    global frame, ids

    # get the face detections
    dets = detector(frame, num_upsamples)
    shapes = []
    preds = []
    # for every detected face in the frame
    for d in dets:
        # get the position of the face
        shape = sp(frame, d)
        shapes.append(shape)
        
        # compute the vector of the face
        target_face = np.array(facerec.compute_face_descriptor(frame, shape))
        potential_ids = {}
        # compute Euclidean distance between the vector and vectors of all known faces
        for i, id in ids.items():
            dist = euclDistance(np.array(id), target_face)
            if  dist < 0.6:
                potential_ids[i] = dist
        # if this face is unknown
        if len(potential_ids) == 0:
            preds.append(0)
        else: # make prediction correspond to smallest distance
            preds.append(min(potential_ids, key=potential_ids.get))

    # draw all bounding boxes
    for i, box in enumerate(shapes):
        drawBox(box, preds[i])
    
# function to calibrate with the given image; num_upsamples defines how many times
#   to upsample the image to try to detect smaller faces
def calibrateImage(img, num_upsamples=0):
    global calibrated, ids, colors

    # get the face detections
    dets = detector(img, num_upsamples)
    # if at least one face was detected
    if len(dets) > 0:
        # for every detected face, store its id vector and assign it a unique random color
        for i, det in enumerate(dets):
            shape = sp(img, det)
            # compute face id (128D vector); third arg -> jitter image 100 times then use avg
            face_id = facerec.compute_face_descriptor(img, shape, 100)
            ids[i+1] = face_id
            rand_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            while rand_color in colors or rand_color == (0,255,0):
                rand_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            colors.append(rand_color)
        calibrated = True


# function to process the arguments given to the script;
#   returns a list of the names of the windows to remove
def process_args():
    global frame

    windows_to_remove = []

    # calibrate with the first arg/image
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

    # make predictions on the rest of the args/images
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


# get the appropriate video source (webcam)
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 60.0)

calibrated = False
# int -> 128D vector; person # to corresponding face id
ids = {}
# index corresponds to (person # + 1) and stores corresponding color
colors = []

if len(sys.argv) >= 2:
    window_names = process_args()
else:
    window_names = None

# keep processing webcam feed
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
        else: # faces are calibrated
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

# release video source and destroy all windows
cap.release()
cv.destroyAllWindows()