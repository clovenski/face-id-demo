#!/usr/bin/python

import os
import dlib
import glob
import cv2 as cv
import numpy as np
from math import sqrt

#   You can download a trained facial shape predictor and recognition model from:
#       http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
#       http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

# adjust these accordingly if needed
predictor_path = './models/shape_predictor_5_face_landmarks.dat'
face_rec_model_path = './models/dlib_face_recognition_resnet_model_v1.dat'
faces_folder_path = './images/'

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
#   dlib's shape predictor; match is true when the face matches the calibrated
#   face
def drawBox(shape, match):
    global img

    # green for match, red for no match
    color = (0,255,0) if match else (0,0,255)
    # get the top left and bottom right points of the bounding box
    box_pt1 = (shape.rect.tl_corner().x, shape.rect.tl_corner().y)
    box_pt2 = (shape.rect.br_corner().x, shape.rect.br_corner().y)
    # draw the bounding box
    cv.rectangle(img, box_pt1, box_pt2, color)

# function to process all the jpg images in faces_folder_path dir;
#   returns when all images were shown or user wants to quit
def processImages():
    global face_id, img

    for f in glob.glob(os.path.join(faces_folder_path, '*.jpg')):
        img = dlib.load_rgb_image(f)

        # get the face detections; second arg -> number of times to upsample
        #   to try to detect smaller faces
        dets = detector(img, 1)

        shapes = []
        matches = []
        for d in dets:
            shape = sp(img, d)
            shapes.append(shape)

            face_id2 = np.array(facerec.compute_face_descriptor(img, shape))
            if euclDistance(np.array(face_id), face_id2) < 0.55: # define threshold here
                matches.append(True)
            else:
                matches.append(False)

        # necessary for dlib loading img and opencv showing img
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        # draw all the bounding boxes with appropriate color as predicitons
        for i, box in enumerate(shapes):
            drawBox(box, matches[i])
        # keep showing img until user hits 'n' key for next img or qutis with 'q'
        while True:
            cv.imshow(f, img)
            key = cv.waitKey(1) & 0xFF
            if key == ord('n'):
                break
            elif key == ord('q'):
                cap.release()
                cv.destroyAllWindows()
                cleanUpImgDir()
                exit()
        cv.destroyAllWindows()

# function to calibrate with the given img;
#   global face_id will equal 128D vector if one and only one face
#   was detected, otherwise face_id = None
def computeFaceId(img):
    global face_id

    dets = detector(img, 0)
    if len(dets) != 1:
        face_id = None
    else:
        shape = sp(img, dets[0])
        face_id = facerec.compute_face_descriptor(img, shape, 100) # 50

# function to clean up faces_folder_path dir; only images that were created
#   by user's photoshoot during demo are deleted
def cleanUpImgDir():
    for f in glob.glob(os.path.join(faces_folder_path, 'photo_of_you_*')):
        os.remove(f)

# get the appropriate video source (webcam)
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 60.0)

calibrated = False
# start with image number 1, for user's photoshoot
img_num = 1

# keep processing webcam feed
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        if not calibrated:
            cv.imshow('Let me identify you!', frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord('c'):
                computeFaceId(frame)
                calibrated = True if face_id != None else False
                if calibrated:
                    cv.destroyWindow('Let me identify you!')
            elif key == ord('q'):
                break
        else: # face is calibrated
            cv.imshow('Take photo number {}!'.format(img_num), frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord('c'):
                cv.imwrite('images/photo_of_you_{}.jpg'.format(img_num), frame)
                cv.destroyWindow('Take photo number {}!'.format(img_num))
                img_num += 1
            elif key == ord('p'):
                cv.destroyWindow('Take photo number {}!'.format(img_num))
                processImages()
                cleanUpImgDir()
                face_id = None
                calibrated = False
                img_num = 1
            elif key == ord('q'):
                break
    else:
        break

# clean up faces_folder_path dir; delete user's photoshoot images
cleanUpImgDir()
# release video source and destroy all windows
cap.release()
cv.destroyAllWindows()