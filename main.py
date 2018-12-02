#!/usr/bin/python

import sys
import os
import dlib
import glob
import cv2 as cv
import numpy as np
from math import sqrt

if len(sys.argv) != 4:
    print(
        "Call this program like this:\n"
        "   ./main.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat images\n"
        "You can download a trained facial shape predictor and recognition model from:\n"
        "    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n"
        "    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
    exit()

predictor_path = sys.argv[1]
face_rec_model_path = sys.argv[2]
faces_folder_path = sys.argv[3]

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

def euclDistance(face1, face2):
    result = face1 - face2
    result = result ** 2
    result = np.sum(result)
    return sqrt(result)

def drawBox(shape, match):
    global img

    color = (0,255,0) if match else (0,0,255)
    box_pt1 = (shape.rect.tl_corner().x, shape.rect.tl_corner().y)
    box_pt2 = (shape.rect.br_corner().x, shape.rect.br_corner().y)
    cv.rectangle(img, box_pt1, box_pt2, color)

def processImages():
    global face_id, img

    for f in glob.glob(os.path.join(faces_folder_path, '*.jpg')):
        img = dlib.load_rgb_image(f)

        dets = detector(img, 1)

        shapes = []
        matches = []
        for d in dets:
            shape = sp(img, d)
            shapes.append(shape)

            face_id2 = np.array(facerec.compute_face_descriptor(img, shape))
            if euclDistance(np.array(face_id), face_id2) < 0.55: # 0.47
                matches.append(True)
            else:
                matches.append(False)

        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        for i, box in enumerate(shapes):
            drawBox(box, matches[i])
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

def computeFaceId(img):
    global face_id

    dets = detector(img, 0)
    if len(dets) != 1:
        face_id = None
    else:
        shape = sp(img, dets[0])
        face_id = facerec.compute_face_descriptor(img, shape, 100) # 50

def cleanUpImgDir():
    for f in glob.glob(os.path.join(faces_folder_path, 'photo_of_you_*')):
        os.remove(f)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 60.0)

calibrated = False
img_num = 1

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
        else:
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

cleanUpImgDir()
cap.release()
cv.destroyAllWindows()