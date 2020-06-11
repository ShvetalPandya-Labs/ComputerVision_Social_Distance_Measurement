import face_recognition
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2 as cv
import os
import sys
from threading import Thread
import math
import pandas as pd
import random

import PIL as Image
# import the necessary packages
from threading import Thread

from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as k
import h5py

from time import sleep
import numpy as np
import argparse
# from wide_resnet import WideResNet
from keras.utils.data_utils import get_file

# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import Queue

# otherwise, import the Queue class for Python 2.7
else:
    from Queue import Queue


class SlowVideoStream:
    def __init__(self, path):
        # Open a pointer to the video stream and start the FPS timer
        self.stream = cv.VideoCapture(path)
        self.fps = FPS().start()
        self.frame = np.asarray([])
        self.grabbed = True

    def VideoStreamRead(self, resize=True, width=700, rgbconvert=False, displaytext=True):
        # grab the frame from the threaded video file stream
        (self.grabbed, self.frame) = self.stream.read()
        if not self.grabbed:
            return
        if resize == True:
            # resize the frame and convert it to grayscale (while still
            # retaining three channels)
            self.frame = imutils.resize(self.frame, width)

        if rgbconvert == True:
            self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
            self.frame = np.dstack([self.frame, self.frame, self.frame])

        if displaytext == True:
            # display a piece of text to the frame (so we can benchmark
            # fairly against the fast method)
            cv.putText(self.frame, "Slow Method", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return self.frame


class FaceCV(object):
    """
    Singleton class for face recongnition task
    """
    CASE_PATH = ".\\pretrained_models\\haarcascade_frontalface_alt.xml"
    WRN_WEIGHTS_PATH = "/home/user/Desktop/Technical/2019_Github_Profile_Projects/Video_Opencv/weights.18-4.06.hdf5"

    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceCV, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=64):
        self.face_size = face_size
        self.model = WideResNet(face_size, depth=depth, k=width)()
        model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
        fpath = get_file('weights.18-4.06.hdf5',
                         self.WRN_WEIGHTS_PATH,
                         cache_subdir=model_dir)
        self.model.load_weights(fpath)


# Function To CROP image based on Face Locations: List of 4 Pixel Locations
def Crop_Image(frame, location, padding):
    face_locations_list = list(location)
    crop_img = frame[face_locations_list[0]:face_locations_list[2], \
               face_locations_list[3]:face_locations_list[1]]
    bbox = [face_locations_list[0], face_locations_list[3], \
            face_locations_list[2], face_locations_list[1]]
    face = frame[max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[0] - 1),
           max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[1] - 1)]

    cv.rectangle(frame, (face_locations_list[3], face_locations_list[0]), \
                 (face_locations_list[1], face_locations_list[2]), (255, 0, 0), 8)

    return crop_img, frame, bbox, face


def PrintShowAdvertisement(workspace, age, gender):
    advdatapath = os.path.join(workspace, 'Advertisement')
    if gender == 'Male':
        if age == 'Young':
            agegender = 'Young_Male'
        elif (age == 'Middle'):
            agegender = 'Middle_Male'
        else:
            agegender = 'Old_Male'
        advdir = os.path.join(advdatapath, agegender)
        print('****************** Advertisement : ', agegender, ' *******************************')
        random_file = random.choice(os.listdir(os.path.join(advdir)))
        print(random_file)
        print('****************************************************************')
        cv.imshow('MaleAdvImage', cv.imread(os.path.join(advdir, random_file)))
        cv.waitKey(100)
        cv.destroyWindow('MaleAdvImage')


    else:
        if age == 'Young':
            agegender = 'Young_Female'
        elif (age == 'Middle'):
            agegender = 'Middle_Female'
        else:
            agegender = 'Old_Female'
        advdir = os.path.join(advdatapath, agegender)
        print('****************** Advertisement : ', agegender, ' *******************************')
        random_file = random.choice(os.listdir(os.path.join(advdir)))
        print(random_file)
        print('****************************************************************')
        cv.imshow('FemaleAdvImage', cv.imread(os.path.join(advdir, random_file)))
        cv.waitKey(100)
        cv.destroyWindow('FemaleAdvImage')


# This function returns cropped Image for Age/Gender Classification:
def FaceDetection_CropFace(workspace, frame, framecount, malecounter, femalecounter, padding):
    padding = padding

    crop_img_list = []
    faces_locations = face_recognition.face_locations(frame)
    if faces_locations:
        print(len(faces_locations), ' Faces detected')
        male_framewiseagelistindex = []
        female_framewiseagelistindex = []
        for location in faces_locations:
            crop_img, frame, bbox, face = Crop_Image(frame, location, padding)

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))
            if gender == 'Male':
                malecounter += 1

            else:
                femalecounter += 1

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            if gender == 'Male':
                male_framewiseagelistindex.append(age)
            else:
                female_framewiseagelistindex.append(age)

            # print("Age Output : {}".format(agePreds))
            print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

            print()

            label = "{},{}".format(gender, age)
            cv.putText(frame, label, (bbox[1], bbox[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)

            # cv.imshow(crop_img)
            # cv.imshow("Frame", crop_img)
            cv.imshow("Frame", frame)
            cv.waitKey(1)
            SlowVideoObject.fps.update()
            crop_img_list.append(crop_img)

        male_framewiseage_series = pd.Series(male_framewiseagelistindex)
        female_framewiseage_series = pd.Series(female_framewiseagelistindex)

        if len(male_framewiseage_series) == 0:
            frequent_male_age = age
        else:
            frequent_male_age = male_framewiseage_series.value_counts().index[0]

        if len(female_framewiseage_series) == 0:
            frequent_female_age = age
        else:
            frequent_female_age = female_framewiseage_series.value_counts().index[0]

        print('framecount', framecount)
        print('malecounter', malecounter)
        print('Femalecounter', femalecounter)

        if (malecounter > femalecounter):
            PrintShowAdvertisement(workspace, frequent_male_age, 'Male')
        elif (femalecounter > malecounter):
            PrintShowAdvertisement(workspace, frequent_female_age, 'Female')
        else:
            PrintShowAdvertisement(workspace, frequent_male_age, 'Male')
            PrintShowAdvertisement(workspace, frequent_female_age, 'Female')

    return crop_img_list, malecounter, femalecounter


faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['Young', 'Young', 'Young', 'Young', 'Young', 'Middle', 'Middle', 'Old', 'Old']
genderList = ['Male', 'Female']

# Load network
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

# Load network
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('--input',
                    help='Path to input image or video file. Skip this argument to capture frames from a camera.')
args = parser.parse_args()

# Open a video file or an image file or a camera stream
video_input = args.input if args.input else 0
SlowVideoObject = SlowVideoStream(video_input)

# Workspace
WORKSPACE = os.path.dirname(os.path.realpath(__file__))
framecount = 0
MALECOUNTER = 0
FEMALECOUNTER = 0

while True:
    frame = SlowVideoObject.VideoStreamRead(resize=True, width=1000, displaytext=True)
    framecount += 1
    # if (framecount < 120) | (framecount > 140):
    #	continue
    # print('Frame Count : ', framecount)

    # For Old 3 Videos
    # crop_img_list, MALECOUNTER, FEMALECOUNTER = FaceDetection_CropFace(WORKSPACE, frame, framecount, MALECOUNTER, FEMALECOUNTER, padding = 20)

    # For new 2 Videos
    crop_img_list = FaceDetection_CropFace(WORKSPACE, frame, framecount, MALECOUNTER, FEMALECOUNTER, padding=20)
