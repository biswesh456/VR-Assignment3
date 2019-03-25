import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from os import getcwd as pwd
import dlib
import cv2
import argparse
import imutils
import math

def getCategories():
    return [x[0][7:] for x in os.walk("./Data/")][1:]

def getData(path, category):
    data = {}

    for folder in category:
        folder_path = os.path.join(path, folder)
        images = []

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            ## read as grayscale image
            img = cv2.imread(file_path)
            images.append(img)
        data[folder] = images

    return data

def getEye(shape, dtype="int"):
    eye1 = np.zeros((4, 2), dtype=dtype)
    eye2 = np.zeros((4, 2), dtype=dtype)
    ## 37 to 42 are detectors of right eye
    eye1[0] = (shape.part(38).x, shape.part(38).y)
    eye1[1] = (shape.part(39).x, shape.part(39).y)
    eye1[2] = (shape.part(40).x, shape.part(40).y)
    eye1[3] = (shape.part(41).x, shape.part(41).y)
    ## 43 to 48 are detectors of left eye
    eye2[0] = (shape.part(44).x, shape.part(44).y)
    eye2[1] = (shape.part(45).x, shape.part(45).y)
    eye2[2] = (shape.part(46).x, shape.part(46).y)
    eye2[3] = (shape.part(47).x, shape.part(47).y)

    return eye1, eye2

def detectEyeNoseCenters(rects, img, predictor):
    ## there maybe multiple faces detected
    for (i, rect) in enumerate(rects):
        shape = predictor(img, rect)
        eye1, eye2 = getEye(shape)
        e1x = 0; e1y = 0
        e2x = 0; e2y = 0
        ## take average of all four points
        for (x, y) in eye1:
            e1x = e1x + x
            e1y = e1y + y

        for (x, y) in eye2:
            e2x = e2x + x
            e2y = e2y + y

        e1Center = (int(e1x/4), int(e1y/4))
        e2Center = (int(e2x/4), int(e2y/4))
        nose = (shape.part(31).x, shape.part(31).y)

        # cv2.imshow("1", img)
        # cv2.waitKey(0)
        return [e1Center, e2Center, nose]

def getEyeNoseCenters(detector, predictor, data):
    rectData = dict()
    for k in data.keys():
        rectList = []
        for img in data[k]:
            ## detects the face first
            rects = detector(img, 1)
            ## determine the facial landmarks for the face region and then get
            ## coordinates of the eyes
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lst = detectEyeNoseCenters(rects, gray_img, predictor)
            rectList.append((img, lst))
        rectData[k] = rectList
    return rectData

def shiftAndRotate(data):
    for cat in data.keys():
        newImageList = []
        for lst in data[cat]:
            img = lst[0]
            if lst[1] != None:
                rightEye = lst[1][0]
                leftEye = lst[1][1]
                nose = lst[1][2]

                pts1 = np.float32([[rightEye[0],rightEye[1]], [leftEye[0],leftEye[1]],[nose[0],nose[1]]])
                pts2 = np.float32([[64,64],[192,64],[100,120]])

                M = cv2.getAffineTransform(pts1,pts2)
                newImage = cv2.warpAffine(img,M,(256,256))

                newImageList.append(newImage)

        data[cat] = newImageList

def saveImages(data):
    try:
        os.stat("./NewData")
    except:
        os.mkdir("./NewData")

    for cat in data.keys():
        count = 0
        try:
            os.stat(os.path.join("./NewData", cat))
        except:
            os.mkdir(os.path.join("./NewData", cat))

        for img in data[cat]:
            folderpath = os.path.join("./NewData", cat)
            filename = str(count)+".png"
            imgpath = os.path.join(folderpath, filename)
            cv2.imwrite(imgpath, img)
            count = count+1

if __name__ == "__main__":
    path = "./Data/"
    category = getCategories()
    ## data is a dictionary with categories as keys and image list as value
    data = getData(path, category)
    print("Detecting......")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("predictorModel.dat")
    ## data is a dictionary with categories as keys and list of tuple of
    ## image and their 2 eye centres and nose center as value
    data = getEyeNoseCenters(detector, predictor, data)
    print("Shifting and Rotating......")
    shiftAndRotate(data)
    saveImages(data)
