
import os
import cv2
import numpy as np
import sys


def mouseHandler(event,x,y,flags,param):
    global im_temp, pts_dst

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(im_temp,(x,y),3,(0,255,255),5,cv2.LINE_AA)
        cv2.imshow("Image", im_temp)
        if len(pts_dst) < 4:
            pts_dst = np.append(pts_dst,[(x,y)],axis=0)


current_dir_path = os.getcwd()

WORKSPACE = os.path.dirname(os.path.realpath(__file__))
DATA_LOCATION = os.path.join('/Users/roambee/Desktop/Technical/2019_Github_Profile_Projects/'
                             'Social_Distance_Detection/sddetector/data/')
FILENAME = 'TownCentreImage.png'
IMAGE_1 = os.path.join(DATA_LOCATION, FILENAME)

"""
cv2.imshow("Original image", img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray_image", gray)
cv2.waitKey(0)
"""

# Read in the image.
im_src = cv2.imread(IMAGE_1)
height, width = im_src.shape[:2]
print(height, width)

# Create a list of points.
pts_src = np.empty((0,2),dtype=np.int32)
pts_src = np.append(pts_src, [(0,0)], axis=0)
pts_src = np.append(pts_src, [(width-1,0)], axis=0)
pts_src = np.append(pts_src, [(width-1,height-1)], axis=0)
pts_src = np.append(pts_src, [(0,height-1)], axis=0)

print(pts_src)

# Destination image
FILENAME = 'screenshot.png'
IMAGE_2 = os.path.join(DATA_LOCATION, FILENAME)
im_dst = cv2.imread(IMAGE_2)

# Create a window
cv2.namedWindow("Image", 1)

im_temp = im_dst
pts_dst = np.empty((0, 2), dtype=np.int32)

cv2.setMouseCallback("Image", mouseHandler)


cv2.imshow("Image", im_temp)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)

tform, status = cv2.findHomography(pts_src, pts_dst)
im_temp = cv2.warpPerspective(im_src, tform,(width,height))
cv2.imshow("Temp Image", im_dst)

cv2.fillConvexPoly(im_dst, pts_dst, 0, cv2.LINE_AA)
im_dst = im_dst + im_temp

cv2.imshow("Destination Image", im_dst)
cv2.waitKey(0)