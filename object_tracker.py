'''
Blue tracking cdoe
https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Changing_ColorSpaces_RGB_HSV_HLS.php
Live video capture
https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html
HSV value color chart
https://newbedev.com/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-with-cv-inrange-opencv
Contours and bounding box
https://docs.opencv.org/4.5.3/dd/d49/tutorial_py_contour_features.html
'''

import cv2
import numpy as np

#cap = cv2.VideoCapture('test/blue_vid.mp4')
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

while(1):

    # Take each frame
    _, frame = cap.read()
    frame = rescale_frame(frame, percent=40)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV (6.1)
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    '''
    # color picker thresolds (6.3)
    lower_blue = np.array([115,200,100])
    upper_blue = np.array([125,255,150])
    '''

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    '''
    # Mask based on RBG values (6.1)
    lower_bound = np.array([90,0,0])
    upper_bound = np.array([255,60,60])
    mask = cv2.inRange(frame, lower_bound, upper_bound)
    '''

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # draw contour lines
    #cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    # find/draw bounding box
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('frame',frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()