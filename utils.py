import matplotlib.pyplot as plt
import cv2
import imutils
import numpy as np


def min_area_rectangle(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    # find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # draw the contours of c
    # print('c is: ',type(c))
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)  # turn into ints
    # rect2 = cv2.drawContours(image.copy(), [box], 0, (255,0,0), 2)

    return box,rect,c



