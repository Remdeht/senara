import cv2
import math
import numpy as np

# General functions

def order_rectangle_coordinates(box):
    """ Orders the coordinates of a rectagle to go clockwise"""

    # Code found at: https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
    # the method is based on a function of the imutils library, however some adjustments have made as the method did not
    # function correctly

    # sort the points based on their x-coordinates
    box.sort(key=lambda x: x[0])
    xSorted = box

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[0:2]
    rightMost = xSorted[2:4]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost.sort(key=lambda x: x[1])
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point

    rightMost.sort(key=lambda x: x[1])
    (tr, br) = rightMost

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl])


def crop_to_contour(cnt, img):
    """Crops an image to a contour"""
    x, y, w, h = cv2.boundingRect(cnt)
    return img[x: x + w, y:y + h]


def distance(p0, p1):
    """ Calculates the distance between two points """
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def fit_rectangle(cnt):
    """ Fits a rectangle around a contour """

    rect = cv2.minAreaRect(cnt[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    return box


def crop_image(image, mask_dial, x_cap, y_cap, r_dial, mask_cap):
    """Crops the original image and mask to the dial and adjusts the center coordinates of the dial"""

    img_dial = image * mask_dial
    img_dial = crop_to_circle(x_cap, y_cap, r_dial, img_dial)
    mask_cap = crop_mask_to_circle(x_cap, y_cap, r_dial, mask_cap)
    height, width = img_dial.shape[:2]
    x_cap = int(width / 2)
    y_cap = int(height / 2)

    return img_dial, mask_cap, x_cap, y_cap
