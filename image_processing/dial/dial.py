# Detects and extracts the dial from the image
# libraries needed for image processing
import os
import cv2
import math
import numpy as np
from .helpers import adaptive_gamma_correction

# root directory
from image_processing.constants import ROOT_DIR


def detect_dial(img, name=None):
    """ Main function of the module that extracts the dial of the water meter from the image"""

    if name is not None:  # Create folder for writing intermediate processing results
        if not os.path.exists(ROOT_DIR + '\\results\\image_processing\\dial'):
            os.makedirs(ROOT_DIR + '\\results\\image_processing\\dial')

    img_orig = np.copy(img)  # original input image
    img_output = np.copy(img)  # copy used for the output

    img_binary = create_binary(img_orig, name)  # create a binary for the contour detection
    img_dial, mask = find_contours_binary(img_binary, img_output)

    mask_dial, mask_cap, x_cap, y_cap, r_dial = find_dial(img_orig, img_binary, mask, name)

    return mask_dial, mask_cap, x_cap, y_cap, r_dial


def create_binary(img, name=None):
    """ Creates a binary that is used to find the contour of the dial of the Water Meter"""

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to Gray

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to Hue, Saturation, Lightness
    channels = cv2.split(img_hsv)  # extract the lightness channel

    img_gray_gamma = adaptive_gamma_correction.agc_1c(img_gray)  # Apply adaptive gamma correction

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Contrast Limited Adaptive Histogram Equalization
    img_value = clahe.apply(channels[2])  # Separate the value

    ret, mask_value = cv2.threshold(img_value, np.mean(img_value), 1, cv2.THRESH_BINARY)

    img_gray_equalized = img_gray_gamma * mask_value  # Mask gray image based on the value

    img_equalized_blurred = cv2.GaussianBlur(img_gray_equalized, (51, 51), 0)  # Apply a Gaussian Blur
    img_equalized_blurred = np.uint8(img_equalized_blurred * 255)
    _, img_binary = cv2.threshold(img_equalized_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Adaptive Bin

    if name is not None:
        if not os.path.exists(ROOT_DIR + '\\results\\image_processing\\dial\\binary'):
            os.makedirs(ROOT_DIR + '\\results\\image_processing\\dial\\binary')

        cv2.imwrite(ROOT_DIR + '\\results\\image_processing\\dial\\binary\\' + name, img_binary)

    return img_binary


def find_dial(img_dial, img_binary, mask, name=None):
    """Creates a mask of the dial area based on the position and size of the black cap"""

    img_output = img_dial.copy()
    mask_value = (255 - img_binary) * mask
    mask_value, x, y, r = find_contours_cap(mask_value, name)

    estimated_radius_dial = int(r * 2.55)
    mask_dial = create_mask_dial_rgb(x, y, estimated_radius_dial, img_output)

    return mask_dial, mask_value, x, y, estimated_radius_dial


def create_mask_dial(x, y, r, img_hue):
    """Creates as mask to use on a one channel image"""
    height, width = img_hue.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.circle(mask, (x, y), r, 1, -1)

    return mask


def create_mask_dial_rgb(x, y, r, img_hue):
    """Creates as mask to use on a three channel RGB image"""

    height, width, depth = img_hue.shape

    mask = np.zeros((height, width, depth), np.uint8)
    cv2.circle(mask, (x, y), r, (1, 1, 1), -1)

    return mask


def find_contours_cap(img_binary, name=None):
    """Returns a mask of the dial, the center of the dial coordinates and the radius of the dial"""

    contours, hir = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) is 0:
        raise ValueError("Unable to detect dial")

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)

    height, width = img_binary.shape

    if np.max(areas) > (height * width * .5):
        del contours[max_index]
        areas = [cv2.contourArea(c) for c in contours]
        if len(areas) is 0:
            raise ValueError("Unable to detect dial")
        max_index = np.argmax(areas)

    cnt = contours[max_index]

    mask = np.zeros((height, width), np.uint8)

    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    distance_from_com = []

    for point in cnt:
        dist = math.hypot(cx - point[0][0], cy - point[0][1])
        distance_from_com.append(dist)

    average_dist = np.mean(distance_from_com)

    cv2.drawContours(mask, [cnt], -1, 255, -1)

    img_binary_cap = mask.copy()

    H, W = mask.shape[:2]
    mask_handlebars = np.zeros((H,W), np.uint8)
    cv2.circle(mask_handlebars, (cx, cy), int(average_dist * .9), 1, -1)

    mask *= mask_handlebars  # Removes the handlebars and pointer from the cap contour

    contours, hir = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Recalculate the center of mass
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    contour_area = cv2.contourArea(cnt)

    estimated_radius_cap = int(math.sqrt(contour_area / math.pi))  # estimates cap radius based on a area of contour

    if estimated_radius_cap < width * .05:  # Throws Exception in case the radius is too small
        raise ValueError("Not possible to extract dial from the image!")

    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    if name is not None:  # Writes the intermediary results to the results folder
        if not os.path.exists(ROOT_DIR + '\\results\\image_processing\\dial\\cap_detection'):
            os.makedirs(ROOT_DIR + '\\results\\image_processing\\dial\\cap_detection\\inverse_dial_bin')
            os.makedirs(ROOT_DIR + '\\results\\image_processing\\dial\\cap_detection\\cap_region')
            os.makedirs(ROOT_DIR + '\\results\\image_processing\\dial\\cap_detection\\cap_region_corrected')

        cv2.imwrite(ROOT_DIR + '\\results\\image_processing\\dial\\cap_detection\\inverse_dial_bin\\' + name, img_binary)
        cv2.imwrite(ROOT_DIR + '\\results\\image_processing\\dial\\cap_detection\\cap_region\\' + name, img_binary_cap)
        cv2.imwrite(ROOT_DIR + '\\results\\image_processing\\dial\\cap_detection\\cap_region_corrected\\' + name, mask)

    return mask, cx, cy, estimated_radius_cap


def find_contours_binary(img_binary, img_output):
    """ Creates a mask of the rough shape of a Dial"""

    contours, hir = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) is 0:
        raise ValueError("Unable to detect dial")

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = [contours[max_index]]

    height, width = img_binary.shape
    mask = np.zeros((height, width, 3), np.uint8)
    mask_result = np.zeros((height, width), np.uint8)

    cv2.drawContours(mask, cnt, -1, (1, 1, 1), -1)
    cv2.drawContours(mask_result, cnt, -1, 1, -1)

    img_output = img_output * mask

    return img_output, mask_result

