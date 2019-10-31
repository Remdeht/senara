import cv2
import numpy as np
from skimage.filters.thresholding import threshold_li


def get_boundary_li(img_digit):
    """Extracts a digit region via thresholding based on the Li thresholding method"""

    img_binary, value = create_binary(img_digit)
    img_binary = pad_template(img_binary)
    results = get_digit_boundary(img_binary)

    return results


def create_binary(image):
    """Creates a binary image via thresholding based on Li thresholding method"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    y, x = image.shape[:2]
    ratio = y / x
    img_size_x = 250
    img_size_y = int(ratio * 250)
    image = cv2.resize(image, (img_size_x, img_size_y))
    image = cv2.GaussianBlur(image, (11, 11), 0)

    image = image[50:img_size_y - 50, :]

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5, 5))
    image = clahe.apply(image)

    thresh_li = threshold_li(image)

    binary_li = image.copy()
    binary_li[binary_li > thresh_li] = 255
    binary_li[binary_li <= thresh_li] = 0

    return 255 - binary_li, image


def get_digit_boundary(binary):
    """Gets the boundary for a digit region based on the regions in the binary input image"""

    original_binary = binary.copy()

    height, width = binary.shape[:2]

    min_cnt_size = height * width * .02  # minimum area for the digit region

    kernel = np.ones((11, 11), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Mophological operation to get rid of small regions
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours = get_contours(binary, min_cnt_size)

    if len(contours) > 2:  # selects the two largest region contours if there multiple large regions found.
        areas = [cv2.contourArea(c) for c in contours]
        ind = np.argpartition(areas, -2)[-2:]
        contours = [contours[i] for i in ind]

    if len(contours) is 2:

        cnt_1 = contours[0]
        cnt_2 = contours[1]

        upper_cnt_binary = np.zeros((height, width), np.uint8)
        lower_cnt_binary = np.zeros((height, width), np.uint8)

        _, y_1, _, h_1 = cv2.boundingRect(cnt_1)
        _, y_2, _, h_2 = cv2.boundingRect(cnt_2)

        if y_2 + h_2 < y_1:  # Determines which is the top contour

            cv2.drawContours(upper_cnt_binary, [cnt_2], -1, 255, -1)
            cv2.drawContours(lower_cnt_binary, [cnt_1], -1, 255, -1)
            holes_upper = get_holes_contour(upper_cnt_binary, original_binary)
            holes_lower = get_holes_contour(lower_cnt_binary, original_binary)

            return [cnt_2, upper_cnt_binary, holes_upper, cnt_1, lower_cnt_binary, holes_lower]

        elif y_1 + h_1 < y_2:

            cv2.drawContours(upper_cnt_binary, [cnt_1], -1, 255, -1)
            cv2.drawContours(lower_cnt_binary, [cnt_2], -1, 255, -1)
            holes_upper = get_holes_contour(upper_cnt_binary, original_binary)
            holes_lower = get_holes_contour(lower_cnt_binary, original_binary)

            return [cnt_1, upper_cnt_binary, holes_upper, cnt_2, lower_cnt_binary, holes_lower]

        else:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]
            cnt_binary = np.zeros((height, width), np.uint8)
            cv2.drawContours(cnt_binary, [cnt], -1, 255, -1)
            holes = get_holes_contour(cnt_binary, original_binary)
            return [contours[0], cnt_binary, holes]

    elif len(contours) is 1:
        cnt_binary = np.zeros((height, width), np.uint8)
        cv2.drawContours(cnt_binary, [contours[0]], -1, 255, -1)
        holes = get_holes_contour(cnt_binary, original_binary)
        return [contours[0], cnt_binary, holes]

    else:
        return None


def contour_check(cnt, binary):
    """Checks whether the region is located in the central part of the image"""
    height, width = binary.shape[:2]
    x, y, w, h = cv2.boundingRect(cnt)
    box_area = (x + w) * (y + h)
    max_box_area = height * width * .9

    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])

    if not int(width) * .1 <= cx < int(width) * .9 or box_area > max_box_area:
        return False
    else:
        return True


def get_contours(binary, min_cnt_size):
    """Selects the the (region) contours within a binary image"""
    contours, hir = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) > min_cnt_size]
    contours = [c for c in contours if contour_check(c, binary) is True]
    return contours


def pad_template(img_binary):
    """Function that pads a binary image with zeros"""
    height, width = img_binary.shape[:2]
    padding = np.zeros((height + 20, width + 20), np.uint8)
    padding[10:10 + height, 10:10 + width] = img_binary

    return padding


def get_holes_contour(bin_cnt, bin_og):
    """Function that determines the amount of holes within the digit region"""

    mask_cnt = bin_cnt.copy()
    mask_cnt[mask_cnt == 255] = 1
    bin_holes = (255 - bin_og) * mask_cnt
    min_cnt_size = np.count_nonzero(bin_cnt) * .05

    contours, hir = cv2.findContours(bin_holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) > min_cnt_size]

    temp_mask = np.zeros(mask_cnt.shape[:2], np.uint8)

    if len(contours) is not 0:
        for cnt in contours:
            cv2.drawContours(temp_mask, [cnt], -1, 255, -1)

    return len(contours)
