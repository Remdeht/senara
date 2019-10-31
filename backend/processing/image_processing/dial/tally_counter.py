# Tally Counter Extraction
# Libraries
import numpy as np
import cv2
import math
import imutils

# Scripts
from .helpers import general_functions, adaptive_gamma_correction
from .fourier import classify


def extract_counter(img, fd, n, flag_knn, flag_templates):
    """Main function from which all the subprocesses are started and handled."""

    img_counter = clip_tally_area(img)  # Clip area based on center cap and red pointer location
    # img_counter = img_counter.copy()

    tally_height, tally_width = calculate_expected_tally(img)
    box = find_tally_contour(img_counter, tally_height,
                             tally_width)  # Search for the contour of the tally counter based on Value & Hue

    if box is None:
        return None, None, None

    # Fit rectangle to contour and divide in 6 areas
    # Get longest side determine the lowest left point of the rectangle and rotate the image so that
    # the rectangle is horizontal. Subsequently the tally is split up in six equal parts, one part per number.

    side_1 = general_functions.distance(box[0], box[1])
    side_2 = general_functions.distance(box[0], box[3])

    (y, x) = img.shape[:2]
    x_orig = int(int(x * .5) * .5)  # X,Y coordinates of the tally cutout relative to the main image
    y_orig = int(int(y * .5) * 1.5)  # These are used to calculate the tally counter coord when the image is shifted

    if side_1 >= side_2:

        side_A = general_functions.distance(box[1], (box[1][0], box[0][1]))
        side_C = side_1

        if box[0][1] < box[1][1]:
            rotated, rotated_img, rotation = rotate_counter(side_A, side_C, img_counter, img, 1)
        else:
            rotated, rotated_img, rotation = rotate_counter(side_A, side_C, img_counter, img, 0)

        box_transformed = []
        for point in box:
            point = [x_orig + point[0], point[1] + y_orig]
            recalculated_point = recalculate_coordinates(img, point, rotation)
            box_transformed.append(recalculated_point)

        center_tally = ((box_transformed[0][0] + int(side_1 * .5)), box_transformed[0][1])

        num1, num2, num3, num4, num5, num6 = divide_tally_counter(box_transformed, rotated_img)
        digits = [num1, num2, num3, num4, num5, num6]
        tally_value = []

        for index,digit in enumerate(digits):
            if digit is not None and len(digit) > 1 and digit.shape[1] is not 0:
                result, avg_distance, accuracy = classify.main_operation(digit, fd, n, flag_knn, flag_templates)
                tally_value.append([result, avg_distance, accuracy])

        return rotated_img, center_tally, tally_value


def calculate_expected_tally(img):
    "Calculates the expected size of the tally counter based on the radius of the dial"
    H, W = img.shape[:2]

    if H is W:
        r = W / 2
    elif H > W:
        r = H / 2
    else:
        r = W / 2

    tally_width = 0.65 * r
    tally_height = tally_width * .25

    return tally_height, tally_width


def clip_tally_area(img):
    """Clips and are where the tally is supposed to be located based on our knowledge of the dial."""

    height, width = img.shape[:2]

    center_dial = (int(width / 2), int(height / 2))

    left = int(center_dial[0] * .5)  # Coordinates for the tally area are calculated
    right = int(center_dial[0] * 1.5)
    high = int(center_dial[1] * 1.5)
    low = int(center_dial[1] * 1.9)

    img_counter = img[high:low, left:right]  # Slices the original image to the Tally area

    return img_counter


def find_tally_contour(img_counter, tally_height, tally_width):
    """This method finds the contour of the tally counter based on the Hue and Value of the image"""

    img_counter_blurred = cv2.GaussianBlur(img_counter, (21, 21), 0)  # Smooth the image

    img_counter_hsv = cv2.cvtColor(img_counter_blurred, cv2.COLOR_BGR2HSV)  # Convert to the HSV color space
    channels = cv2.split(img_counter_hsv)  # Split Channels
    img_counter_value = adaptive_gamma_correction.agc_1c(channels[2])  # Value

    img_counter_yuv = cv2.cvtColor(img_counter_blurred, cv2.COLOR_BGR2YUV)
    channels = cv2.split(img_counter_yuv)
    u = cv2.equalizeHist(channels[1])

    binary = cv2.inRange(u, np.mean(u) + (.75 * np.std(u)), 255)
    bound = np.mean(img_counter_value) - (2 * np.std(img_counter_value))

    # Creation of Binary image and masks based on the Hue and Value of the image
    mask_value = cv2.inRange(img_counter_value, 0, bound)
    ret, mask_value = cv2.threshold(mask_value, .5, 1, cv2.THRESH_BINARY_INV)

    img_binary_counter = binary * mask_value  # Binary Image for Contour Detection

    expected_tally_area = tally_width * tally_height

    contour_area, cnt = get_biggest_contour(img_binary_counter)
    cnt = check_contour(contour_area, cnt, expected_tally_area)

    if cnt is None:
        return None

    if cnt is 0:
        cnt = closing(img_binary_counter, expected_tally_area)
        if cnt is 1:
            cnt = opening(img_binary_counter, expected_tally_area)
            if cnt is None or cnt is 1 or cnt is 0:
                return None
        elif cnt is None or cnt is 0:
            return None

    elif cnt is 1:
        cnt = opening(img_binary_counter, expected_tally_area)
        if cnt is 0:
            cnt = closing(img_binary_counter, expected_tally_area)
            if cnt is None or cnt is 1 or cnt is 0:
                return None
        elif cnt is None or cnt is 1:
            return None

    box = general_functions.fit_rectangle(cnt)
    box = general_functions.order_rectangle_coordinates(box.tolist())
    height_difference = int(abs(general_functions.distance(box[0], box[3]) - tally_height))

    if height_difference > tally_height * .1:

        changed_crd_lower_l = [box[0][0], box[0][1] + height_difference]
        changed_crd_lower_2 = [box[1][0], box[1][1] + height_difference]
        box_lower = np.int0([changed_crd_lower_l, changed_crd_lower_2, box[2], box[3]])

        mask_lower = np.zeros((img_binary_counter.shape[:2]), np.uint8)
        cv2.drawContours(mask_lower, [box_lower], 0, 1, -1)

        changed_crd_upper_l = [box[2][0], box[2][1] - height_difference]
        changed_crd_upper_2 = [box[3][0], box[3][1] - height_difference]
        box_upper = np.int0([box[0], box[1], changed_crd_upper_l, changed_crd_upper_2])

        mask_upper = np.zeros((img_binary_counter.shape[:2]), np.uint8)
        cv2.drawContours(mask_upper, [box_upper], 0, 1, -1)

        contour_masked_upper = mask_upper * img_binary_counter
        contour_masked_lower = mask_lower * img_binary_counter

        if np.count_nonzero(contour_masked_upper == 255) <= np.count_nonzero(contour_masked_lower == 255):
            box = box_lower
        else:
            box = box_upper

    width_difference = int(abs(general_functions.distance(box[0], box[1]) - tally_width))
    if width_difference > tally_width * .1:

        changed_crd_left_l = [box[0][0] + width_difference, box[0][1]]
        changed_crd_left_2 = [box[3][0] + width_difference, box[3][1]]
        box_right = np.int0([changed_crd_left_l, box[1], box[2], changed_crd_left_2])

        mask_right = np.zeros((img_binary_counter.shape[:2]), np.uint8)
        cv2.drawContours(mask_right, [box_right], 0, 1, -1)

        changed_crd_right_l = [box[1][0] - width_difference, box[1][1]]
        changed_crd_right_2 = [box[2][0] - width_difference, box[2][1]]
        box_left = np.int0([box[0], changed_crd_right_l, changed_crd_right_2, box[3]])

        mask_left = np.zeros((img_binary_counter.shape[:2]), np.uint8)
        cv2.drawContours(mask_left, [box_left], 0, 1, -1)

        contour_masked_right = mask_right * img_binary_counter
        contour_masked_left = mask_left * img_binary_counter

        if np.count_nonzero(contour_masked_right == 255) <= np.count_nonzero(contour_masked_left == 255):
            box = box_left
        else:
            box = box_right

    box = general_functions.order_rectangle_coordinates(box.tolist())
    box = np.int0(box)

    return box


def check_contour(area, cnt, expected_tally_area):

    box = general_functions.order_rectangle_coordinates(general_functions.fit_rectangle(cnt).tolist())
    box_area = general_functions.distance(box[0], box[1]) * general_functions.distance(box[0], box[3])

    if box_area > expected_tally_area * 1.5:
        return 1
    if expected_tally_area * .75 < area < expected_tally_area * 1.2:
        return cnt
    elif expected_tally_area * .9 < area:
        return 1
    else:
        return 0


def get_biggest_contour(img_binary):
    contours, hir = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Contour Detection
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    area = areas[max_index]
    cnt = [contours[max_index]]

    return area, cnt


def opening(img_binary_counter, expected_tally_area):
    kernel = np.ones((3, 3), np.uint8)
    img_binary_counter = cv2.erode(img_binary_counter, kernel, iterations=1)
    contour_area, cnt = get_biggest_contour(img_binary_counter)
    cnt = check_contour(contour_area, cnt, expected_tally_area)
    i = 0
    while cnt is 1 and i < 5:
        i += 1
        img_binary_counter = cv2.erode(img_binary_counter, kernel, iterations=1)
        contour_area, cnt = get_biggest_contour(img_binary_counter)
        cnt = check_contour(contour_area, cnt, expected_tally_area)

    return cnt


def closing(img_binary_counter, expected_tally_area):
    kernel = np.ones((7, 7), np.uint8)
    img_binary_counter = cv2.dilate(img_binary_counter, kernel, iterations=1)
    contour_area, cnt = get_biggest_contour(img_binary_counter)
    cnt = check_contour(contour_area, cnt, expected_tally_area)
    i = 0
    while cnt is 0 and i < 5:
        i += 1
        img_binary_counter = cv2.dilate(img_binary_counter, kernel, iterations=1)
        contour_area, cnt = get_biggest_contour(img_binary_counter)
        cnt = check_contour(contour_area, cnt, expected_tally_area)

    return cnt


def rotate_counter(side_A, side_C, img_counter, img, flag):
    """ Rotates the rectangle that was fitted around the contour in order to make it horizontal """

    sine_alfa = side_A / side_C
    rotation = np.degrees(np.arcsin(sine_alfa))

    if flag is 0:  # Depending on the orientation of the rectangle the orientation of the rotation is changed
        rotation = -rotation

    rotated = imutils.rotate(img_counter, rotation)  # Both the image of the counter as the original image are rotated
    rotated_img = imutils.rotate(img, rotation)

    return rotated, rotated_img, rotation


def divide_tally_counter(box_transformed, rotated):
    """Divides the tally counter area into six equal parts and returns them individually"""

    box_transformed = np.asarray(general_functions.order_rectangle_coordinates(box_transformed))

    l = min([points[0] for points in box_transformed])  # The extremities of the tally area are selected
    r = max([points[0] for points in box_transformed])
    u = min([points[1] for points in box_transformed])
    d = max([points[1] for points in box_transformed])

    extremities = [l, r, u, d]

    for ext in range(len(extremities)):  # Replaces a negative value with 0 in the rare case that this occurs
        if extremities[ext] < 0:
            extremities[ext] = 0

    l, r, u, d = extremities

    dif = r - l
    width_num = int(dif / 6)

    num1 = rotated[u:d, l: l + width_num]  # The original contour area is divided into six equal parts
    num2 = rotated[u:d, l + width_num: l + (width_num * 2)]
    num3 = rotated[u:d, l + (width_num * 2): l + (width_num * 3)]
    num4 = rotated[u:d, l + (width_num * 3): l + (width_num * 4)]
    num5 = rotated[u:d, l + (width_num * 4): l + (width_num * 5)]
    num6 = rotated[u:d, l + (width_num * 5): r]

    return num1, num2, num3, num4, num5, num6


def recalculate_coordinates(original_image, point, rotation):
    """ Recalculates the coordinates based on the rotation"""

    height, width = original_image.shape[:2]
    center_img = (int(width / 2), int(height / 2))

    x_temp = point[0] - center_img[0]
    y_temp = point[1] - center_img[1]

    x_transformed = center_img[0] + x_temp * math.cos(math.radians(-rotation)) - y_temp * math.sin(
        math.radians(-rotation))
    y_transformed = center_img[1] + x_temp * math.sin(math.radians(-rotation)) - y_temp * math.cos(
        math.radians(-rotation)) * -1

    return [int(x_transformed), int(y_transformed)]
