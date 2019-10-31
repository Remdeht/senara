# Creates digit templates with digit randomly placed on the template
import os
import re
import cv2
import shutil
import random
import numpy as np
from image_processing.constants import ROOT_DIR

DATA_DIR_BACKGROUNDS = ROOT_DIR + "\\data\\background"
backgrounds = os.listdir(DATA_DIR_BACKGROUNDS)

DATA_DIR_CLASSIFIERS = ROOT_DIR + "\\data\\templates"
digits = os.listdir(DATA_DIR_CLASSIFIERS)

DATA_DIR_RESULTS = ROOT_DIR + "\\data\\random_templates"


def create_background(rows, columns):
    """Creates a background to place the digits on"""
    background_number = random.randint(0, len(backgrounds) - 1)
    background = cv2.imread(str(DATA_DIR_BACKGROUNDS) + "\\" + backgrounds[background_number], 0)

    (values, counts) = np.unique(background, return_counts=True)
    ind = np.argmax(counts)

    base_array = np.tile(values[ind], (rows, columns))

    avg = values[ind]
    minimum = np.min(background)
    minimum = int(minimum) - int(avg)
    maximum = np.max(background)
    maximum = int(maximum) - int(avg)

    base_array = np.asarray(base_array + np.random.randint(minimum, maximum, (rows, columns)), dtype=np.uint8)
    base_array = cv2.GaussianBlur(base_array, (15, 15), 3)

    return base_array


def create_binary(img_digit):
    """Creates a binary image """
    img_digit_hsv = cv2.cvtColor(img_digit, cv2.COLOR_BGR2HSV)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
    channels = cv2.split(img_digit_hsv)
    img_digit_value = clahe.apply(channels[2])

    img_digit_value_normalized = cv2.normalize(img_digit_value, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    img_digit_binary = cv2.inRange(img_digit_value_normalized, -1, .5)

    return img_digit_binary


def find_contour_digit(img_binary):
    """Get the biggest region from the binary image"""
    contours, hir = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = [contours[max_index]]

    x, y, w, h = cv2.boundingRect(contours[max_index])

    # Create mask to draw the contour on
    (W, H) = img_binary.shape[:2]
    cnt_canvas = np.zeros((W, H), np.uint8)
    cv2.drawContours(cnt_canvas, cnt, -1, 1, -1)

    return cnt_canvas, x, y, w, h


def create_digit_masks(img_digit):
    """Creates a mask of the digit"""
    img_digit_binary = create_binary(img_digit)
    img_digit_gray = cv2.cvtColor(img_digit, cv2.COLOR_BGR2GRAY)

    mask_digit, x, y, w, h = find_contour_digit(img_digit_binary)
    img_digit_gray *= mask_digit

    img_digit_gray[img_digit_gray > 250] = 0
    mask_digit *= img_digit_gray
    mask_digit[mask_digit > 0] = 1

    return mask_digit[y:y + h, x:x + w], img_digit_gray[y:y + h, x:x + w]


def place_digits(background, H, W):
    """Randomly places the digits on a background"""
    # Digit selection
    random_number = random.randint(0, len(digits) - 1)  # generate random int
    img_loc = str(DATA_DIR_CLASSIFIERS) + "\\" + digits[random_number]  # use random int to load a template
    img_first_digit = cv2.imread(img_loc)
    img_first_digit = cv2.resize(img_first_digit, (H, W))

    mask_first_digit, img_first_digit_gray = create_digit_masks(img_first_digit)  # create masks of the digit

    # Addition of the second digit based on the digit that was chosen before
    pattern = r"([0-9]).*.jpg"
    name = re.findall(pattern, digits[random_number])  # digit is specified in the file name

    if int(name[0]) is 9:
        second_digit = 0
    else:
        second_digit = int(name[0]) + 1

    img_loc = str(DATA_DIR_CLASSIFIERS) + "\\" + str(second_digit) + ".jpg"
    img_second_digit = cv2.imread(img_loc)
    img_second_digit = cv2.resize(img_second_digit, (H, W))

    mask_second_digit, img_second_digit_gray = create_digit_masks(img_second_digit)  # create masks of the digit

    # Use both maks and images of the digits to combine them in two arrays comparable to the representation in the \
    # tally counter

    height_first_digit, width_first_digit = mask_first_digit.shape[:2]
    height_second_digit, width_second_digit = mask_second_digit.shape[:2]

    # Resize the digit so that they have equal heights
    ratio_between_digits = height_first_digit / height_second_digit
    height_second_digit = height_first_digit
    width_second_digit *= ratio_between_digits
    width_second_digit = int(width_second_digit)

    mask_second_digit = cv2.resize(mask_second_digit, (width_second_digit, height_second_digit)) * 0
    img_second_digit_gray = cv2.resize(img_second_digit_gray, (width_second_digit, height_second_digit)) * 0

    # Pad the smallest array with zeros for the concatenation of arrays
    if width_first_digit is width_second_digit:
        buffer = np.zeros((int(height_first_digit * .25), width_first_digit), np.uint8)  # Spacing between digits
    elif width_first_digit > width_second_digit:
        padding_mask = np.zeros((height_first_digit, width_first_digit), np.uint8)
        padding_img = np.zeros((height_first_digit, width_first_digit), np.uint8)
        buffer = np.zeros((int(height_first_digit * .25), width_first_digit), np.uint8)
        if np.abs(width_first_digit - width_second_digit) > 2:
            start_pos = int((np.abs(width_first_digit - width_second_digit) / 2) - 1)
        else:
            start_pos = 0

        # Addition of the smallest digit arrays in the center of the padding arrays
        padding_mask[:, start_pos:start_pos + width_second_digit] = mask_second_digit
        mask_second_digit = padding_mask
        padding_img[:, start_pos:start_pos + width_second_digit] = img_second_digit_gray
        img_second_digit_gray = padding_img

    else:
        padding_mask = np.zeros((height_second_digit, width_second_digit), np.uint8)
        padding_img = np.zeros((height_second_digit, width_second_digit), np.uint8)
        buffer = np.zeros((int(height_second_digit * .25), width_second_digit), np.uint8)
        if np.abs(width_first_digit - width_second_digit) > 2:
            start_pos = int((np.abs(width_first_digit - width_second_digit) / 2) - 1)
        else:
            start_pos = 0

        padding_mask[:, start_pos:start_pos + width_first_digit] = mask_first_digit
        mask_first_digit = padding_mask
        padding_img[:, start_pos:start_pos + width_first_digit] = img_first_digit_gray
        img_first_digit_gray = padding_img

    # Concatenation of the array containing the digit arrays
    mask_digits = np.concatenate([mask_first_digit, buffer, mask_second_digit])
    img_digits = np.concatenate([img_first_digit_gray, buffer, img_second_digit_gray])

    # Placement of the digits on the generated background
    height, width = img_digits.shape[:2]
    height_background, width_background = background.shape[:2]

    # random selection of x and y for the digits to be placed
    random_x = random.randint(-int(width_background * .5), int(width_background * .5))
    random_y = random.randint(-int(height_background * .5), int(height_background* .75))

    if random_y < 0:  # if the random y is below 0 part of the top of the digits is cut off
        mask_digits = mask_digits[-random_y:height, :]
        img_digits = img_digits[-random_y:height, :]
        height, width = img_digits.shape[:2]
        random_y = 0

    if random_x < 0:
        mask_digits = mask_digits[:, -random_x:width]
        img_digits = img_digits[:, -random_x:width]
        height, width = img_digits.shape[:2]
        random_x = 0

    if random_x + width <= width_background and random_y + height <= height_background:
        background[random_y:random_y + height, random_x:random_x + width] = background[random_y:random_y + height,
                                                                            random_x:random_x + width]\
                                                                            * (1 - mask_digits)
        background[random_y:random_y + height, random_x:random_x + width] += img_digits

    elif random_x + width > width_background and random_y + height > height_background:
        background[
        random_y: height_background,
        random_x: width_background] = background[
                                      random_y: height_background,
                                      random_x: width_background] * (1 - mask_digits)[
                                                                    0:(height_background - random_y),
                                                                    0:(width_background - random_x)
                                                                    ]

        background[random_y:height_background, random_x:width_background] += img_digits[
                                                                             0:(height_background - random_y),
                                                                             0:(width_background - random_x)]
    elif random_x + width > width_background:
        background[random_y:random_y + height, random_x:width_background] = background[
                                                                            random_y:random_y + height,
                                                                            random_x:width_background
                                                                            ] * (1 - mask_digits)[
                                                                                :,
                                                                                0:(width_background - random_x)
                                                                                ]
        background[random_y:random_y + height, random_x:width_background] += img_digits[
                                                                             :, 0:(width_background - random_x)]
    else:
        background[random_y:height_background, random_x:random_x + width] = background[
                                                                            random_y:height_background,
                                                                            random_x:random_x + width
                                                                            ] * (1 - mask_digits)[
                                                                                0:(height_background - random_y),
                                                                                :
                                                                                ]
        background[random_y:height_background, random_x:random_x + width] += img_digits[
                                                                             0:(height_background - random_y),
                                                                             :]

    background = cv2.GaussianBlur(background, (3, 3), 3)
    return background, name


if __name__ == '__main__':
    # Run the script to generate the templates
    Y = 250
    X = 150

    if os.path.exists(DATA_DIR_RESULTS):
        shutil.rmtree(DATA_DIR_RESULTS)
        os.mkdir(DATA_DIR_RESULTS)
    else:
        os.mkdir(DATA_DIR_RESULTS)

    for i in range(100):
        base_array = create_background(Y, X)
        result, digit = place_digits(base_array, int(X * .75), int(Y * .75))

        cv2.imwrite(str(DATA_DIR_RESULTS) + "//" + digit[0] + "_iteration_" + str(i) + ".jpg", result)
