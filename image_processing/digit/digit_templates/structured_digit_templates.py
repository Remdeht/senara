import cv2
import numpy as np
import os
import re
import shutil
from image_processing.constants import ROOT_DIR

DATA_DIR_RESULTS = ROOT_DIR + "\\data\\structured_templates"
DATA_DIR_BASE_TEMPLATES = ROOT_DIR + "\\data\\templates"
DATA_DIR_BASE_TEMPLATES_2 = ROOT_DIR + "\\data\\templates_2"


def get_biggest_contour(contours):
    """Function that gets the biggest contour area from an array of contours"""

    if contours is None or len(contours) == 0:
        return 0

    areas = [cv2.contourArea(c) for c in contours]
    area = np.max(areas)

    return area


def pad_template(img_bin):
    """Function that pads a binary image with zeros"""
    H, W = img_bin.shape[:2]
    padding = np.zeros((H + 10, W + 10), np.uint8)
    padding[5:5 + H, 5:5 + W] = img_bin

    return padding



def split_template(digits, directory_path_base, version):
    """Splits the templates to make a set of feature templates for the classification"""

    for digit in digits:
        pattern = r"([0-9]).*.jpg"
        digit_ = re.findall(pattern, digit)[0]

        directory_path = directory_path_base + "\\" + digit_
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        if version is 1:
            img_loc = str(DATA_DIR_BASE_TEMPLATES) + "\\" + digit
        elif version is 2:
            img_loc = str(DATA_DIR_BASE_TEMPLATES_2) + "\\" + digit

        img_digit = cv2.imread(img_loc, 0)
        X = 75
        Y = int(X * 1.6)
        img_digit =  cv2.resize(img_digit, (X, Y))
        img_digit = cv2.GaussianBlur(img_digit, (7,7), 0)
        img_gray_bin = cv2.inRange(img_digit, 0, np.mean(img_digit)+10)
        img_gray_bin = pad_template(img_gray_bin)

        subclass_directory_path = directory_path + "\\Whole"
        if not os.path.exists(subclass_directory_path):
            os.makedirs(subclass_directory_path)
        cv2.imwrite(subclass_directory_path + "//" + digit_ + "_v" + str(version) + ".jpg", img_gray_bin)

        height, width = img_gray_bin.shape[:2]

        y_boundaries = np.arange(0, int(height * .66), 5)
        x_boundaries = np.arange(0, int(width * .66), 5)

        for y_b in y_boundaries:
            upper = img_gray_bin[0:y_b, 0:width]
            upper = pad_template(upper)
            lower = img_gray_bin[y_b:height, 0:width]
            lower = pad_template(lower)

            contours, _ = cv2.findContours(upper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if get_biggest_contour(contours) > 750:
                subclass_directory_path = directory_path + "\\Upper"
                if not os.path.exists(subclass_directory_path):
                    os.makedirs(subclass_directory_path)
                cv2.imwrite(subclass_directory_path + "//" + digit_ + "_u_" + str(y_b) + "_v"+ str(version) +".jpg", upper)

            contours, _ = cv2.findContours(lower, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if get_biggest_contour(contours) > 750:
                subclass_directory_path = directory_path + "\\Lower"
                if not os.path.exists(subclass_directory_path):
                    os.makedirs(subclass_directory_path)
                cv2.imwrite(subclass_directory_path + "//" + digit_ + "_l_" + str(y_b) + "_v"+ str(version) +".jpg", lower)

        for x_b in x_boundaries:
            left = img_gray_bin[0:height, 0:x_b]
            left = pad_template(left)
            right = img_gray_bin[0:height, x_b:width]
            right = pad_template(right)

            contours, _ = cv2.findContours(left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if get_biggest_contour(contours) > 750:
                subclass_directory_path = directory_path + "\\Left"
                if not os.path.exists(subclass_directory_path):
                    os.makedirs(subclass_directory_path)
                cv2.imwrite(subclass_directory_path + "//" + digit_ + "_i_" + str(x_b) + "_v"+ str(version) +".jpg", left)

            contours, _ = cv2.findContours(right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if get_biggest_contour(contours) > 750:
                subclass_directory_path = directory_path + "\\Right"
                if not os.path.exists(subclass_directory_path):
                    os.makedirs(subclass_directory_path)
                cv2.imwrite(subclass_directory_path + "//" + digit_ + "_d_" + str(x_b) + "_v"+ str(version) +".jpg", right)

        y_boundaries = np.arange(20, int(height * .66), 10)
        x_boundaries = np.arange(20, int(width * .66), 10)

        for y_b in y_boundaries:
            for x_b in x_boundaries:
                top_left = img_gray_bin[0:y_b, 0:x_b]
                top_left = pad_template(top_left)
                top_right = img_gray_bin[0:y_b, x_b:width]
                top_right = pad_template(top_right)
                bottom_left = img_gray_bin[y_b:height, 0:x_b]
                bottom_left = pad_template(bottom_left)
                bottom_right = img_gray_bin[y_b:height, x_b:width]
                bottom_right = pad_template(bottom_right)

                contours, _ = cv2.findContours(top_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if get_biggest_contour(contours) > 750:
                    subclass_directory_path = directory_path + "\\TopLeft"
                    if not os.path.exists(subclass_directory_path):
                        os.makedirs(subclass_directory_path)
                    cv2.imwrite(subclass_directory_path + "//" + digit_ + "_tl_" + str(x_b) + "_" + str(y_b) + "_v"+ str(version) +".jpg",
                                top_left)

                contours, _ = cv2.findContours(top_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if get_biggest_contour(contours) > 750:
                    subclass_directory_path = directory_path + "\\TopRight"
                    if not os.path.exists(subclass_directory_path):
                        os.makedirs(subclass_directory_path)
                    cv2.imwrite(subclass_directory_path + "//" + digit_ + "_tr_" + str(x_b) + "_" + str(y_b) + "_v"+ str(version) +".jpg",
                                top_right)

                contours, _ = cv2.findContours(bottom_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if get_biggest_contour(contours) > 750:
                    subclass_directory_path = directory_path + "\\BottomLeft"
                    if not os.path.exists(subclass_directory_path):
                        os.makedirs(subclass_directory_path)
                    cv2.imwrite(subclass_directory_path + "//" + digit_ + "_bl_" + str(x_b) + "_" + str(y_b) + "_v"+ str(version) +".jpg",
                                bottom_left)

                contours, _ = cv2.findContours(bottom_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if get_biggest_contour(contours) > 750:
                    subclass_directory_path = directory_path + "\\BottomRight"
                    if not os.path.exists(subclass_directory_path):
                        os.makedirs(subclass_directory_path)
                    cv2.imwrite(subclass_directory_path + "//" + digit_ + "_br_" + str(x_b) + "_" + str(y_b) + "_v"+ str(version) +".jpg",
                                bottom_right)

def split_templates():

    directory_path = str(DATA_DIR_RESULTS)
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)

    digit_templates_1 = os.listdir(DATA_DIR_BASE_TEMPLATES)
    digit_templates_2 = os.listdir(DATA_DIR_BASE_TEMPLATES_2)

    split_template(digit_templates_1, directory_path, 1)
    split_template(digit_templates_2, directory_path, 2)


if __name__ == '__main__':
    digit_templates_1 = os.listdir(DATA_DIR_BASE_TEMPLATES)
    digit_templates_2 = os.listdir(DATA_DIR_BASE_TEMPLATES_2)
    directory_path = str(DATA_DIR_RESULTS)
    # if os.path.exists(directory_path):
    #     shutil.rmtree(directory_path)
    split_template(digit_templates_1, directory_path, 1)
    split_template(digit_templates_2, directory_path, 2)

