# Creates the datasets used to train the k-NN classifiers.
import os
import re
import csv
import cv2
import numpy as np
from image_processing.constants import ROOT_DIR
from helpers import fourier_descriptors


DATA_DIR_STRUCTURED = ROOT_DIR + "\\data\\structured_templates"  # Directory containing the structured digit templates
classes = os.listdir(DATA_DIR_STRUCTURED)

DATA_DIR_RANDOM = ROOT_DIR + "\\data\\random_templates"  # Directory containing the random digit templates
unknown_digits = os.listdir(DATA_DIR_RANDOM)

# Directories to store the datasets
DATA_DIR_KNN_STRUCTURED = ROOT_DIR + "\\data\\knn_structured_templates"
DATA_DIR_KNN_RANDOM = ROOT_DIR + "\\data\\knn_random_templates"


def reposition_contour(cnt):
    """Repositions the contour coordinates in order for it's center of mass to be it's origin"""

    M = cv2.moments(cnt)  # Get moments of the contour
    cX = int(M["m10"] / M["m00"])  # calculate x,y coordinate of center of mass
    cY = int(M["m01"] / M["m00"])

    centered_cnt = []

    for point in cnt:  # adjust each point based on the center of mass
        x = (point[0][0] - cX)
        y = (point[0][1] - cY)
        coord = [x, y]
        centered_cnt.append([coord])

    centered_cnt = np.asarray(centered_cnt)

    return centered_cnt


def find_contour_digit(img_binary):
    """Get the biggest contour from the binary image"""
    try:
        contours, hir = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
    except Exception:
        print("No contours detected!")
        return None, None
    else:
        cnt = [contours[max_index]]

        if len(contours[max_index]) < 51:
            return None, None

        # Create mask to draw the contour on
        (W, H) = img_binary.shape[:2]
        cnt_canvas = np.zeros((W, H), np.uint8)
        cv2.drawContours(cnt_canvas, cnt, -1, 1, -1)

        return contours[max_index], cnt_canvas


def create_binary(img_digit):
    """Creates a binary image for region extraction"""

    img_digit_hsv = cv2.cvtColor(img_digit, cv2.COLOR_BGR2HSV)
    channels = cv2.split(img_digit_hsv)
    img_digit_value = channels[2]
    img_digit_value = cv2.GaussianBlur(img_digit_value, (7, 7), 0)
    img_digit_binary = cv2.inRange(img_digit_value, 0, np.mean(img_digit_value))

    H, W = img_digit_binary.shape[:2]

    padding = np.zeros((H + 10, W + 10), np.uint8)
    padding[5:5 + H, 5:5 + W] = img_digit_binary
    img_digit_binary = padding

    return img_digit_binary


def create_knn_dataset_random_templates(number_of_descriptors, size=100):
    """Creates a csv file of the randomly generated digit templates containing their fourier descriptors
    and number of holes to be used to train the k-NN classifier"""
    if not os.path.exists(DATA_DIR_KNN_RANDOM):
        os.makedirs(DATA_DIR_KNN_RANDOM)

    file_name = "\\random_FD_%d.csv" % number_of_descriptors  # Csv file containing the training data

    if os.path.exists(DATA_DIR_KNN_RANDOM + file_name):
        os.remove(DATA_DIR_KNN_RANDOM + file_name)

    with open(DATA_DIR_KNN_RANDOM + file_name, 'a+') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL,
                            lineterminator='\n')
        column_names_x = []
        column_names_y = []

        for i in range(number_of_descriptors):
            column_names_x.append("x_%d" % (i + 1))
            column_names_y.append("y_%d" % (i + 1))

        row = column_names_x + column_names_y + ["inner contours"] + ["class"]

        writer.writerow(row)
        csv_file.close()

    for unknown_digit in unknown_digits:

        pattern = r"([0-9]).*.jpg"
        correct_result = int(re.findall(pattern, unknown_digit)[0])

        img_size_X = size
        img_size_Y = int(img_size_X * 1.8)

        img_unknown_digit = cv2.imread(DATA_DIR_RANDOM + "\\" + unknown_digit)
        img_unknown_digit = cv2.resize(img_unknown_digit, (img_size_X, img_size_Y))
        img_unknown_digit_bin = create_binary(img_unknown_digit)

        cnt, canvas = find_contour_digit(img_unknown_digit_bin)

        if cnt is None:
            continue

        mask = np.zeros(img_unknown_digit_bin.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 1, -1)
        img_digit_inv = ((255 - img_unknown_digit_bin) * mask)

        contours, _ = cv2.findContours(img_digit_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(c) for c in contours]
        inner_contours = 0
        for area in areas:
            if area > 25:
                inner_contours += 1

        if inner_contours > 2:
            continue

        centered_cnt_unknown_digit = reposition_contour(cnt)  # reposition contour based on center of gravity

        fourier_x_invariant, fourier_y_invariant = fourier_descriptors.calc_fourier_descriptors(
            centered_cnt_unknown_digit,
            number_of_descriptors)

        if fourier_x_invariant is not None and fourier_y_invariant is not None:
            with open(DATA_DIR_KNN_RANDOM + file_name, 'a+') as csv_file:
                writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL,
                                    lineterminator='\n')

                row = fourier_x_invariant + fourier_y_invariant + [inner_contours/2] + [correct_result]

                writer.writerow(row)
                csv_file.close()


def create_knn_dataset_structured_templates(number_of_descriptors):
    """Creates a csv file of the generated digit templates using a template set containing their fourier descriptors
        and number of holes to be used to train the k-NN classifier"""
    if not os.path.exists(DATA_DIR_KNN_STRUCTURED):
        os.makedirs(DATA_DIR_KNN_STRUCTURED)

    if os.path.exists(DATA_DIR_KNN_STRUCTURED + "\\fourier_classifiers_meta.csv"):
        os.remove(DATA_DIR_KNN_STRUCTURED + "\\fourier_classifiers_meta.csv")

    dir_name = "\\structured_FD_%d.csv" % number_of_descriptors

    if os.path.exists(DATA_DIR_KNN_STRUCTURED + dir_name):
        os.remove(DATA_DIR_KNN_STRUCTURED + dir_name)

    with open(DATA_DIR_KNN_STRUCTURED + "\\fourier_classifiers_meta.csv", 'a+') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL,
                            lineterminator='\n')

        row = ["Class ID", "Digit class", "Digit Subclass", "Template"]

        writer.writerow(row)
        csv_file.close()

    with open(DATA_DIR_KNN_STRUCTURED + dir_name, 'a+') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL,
                            lineterminator='\n')

        column_names_x = []
        column_names_y = []

        for i in range(number_of_descriptors):
            column_names_x.append("x_%d" % (i + 1))
            column_names_y.append("y_%d" % (i + 1))

        row = column_names_x + column_names_y + ["inner contours"] + ["class"] + ["digit class"]

        writer.writerow(row)
        csv_file.close()

    class_number = 0

    for cl in classes:
        sub_classes = os.listdir(DATA_DIR_STRUCTURED + "\\" + cl)

        for sub_class in sub_classes:
            templates = os.listdir(DATA_DIR_STRUCTURED + "\\" + cl + "\\" + sub_class)

            for template in templates:
                pattern = r"(.*).*.jpg"
                template_name = re.findall(pattern, template)[0]

                with open(DATA_DIR_KNN_STRUCTURED + "\\fourier_classifiers_meta.csv", 'a+') as csv_file:
                    writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL,
                                        lineterminator='\n')

                    row = [class_number, cl, sub_class, template_name]
                    writer.writerow(row)
                    csv_file.close()

                img_digit = cv2.imread(DATA_DIR_STRUCTURED + "\\" + cl + "\\" + sub_class + "\\" + template, 0)
                img_digit_bin = cv2.inRange(img_digit, np.mean(img_digit), 255)

                contours, _ = cv2.findContours(img_digit_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                areas = [cv2.contourArea(c) for c in contours]
                max_index = np.argmax(areas)
                cnt = contours[max_index]

                mask = np.zeros(img_digit.shape, np.uint8)
                cv2.drawContours(mask, [cnt], -1, 1, -1)
                img_digit_inv = ((255 - img_digit_bin) * mask)

                contours, _ = cv2.findContours(img_digit_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                areas = [cv2.contourArea(c) for c in contours]
                inner_contours = 0
                for area in areas:
                    if area > 25:
                        inner_contours += 1

                boundary = reposition_contour(cnt)
                fourier_x_inv, fourier_y_inv = fourier_descriptors.calc_fourier_descriptors(boundary.tolist(),
                                                                                            number_of_descriptors)

                if len(fourier_x_inv) > 1 and len(fourier_y_inv) > 1:
                    with open(DATA_DIR_KNN_STRUCTURED + dir_name, 'a+') as csv_file:
                        writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL,
                                            lineterminator='\n')

                        row = fourier_x_inv + fourier_y_inv + [inner_contours/2] + [class_number] + [cl]
                        writer.writerow(row)
                        csv_file.close()

                class_number += 1

    return False


if __name__ == '__main__':
    # Run this script to create the datasets
    create_knn_dataset_structured_templates(number_of_descriptors=10)
    create_knn_dataset_structured_templates(number_of_descriptors=20)
    create_knn_dataset_structured_templates(number_of_descriptors=30)
    create_knn_dataset_structured_templates(number_of_descriptors=40)
    create_knn_dataset_structured_templates(number_of_descriptors=50)
    create_knn_dataset_random_templates(number_of_descriptors=10)
    create_knn_dataset_random_templates(number_of_descriptors=20)
    create_knn_dataset_random_templates(number_of_descriptors=30)
    create_knn_dataset_random_templates(number_of_descriptors=40)
    create_knn_dataset_random_templates(number_of_descriptors=50)


