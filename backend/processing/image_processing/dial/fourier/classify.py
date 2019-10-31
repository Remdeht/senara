import numpy as np
import cv2
import pandas as pd

from .boundary import get_boundary_li
from . import fourier_descriptors
from sklearn.externals import joblib
from collections import Counter
from ...constants import ROOT_DIR


def classify_digit(img_digit, fd, n, flag_knn, flag_templates):
    try:
        return main_operation(img_digit=img_digit, fd=fd, n=n, flag_knn=flag_knn, flag_templates=flag_templates)
    except Exception as e:
        raise


def main_operation(img_digit, fd, n, flag_knn, flag_templates):
    """Main operation for the classification of a input image of a digit.

    fd = number of Fourier descriptors (10,20,30,40,50)
    n = number of neighbours for k-NN (5,7,9,11)
    flag_knn = Whether to use a distance weighted classification (1) or a uniformly weighted classification
    flag_templates = Which template set to use: Randomly generated Digit templates (1) or the alternative Digit templates(0)
    """

    # Loads the classifiers and result labels
    if flag_templates is 1 and flag_knn is 0:
        CLASSIFIER_LOCATION = ROOT_DIR + "\\data\\classifiers\\digits_classifier_random_uniform_%dn_%d.joblib.pkl" % (
            n, fd)
        Y_FINAL = ROOT_DIR + "\\data\\result_labels\\ytrain_random_uniform_%dn_%d.csv" % (n, fd)

    elif flag_templates is 1 and flag_knn is 1:
        CLASSIFIER_LOCATION = ROOT_DIR + "\\data\\classifiers\\digits_classifier_random_weighted_%dn_%d.joblib.pkl" % (
            n, fd)
        Y_FINAL = ROOT_DIR + "\\data\\result_labels\\ytrain_random_weighted_%dn_%d.csv" % (n, fd)

    elif flag_templates is 0 and flag_knn is 0:
        CLASSIFIER_LOCATION = ROOT_DIR + "\\data\\classifiers\\digits_classifier_uniform_%dn_%d.joblib.pkl" % (
            n, fd)
        Y_FINAL = ROOT_DIR + "\\data\\result_labels\\ytrain_%dn_%d.csv" % (n, fd)

    else:
        CLASSIFIER_LOCATION = ROOT_DIR + "\\data\\classifiers\\digits_classifier_weighted_%dn_%d.joblib.pkl" % (
            n, fd)
        Y_FINAL = ROOT_DIR + "\\data\\result_labels\\ytrain_%dn_%d.csv" % (n, fd)

    y_final = np.array(pd.read_csv(Y_FINAL)['training'])  # Result labels
    clf = joblib.load(CLASSIFIER_LOCATION)  # k-NN Classifier

    boundary_result = get_boundary_li(img_digit=img_digit)  # Gets the boundary of the digit region

    if boundary_result is None:
        return None, None, None

    elif len(boundary_result) is 6:  # Two digit boundaries are extracted
        cnt_digit_upper, img_binary_upper, number_of_holes_upper = boundary_result[:3]
        cnt_digit_lower, img_binary_lower, number_of_holes_lower = boundary_result[3:]

        contour_features_upper = get_digit_features(cnt_digit_upper, number_of_holes_upper, fd)
        contour_features_lower = get_digit_features(cnt_digit_lower, number_of_holes_lower, fd)

        if flag_knn is 0:  # Based on the weighting of the k_NN a different method to calculate the reliability score is used
            result_upper, avg_class_distance_upper, reliability_score_upper = predict(contour_features_upper, n, clf, y_final)
            result_lower, avg_class_distance_lower, reliability_score_lower = predict(contour_features_lower, n, clf, y_final)

            result, avg_class_distance, reliability_score, img_binary = determine_best_result(
                result_upper, avg_class_distance_upper, reliability_score_upper, img_binary_upper,
                result_lower, avg_class_distance_lower, reliability_score_lower, img_binary_lower,
            )

        else:
            result_upper, avg_class_distance_upper, reliability_score_upper = predict_weighted(contour_features_upper, n, clf,
                                                                                      y_final)
            result_lower, avg_class_distance_lower, reliability_score_lower = predict_weighted(contour_features_lower, n, clf,
                                                                                      y_final)

            result, avg_class_distance, reliability_score, img_binary = determine_best_result(
                result_upper, avg_class_distance_upper, reliability_score_upper, img_binary_upper,
                result_lower, avg_class_distance_lower, reliability_score_lower, img_binary_lower
            )
        return result, avg_class_distance, reliability_score

    elif len(boundary_result) is 3:  # only one digit boundary was extracted
        cnt_digit, img_binary, number_of_holes, = boundary_result
        contour_features = get_digit_features(cnt_digit, number_of_holes, fd)

        if flag_knn is 0:
            result, avg_class_distance, reliability_score = predict(contour_features, n, clf, y_final)
        else:
            result, avg_class_distance, reliability_score = predict_weighted(contour_features, n, clf, y_final)
        return result, avg_class_distance, reliability_score


def reposition_contour(cnt):
    """Repositions the contour so the centre of mass is the centre coordinate (0,0) of the boundary function"""
    m = cv2.moments(cnt)  # Get moments of the contour
    cX = int(m["m10"] / m["m00"])  # Calculate x,y coordinate of center
    cY = int(m["m01"] / m["m00"])

    centered_cnt = []

    for point in cnt:  # Adjust each point based on the center of mass
        x = (point[0][0] - cX)
        y = (point[0][1] - cY)
        coord = [x, y]
        centered_cnt.append(coord)

    centered_cnt = np.asarray(centered_cnt)

    return centered_cnt


def predict(contour_features, k_neighbors, clf, y_final):
    """Predicts the digit class based on the features extracted from the digit in the input image using a
        uniformly weighted k-NN classification"""
    result = clf.predict(contour_features)  # Digit Class predicted by the classifier
    neighbors, neighbors_ind = clf.kneighbors(contour_features, k_neighbors)  # Information about the neighbors

    closest_classes = []
    avg_class_distance = np.mean(neighbors[0])

    for index, digit in enumerate(neighbors_ind[0]):  # Extracts the digit class of the nearest neighbours
        closest_classes.append(y_final[digit])

    c = Counter(closest_classes)
    a = c.most_common(1)[0][1]  # Gets the largest neighbor class

    reliability_score = a / k_neighbors

    return int(result), avg_class_distance, reliability_score


def predict_weighted(contour_features, k_neighbors, clf, y_final):
    """Predicts the digit class based on the features extracted from the digit in the input image using a
    distance weighted k-NN classification"""

    result = clf.predict(contour_features)  # Digit Class predicted by the classifier
    neighbors, neighbors_ind = clf.kneighbors(contour_features, k_neighbors)  # Information about the neighbors

    closest_classes = []
    closest_classes_weights = []
    avg_class_distance = np.mean(neighbors[0])

    for index, digit in enumerate(neighbors_ind[0]):  # Extracts the digit class of the nearest neighbours
        closest_classes.append(y_final[digit])
        closest_classes_weights.append(y_final[digit])

    for index, dist in enumerate(neighbors[0]):
        weight = 1.0 / dist  # Calculates the weight of each neighbor in classification based on their distance
        closest_classes_weights[index] = (closest_classes_weights[index], weight, dist)

    c = Counter(closest_classes)
    a = c.most_common(1)[0][0]  # Gets the largest digit class within the group of k - nearest neighbors
    total_weights = sum(w for c, w, d in closest_classes_weights)
    weights_biggest_class = 0
    for cls in closest_classes_weights:
        if int(cls[0]) is int(a):
            weights_biggest_class += cls[1]

    reliability_score = weights_biggest_class / total_weights

    return int(result), avg_class_distance, reliability_score


def get_digit_features(cnt, number_of_holes, fd):
    """Calculates the features of the digit extracted from the input image i.e. the fourier descriptors
    and number of holes"""

    centered_cnt_digit = reposition_contour(cnt)

    fourier_x, fourier_y = fourier_descriptors.calc_fourier_descriptors(
        boundary=centered_cnt_digit,
        number_of_descriptors=fd
    )

    return [fourier_x + fourier_y + [number_of_holes/2]]


def determine_best_result(result_upper, avg_class_distance_upper, reliability_score_upper, binary_upper,
                          result_lower, avg_class_distance_lower, reliability_score_lower, binary_lower):
    """Determines the best result in case two digit regions are extracted from the input image"""
    if result_upper is 9 and result_lower is 0:
        return result_upper, avg_class_distance_upper, reliability_score_upper, binary_upper

    elif result_upper is not 9 and result_lower is (result_upper + 1):
        return result_upper, avg_class_distance_upper, reliability_score_upper, binary_upper

    elif avg_class_distance_upper <= avg_class_distance_lower and reliability_score_upper >= reliability_score_lower:
        return result_upper, avg_class_distance_upper, reliability_score_upper, binary_upper

    elif avg_class_distance_upper <= avg_class_distance_lower:
        return result_upper, avg_class_distance_upper, reliability_score_upper, binary_upper

    elif reliability_score_upper >= reliability_score_lower:
        return result_upper, avg_class_distance_upper, reliability_score_upper, binary_upper
    else:
        return result_lower, avg_class_distance_lower, reliability_score_lower, binary_lower

