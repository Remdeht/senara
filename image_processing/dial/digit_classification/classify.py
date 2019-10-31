# Performs k-NN classification based on the features extracted from the digit boundary
# libraries
import cv2
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.externals import joblib

# root directory
from image_processing.constants import ROOT_DIR

# other scripts from the package
from .boundary import get_boundary_li
from . import fourier_descriptors


def classify_digit(img, fd, n, flag_knn, flag_templates, img_name=None, directory=None):
    try:
        return main_operation(img=img, fd=fd, n=n, flag_knn=flag_knn, flag_templates=flag_templates,
                              img_name=img_name, dir_name=directory)
    except Exception as e:
        raise


def main_operation(img, fd, n, flag_knn, flag_templates, img_name=None, dir_name=None):
    """Main operation for the classification of a input image of a digit.

        fd = number of Fourier descriptors (10,20,30,40,50)
        n = number of neighbours for k-NN (5,7,9,11)
        flag_knn = Whether to use a distance weighted classification (1) or a uniformly weighted classification
        flag_templates = Which template set to use: Randomly generated Digit templates (1) or the alternative Digit templates(0)
        """

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

    y_final = np.array(pd.read_csv(Y_FINAL)['training'])
    clf = joblib.load(CLASSIFIER_LOCATION)

    boundary_result = get_boundary_li(img)

    if boundary_result is None:
        return None, None, None

    elif len(boundary_result) is 6:  # In case two digit regions were detected
        cnt_digit_upper, img_binary_upper, number_of_holes_upper = boundary_result[:3]
        cnt_digit_lower, img_binary_lower, number_of_holes_lower = boundary_result[3:]

        contour_features_upper = get_digit_features(cnt_digit_upper, number_of_holes_upper, fd)
        contour_features_lower = get_digit_features(cnt_digit_lower, number_of_holes_lower, fd)

        if flag_knn is 0:
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

        if img_name is not None and dir_name is not None:
            write_results(result, reliability_score, avg_class_distance, img_binary, fd, n, flag, img_name, dir_name)

        return result, avg_class_distance, reliability_score

    elif len(boundary_result) is 3:
        cnt_digit, img_binary, number_of_holes, = boundary_result
        contour_features = get_digit_features(cnt_digit, number_of_holes, fd)

        if flag_knn is 0:
            result, avg_class_distance, reliability_score = predict(contour_features, n, clf, y_final)
        else:
            result, avg_class_distance, reliability_score = predict_weighted(contour_features, n, clf, y_final)

        if img_name is not None and dir_name is not None:
            write_results(result, reliability_score, avg_class_distance, img_binary, fd, n, flag, img_name, dir_name)

        return result, avg_class_distance, reliability_score


def reposition_contour(cnt):
    """Repositions the contour so the centre of mass is the centre coordinate (0,0) of the boundary function"""
    M = cv2.moments(cnt)  # Get moments of the contour
    cX = int(M["m10"] / M["m00"])  # Calculate x,y coordinate of center
    cY = int(M["m01"] / M["m00"])

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
        centered_cnt_digit,
        fd
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


def write_results(result, reliability_score, avg_class_distance, img_binary, fd, k, flag, img_name, dir_name):
    """Writes the classification results on a binary image containing the digit region and saves the image"""

    H, W = img_binary.shape[:2]
    img_binary = cv2.resize(img_binary, (W * 2, H * 2))
    img_binary = cv2.merge([img_binary, img_binary, img_binary])

    cv2.putText(img_binary, 'Result:%d' % result,
                (int(W * .2), int(H * 1.6)),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 255, 0), 2)
    cv2.putText(img_binary, 'Reliability:%.2f' % reliability_score,
                (int(W * .2), int(H * 1.7)),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 255, 0), 2)
    cv2.putText(img_binary, 'Distance:%.2f' % avg_class_distance,
                (int(W * .2), int(H * 1.8)),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 255, 0), 2)

    if not os.path.exists(dir_name + "\\%d\\%d\\" % (fd, k)):
        os.makedirs(dir_name + "\\%d\\%d\\" % (fd, k))

    cv2.imwrite(dir_name + "\\%d\\%d\\" % (fd, k) + img_name, img_binary)


if __name__ == '__main__':   # To execute run this script as the main program
    from pathlib import Path
    import os, re, csv
    import pandas as pd
    from shutil import copyfile

    # Classifies each digit image in the data digits folder and writes the result to a folder

    DATA_DIR = Path(ROOT_DIR + '\\data\\digits\\')
    digits = os.listdir(DATA_DIR)

    fd = [10,20,30,40,50]  # number of fourier descriptors
    n = [5, 7, 9, 11]  # number of neighbors
    flags_templates = [1]  # 0 is structured templates and 1 is random templates
    flags_knn = [0, 1]  # 0 is uniform and 1 is weighted

    for digit in digits:

        loc = str(DATA_DIR) + "\\" + digit
        img = cv2.imread(loc)

        pattern = r"([1-6])_IMG_.*.jpg"
        dig = re.match(string=digit, pattern=pattern)[1]
        pattern = r"[1-6]_(IMG_.*).jpg"
        image_name = re.match(string=digit, pattern=pattern)[1] + "_" + dig + ".jpg"

        for flag in flags_templates:
            for knn in flags_knn:
                dir_name = "digit_classification_"
                if flag is 0:
                    dir_name += "structured_"
                else:
                    dir_name += "random_"
                if knn is 0:
                    dir_name += "uniform"
                else:
                    dir_name += "weighted"

                final_dir_name = ROOT_DIR + '\\results\\digit_bins_extended\\' + dir_name

                if not os.path.exists(final_dir_name):
                    os.makedirs(final_dir_name)

                for i in fd:
                    for j in n:
                        result, avg_class_distance, reliability_score = classify_digit(
                            img=img, fd=i, n=j,
                            flag_templates=flag, flag_knn=knn,
                            img_name=image_name, directory=final_dir_name)

    # Creates a separate csv file containing the classification results for each digit of each run

    file_name = "temp_file.csv"

    if os.path.exists(file_name):
        os.remove(file_name)

    if not os.path.exists(file_name):
        with open(file_name, 'a+') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')
            row = [
                "Image",
                "Fourier Descriptors",
                "Neighbors",
                "TC 1 result",
                "TC 2 result",
                "TC 3 result",
                "TC 4 result",
                "TC 5 result",
                "TC 6 result",
                "TC 1 rel",
                "TC 1 dist",
                "TC 2 rel",
                "TC 2 dist",
                "TC 3 rel",
                "TC 3 dist",
                "TC 4 rel",
                "TC 4 dist",
                "TC 5 rel",
                "TC 5 dist",
                "TC 6 rel",
                "TC 6 dist"
            ]

            writer.writerow(row)
            csv_file.close()

    names = {}
    for digit in digits:

        pattern = r"[1-6]_(IMG_.*).jpg"
        image_name = re.match(string=digit, pattern=pattern)[1]

        if image_name not in names:
            names[image_name] = 0
            with open(file_name, 'a+') as csv_file:
                writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL,
                                    lineterminator='\n')
                row = [
                    image_name]
                writer.writerow(row)
                csv_file.close()

    for digit in digits:
        loc = str(DATA_DIR) + "\\" + digit
        img = cv2.imread(loc)

        fd = [10, 20, 30, 40, 50]  # number of fourier descriptors
        n = [5, 7, 9, 11]  # number of neighbors
        flags_templates = [1]  # 0 is structured templates and 1 is random templates
        flags_knn = [1]  # 0 is uniform and 1 is weighted

        pattern = r"([1-6])_IMG_.*.jpg"
        dig = re.match(string=digit, pattern=pattern)[1]
        pattern = r"[1-6]_(IMG_.*).jpg"
        image_name = re.match(string=digit, pattern=pattern)[1]

        for flag in flags_templates:
            for knn in flags_knn:
                dir_name = "digit_classification_extended"
                if flag is 0:
                    dir_name += "structured_"
                else:
                    dir_name += "random_"
                if knn is 0:
                    dir_name += "uniform"
                else:
                    dir_name += "weighted"

                final_dir_name = ROOT_DIR + '\\results\\digit_classification\\detailed_results_separated\\' + dir_name

                if not os.path.exists(final_dir_name):
                    os.makedirs(final_dir_name)

                for i in fd:
                    for j in n:
                        print(flag, knn, i,j)
                        local_dir = final_dir_name + '\\' + str(i) + '_fourier_descriptors\\' + str(j) + '_nn'
                        if not os.path.exists(local_dir):
                            os.makedirs(local_dir)
                            copyfile(file_name, local_dir + "\\results.csv")

                        result, avg_class_distance, reliability_score = classify_digit(img=img, fd=i, n=j,
                                                                              flag_templates=flag, flag_knn=knn)
                        df = pd.read_csv(local_dir + "\\results.csv")

                        df.loc[df['Image'] == image_name, 'Fourier Descriptors'] = int(i)
                        df.loc[df['Image'] == image_name, 'Neighbors'] = int(j)
                        if result is None:
                            df.loc[df['Image'] == image_name, 'TC %d result' % int(dig)] = 'NaN'
                            df.loc[df['Image'] == image_name, 'TC %d acc' % int(dig)] = 'NaN'
                            df.loc[df['Image'] == image_name, 'TC %d dist' % int(dig)] = 'NaN'
                        else:
                            df.loc[df['Image'] == image_name, 'TC %d result' % int(dig)] = int(result)
                            df.loc[df['Image'] == image_name, 'TC %d acc' % int(dig)] = reliability_score
                            df.loc[df['Image'] == image_name, 'TC %d dist' % int(dig)] = avg_class_distance

                        df.to_csv(local_dir + "\\results.csv", index=False)
