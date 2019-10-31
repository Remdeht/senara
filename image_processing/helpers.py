# Helper functions for the main program

import math
import imutils
import os
import cv2
from image_processing.constants import ROOT_DIR


def rotate_dial(img_dial, x_cap, y_cap, x_pointer, y_pointer):
    """Rotates the image of the dial based on the position of the red pointer"""
    try:
        height, width = img_dial.shape[:2]

        side_a = math.sqrt((abs(y_cap - y_pointer) ** 2) + (abs(x_cap - x_pointer) ** 2))
        side_b = abs(y_cap - height)
        side_c = math.sqrt((abs(x_cap - x_pointer) ** 2) + (abs(y_pointer - height) ** 2))
        cosine_alfa = ((side_c ** 2) - (side_a ** 2) - (side_b ** 2)) / (-2 * side_a * side_b)
        angle_alfa = math.degrees(math.acos(cosine_alfa))

        if x_pointer >= x_cap:
            rotation = 43 - angle_alfa
            img_dial = imutils.rotate(img_dial, rotation)

        else:
            rotation = 43 + angle_alfa
            img_dial = imutils.rotate(img_dial, rotation)

    except Exception:
        raise
    else:
        return img_dial


def crop_image(image, mask_dial, x_cap, y_cap, r_dial, mask_cap, name=None):
    """Crops the original image and mask to the dial and adjusts the center coordinates of the dial"""

    img_dial = image * mask_dial
    img_dial = crop_to_circle(x_cap, y_cap, r_dial, img_dial)
    mask_cap = crop_to_circle(x_cap, y_cap, r_dial, mask_cap)
    height, width = img_dial.shape[:2]
    x_cap = int(width / 2)
    y_cap = int(height / 2)

    if name is not None:  # writes the result of the dial cropping
        if not os.path.exists(ROOT_DIR + '\\results\\image_processing\\dial\\cropped_dial'):
            os.makedirs(ROOT_DIR + '\\results\\image_processing\\dial\\cropped_dial')

        cv2.imwrite(ROOT_DIR + '\\results\\image_processing\\dial\\cropped_dial\\' + name, img_dial)

    return img_dial, mask_cap, x_cap, y_cap


def crop_to_circle(x, y, r, mask):
    """Crops an image to a circle based on the center and radius of the circle"""

    height, width = mask.shape[:2]

    a = y - r
    if a < 0:
        a = 0
    b = y + r
    if b > height:
        b = height
    c = x - r
    if c < 0:
        c = 0
    d = x + r
    if d > width:
        d = width

    return mask[a:b, c:d]


def write_results(pointer_value, tally_values, fd, n, img_dial, name, flag):
    """Writes the results of the image processing to a an image and a csv file"""

    if flag is 1:
        dir_name = 'results\\total_results_unsupervised_FD%d_KNN%d' % (fd, n)
    elif flag is 0:
        dir_name = 'results\\total_results_templates_FD%d_KNN%d' % (fd, n)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # The following lines write the results to the image being analysed.
    h, w = img_dial.shape[:2]
    img_dial[int(h * .3):int(h * .7), int(w * .6):w] = 255

    if pointer_value is not None:
        cv2.putText(img_dial, 'pointer value = %d' % pointer_value, (int(w * .61), int(h * .35)), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 0, 0), 1)
    elif pointer_value is None:
        cv2.putText(img_dial, 'pointer value = None', (int(w * .61), int(h * .35)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 0, 0), 1)
    for ind, val in enumerate(tally_values):
        if val[0] is not None:
            cv2.putText(img_dial, 'T%d:%d, A:%.2f, D:%.2f' % ((ind+1), val[0], val[2], val[1]), (int(w * .61), int(h * (.4 + (ind * .05)))),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 1)
        else:
            cv2.putText(img_dial, 'T%d = None' % (ind+1),
                        (int(w * .61), int(h * (.4 + (ind * .05)))),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 1)

    cv2.imwrite(dir_name + '\\' + name, img_dial)
