# Locates the red pointer for correctional rotation
import os
import cv2
import numpy as np
from .helpers import general_functions
from .helpers import adaptive_gamma_correction
from image_processing.constants import ROOT_DIR


def get_red_pointer(img, mask_cap):
    """Creates a binary mask of pixels that can be classified as the red pointer"""

    image_corrected = adaptive_gamma_correction.agc(img)
    image_blurred = cv2.GaussianBlur(image_corrected, (9, 9), 1)  # Smoothing for noise removal

    image_hsv = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2HSV)  # Convert to the HSV color space
    channels = cv2.split(image_hsv)
    saturation = adaptive_gamma_correction.agc_1c(channels[1])  # Saturation is a good indicator for the red pointer
    value = adaptive_gamma_correction.agc_1c(channels[2])  # Lightness comes in handy to remove possible noise

    # Create masks of both channels for the contour detection later
    mask_saturation = cv2.inRange(saturation, np.mean(saturation) + (3 * np.std(saturation)), np.max(saturation))
    mask_value = cv2.inRange(value, np.mean(value) - (.5 * np.std(value)), np.max(value))

    if np.count_nonzero(mask_saturation) < 100:  # Safety clause in case saturation is low because of shade
        mask_saturation = cv2.inRange(saturation, np.mean(saturation) + (2.5 * np.std(saturation)), np.max(saturation))

    mask_cap = (255 - mask_cap)
    total_mask = mask_cap * mask_saturation * mask_value

    return total_mask


def get_center_pointer(img, mask_cap, name=None):
    """Main Function taking care of the extraction of the red pointer"""

    mask_pointer = get_red_pointer(img, mask_cap)
    contours, hir = cv2.findContours(mask_pointer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) is 0:
        raise ValueError('The Red Pointer is not visible!')

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    W, H = img.shape[:2]

    if len(cnt) < W * .2:
        raise ValueError('The Red Pointer is not visible!')

    # Fit Rectangle around contour

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Split Contour

    tl, tr, br, bl = general_functions.order_rectangle_coordinates(list(box))

    northern_points = [tl, tr]
    northern_points.sort(key=lambda x: x[1])
    north_point_rs = northern_points[0]

    southern_points = [br, bl]
    southern_points.sort(key=lambda x: x[1])
    south_point_rs = southern_points[1]

    horizontal_points = [northern_points[1], southern_points[0]]
    horizontal_points.sort(key=lambda x: x[0])
    west_point_rs = horizontal_points[0]
    east_point_rs = horizontal_points[1]

    side_1 = general_functions.distance(north_point_rs, east_point_rs)
    side_2 = general_functions.distance(north_point_rs, west_point_rs)

    if side_1 > side_2:

        point1_x = int(east_point_rs[0] - (((east_point_rs[0] - north_point_rs[0]) / 3) * 2))
        point1_y = int(north_point_rs[1] - ((north_point_rs[1] - east_point_rs[1]) / 3))

        point2_x = int(west_point_rs[0] - ((west_point_rs[0] - south_point_rs[0]) / 3))
        point2_y = int(west_point_rs[1] - ((west_point_rs[1] - south_point_rs[1]) / 3))

        point3_x = int(east_point_rs[0] - ((east_point_rs[0] - north_point_rs[0]) / 3))
        point3_y = int(north_point_rs[1] - (((north_point_rs[1] - east_point_rs[1]) / 3) * 2))

        point4_x = int(west_point_rs[0] - (((west_point_rs[0] - south_point_rs[0]) / 3) * 2))
        point4_y = int(west_point_rs[1] - (((west_point_rs[1] - south_point_rs[1]) / 3) * 2))

        box1 = np.array(general_functions.order_rectangle_coordinates(
            [[point2_x, point2_y], list(west_point_rs), list(north_point_rs), [point1_x, point1_y]]
        ))
        box2 = np.array(general_functions.order_rectangle_coordinates(
            [list(south_point_rs), list(east_point_rs), [point3_x, point3_y], [point4_x, point4_y]]
        ))

    else:
        point1_x = int(west_point_rs[0] + ((abs(north_point_rs[0] - west_point_rs[0]) / 3) * 2))
        point1_y = int(north_point_rs[1] + (abs(north_point_rs[1] - west_point_rs[1]) / 3))

        point2_x = int(east_point_rs[0] - (abs(east_point_rs[0] - south_point_rs[0]) / 3))
        point2_y = int(east_point_rs[1] + (abs(east_point_rs[1] - south_point_rs[1]) / 3))

        point3_x = int(west_point_rs[0] + ((abs(north_point_rs[0] - west_point_rs[0]) / 3)))
        point3_y = int(north_point_rs[1] + ((abs(north_point_rs[1] - west_point_rs[1]) / 3) * 2))

        point4_x = int(east_point_rs[0] - ((abs(east_point_rs[0] - south_point_rs[0]) / 3) * 2))
        point4_y = int(east_point_rs[1] + ((abs(east_point_rs[1] - south_point_rs[1]) / 3) * 2))

        box1 = np.array(general_functions.order_rectangle_coordinates(
            [list(north_point_rs), [point1_x, point1_y], list(east_point_rs), [point2_x, point2_y]]
        ))
        box2 = np.array(general_functions.order_rectangle_coordinates(
            [[point4_x, point4_y], list(west_point_rs), [point3_x, point3_y], list(south_point_rs)]
        ))

    mask_box_1 = create_mask_box(box1, mask_pointer)  # Now we need to get the center of mass of the biggest contour
    mask_box_2 = create_mask_box(box2, mask_pointer)

    part_pointer_box1 = mask_box_1 * mask_pointer
    part_pointer_box2 = mask_box_2 * mask_pointer

    contours_box1, hir = cv2.findContours(part_pointer_box1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in contours_box1]
    max_index = np.argmax(areas)
    cnt_box1 = contours_box1[max_index]

    contours_box2, hir = cv2.findContours(part_pointer_box2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in contours_box2]
    max_index = np.argmax(areas)
    cnt_box2 = contours_box2[max_index]

    if cv2.contourArea(cnt_box1) > cv2.contourArea(cnt_box2):
        M = cv2.moments(cnt_box1)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    else:
        M = cv2.moments(cnt_box2)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    if name is not None:
        if not os.path.exists(ROOT_DIR + '\\results\\image_processing\\red_pointer'):
            os.makedirs(ROOT_DIR + '\\results\\image_processing\\red_pointer\\binary\\')
            os.makedirs(ROOT_DIR + '\\results\\image_processing\\red_pointer\\pointer_region\\')

        cv2.imwrite(ROOT_DIR + '\\results\\image_processing\\red_pointer\\binary\\' + name, mask_pointer)
        cv2.imwrite(ROOT_DIR + '\\results\\image_processing\\red_pointer\\pointer_region\\' + name,
                    part_pointer_box1 + part_pointer_box2)

    return (cx, cy)


def create_mask_box(box, img):
    """Creates a mask of the box that it receives as input"""

    height, width = img.shape[:2]

    mask = np.zeros((height, width), np.uint8)
    cv2.drawContours(mask, [box], -1, 1, -1)

    return mask
