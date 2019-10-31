import numpy as np
import cv2


def main_process(img, center_tally):
    mask_zero_area = create_mask_zero_area(img, center_tally)
    zero_area_binary = create_binary(mask_zero_area, img)
    zero_coords = get_zero_coordinates(zero_area_binary)

    if zero_coords is None:
        return center_tally
    return zero_coords


def create_mask_zero_area(img, center_tally):
    W, H = img.shape[:2]
    mask = np.zeros((W, H), np.uint8)

    size_boundaries = int(W * 0.015)

    mask[
        center_tally[1] - size_boundaries: center_tally[1],
        center_tally[0] - size_boundaries: center_tally[0] + size_boundaries] = 1

    return mask


def create_binary(mask_zero_area, img):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    channels = cv2.split(img_hsv)
    value = channels[2] * mask_zero_area

    mean_masked_value = np.mean(np.ma.masked_where(value == 0, value))
    std_masked_value = np.std(np.ma.masked_where(value == 0, value))

    bin = cv2.inRange(value, 0, mean_masked_value - (1.5 * std_masked_value)) * mask_zero_area

    return bin


def get_zero_coordinates(zero_binary_area):

    contours, hir = cv2.findContours(zero_binary_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Contour Detection

    if len(contours) is 0:
        return None

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    M = cv2.moments(cnt)
    try:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)
    except Exception:
        return None

