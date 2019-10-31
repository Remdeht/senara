# Gets the coordinates of the zero point of the dial
import numpy as np
import cv2


def main_process(img, center_tally):
    """Main function for the detection of the zero point of the dial"""
    mask_zero_area = create_mask_zero_area(img, center_tally)
    zero_area_binary = create_binary(mask_zero_area, img)
    zero_coords = get_zero_coordinates(zero_area_binary)

    if zero_coords is None:
        return center_tally
    else:
        return zero_coords


def create_mask_zero_area(img, center_tally):
    """Creates a mask to extract the area around the zero point"""
    width, height = img.shape[:2]
    mask = np.zeros((width, height), np.uint8)

    size_boundaries = int(width * 0.015)

    mask[
        center_tally[1] - size_boundaries: center_tally[1],
        center_tally[0] - size_boundaries: center_tally[0] + size_boundaries] = 1

    return mask


def create_binary(mask_zero_area, img):
    """Creates a binary image for region detection"""

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    channels = cv2.split(img_hsv)
    value = channels[2] * mask_zero_area

    mean_masked_value = np.mean(np.ma.masked_where(value == 0, value))
    std_masked_value = np.std(np.ma.masked_where(value == 0, value))

    binary = cv2.inRange(value, 0, mean_masked_value - (1.5 * std_masked_value)) * mask_zero_area

    return binary


def get_zero_coordinates(zero_binary_area):
    """Selects the coordinates for the zero point based on the centre of mass of the largest region within the
    extracted area"""

    contours, hir = cv2.findContours(zero_binary_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Contour Detection

    if len(contours) is 0:
        return None

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]  # Largest region contour is selected as the zero point

    m = cv2.moments(cnt)  # Gets the centre of mass of the selected region as the coordinates for the zero point

    try:
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])
        return cx, cy
    except Exception:
        return None

