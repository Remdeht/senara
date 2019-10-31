import numpy as np
import cv2
from .helpers import general_functions
from .helpers import adaptive_gamma_correction
import math

INNER_THRESHOLD = .40
OUTER_THRESHOLD = .55
BREAKPOINT = float("%.2f" % (INNER_THRESHOLD + ((OUTER_THRESHOLD - INNER_THRESHOLD) / 2)))


def read_pointer(img_dial, center_tally, ):
    mask_pointer = create_mask_pointer(img_dial)
    mask_value, img_pointer = create_binary(img_dial, mask_pointer)

    contours, hir = cv2.findContours(mask_value, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    H, W = img_pointer.shape[:2]

    possible_pointer_contours = check_possible_pointers(contours, H, W)

    if possible_pointer_contours is None:
        pointer = alternative_process(img_dial, mask_pointer, )

    else:
        maximum_pointer_size = (np.count_nonzero(mask_pointer)) * .04
        pointer = determine_final_pointer(possible_pointer_contours, img_pointer, maximum_pointer_size, img_dial)

        if pointer is None:
            pointer = alternative_process(img_dial, mask_pointer, )

    if pointer is None:
        return None

    (W, H) = img_dial.shape[:2]
    (X, Y) = int(W / 2), int(H / 2)

    M = cv2.moments(pointer)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    value = determine_pointer_value((cx, cy), (X, Y), center_tally)

    return value


def alternative_process(img_dial, mask_pointer, ):
    binary = create_binary_yuv(img_dial, mask_pointer)

    contours, hir = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) is 0:
        return None

    H, W = binary.shape[:2]
    possible_pointer_contours = check_possible_pointers_yuv(contours, H, W)

    if possible_pointer_contours is None:
        return None

    maximum_pointer_size = (np.count_nonzero(mask_pointer)) * .04
    pointer = determine_final_pointer(possible_pointer_contours, binary, maximum_pointer_size, img_dial)

    if pointer is None:
        return None

    return pointer


def determine_pointer_value(pointer, center, zero):

    sideA = general_functions.distance(pointer, center)
    sideB = general_functions.distance(center, zero)
    sideC = general_functions.distance(zero, pointer)
    cosine_alfa = ((sideC ** 2) - (sideA ** 2) - (sideB ** 2)) / (-2 * sideA * sideB)
    angle_alfa = math.degrees(math.acos(cosine_alfa))
    value_per_degree = 130/180

    if pointer[0] < center[0]:
        value = angle_alfa * value_per_degree
    else:
        value = 130 + ((180 - angle_alfa) * value_per_degree)

    if value > 200:
        value = 0

    return int(np.round(value))


def create_binary_yuv(img, mask):
    img = adaptive_gamma_correction.agc_dials(img)
    img_dial_blurred = cv2.blur(img, (3, 3))  # Smooth the image

    img_yuv = cv2.cvtColor(img_dial_blurred, cv2.COLOR_BGR2YUV)
    channels = cv2.split(img_yuv)
    u = np.ma.masked_where(channels[1] == 0, channels[1])
    u = cv2.equalizeHist(u)
    u *= cv2.split(mask)[0]
    binary = cv2.inRange(u, np.mean(u) + (6 * np.std(u)), 255) * cv2.split(mask)[0]

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=1)

    return binary


def check_possible_pointers(contours, H, W):
    pointers = []
    areas = [cv2.contourArea(c) for c in contours]

    max_pixels = int(np.pi * W * OUTER_THRESHOLD) * 0.03 * (W * 0.01)

    if len(contours) > 3:
        ind = np.argpartition(areas, -3)[-3:]

        largest_contours = []
        for i in ind:
            largest_contours.append(contours[i])  # Get the 3 biggest contours
        contours = largest_contours

    for cnt in contours:
        pixels_edge = check_pixels_on_edge(cnt, H, W)

        if 250 < pixels_edge < max_pixels:
            pointers.append(cnt)
        else:
            pass

    if len(pointers) > 0:
        return pointers
    else:
        return None


def check_possible_pointers_yuv(contours, H, W):
    pointers = []
    areas = [cv2.contourArea(c) for c in contours]

    if len(contours) > 5:
        ind = np.argpartition(areas, -5)[-5:]
        contours = [contours[i] for i in ind]  # Get the 5 biggest contours

    for cnt in contours:
        pixels_edge = check_pixels_on_edge_yuv(cnt, H, W)

        if pixels_edge > 250:
            pointers.append(cnt)
        else:
            pass

    if len(pointers) > 0:
        return pointers
    else:
        return None


def check_pixels_on_edge(cnt, H, W):
    cnt_canvas = np.zeros((W, H), np.uint8)
    cv2.drawContours(cnt_canvas, [cnt], -1, 255, -1)

    mask_edge = create_mask_edge(W, H) * cnt_canvas

    pixels_edge = np.count_nonzero((mask_edge == [255]))

    return pixels_edge


def check_pixels_on_edge_yuv(cnt, H, W):
    cnt_canvas = np.zeros((W, H), np.uint8)
    cv2.drawContours(cnt_canvas, [cnt], -1, 255, -1)
    mask_edge = create_mask_edge_yuv(W, H) * cnt_canvas

    pixels_edge = np.count_nonzero((mask_edge == [255]))

    return pixels_edge


def create_binary(img, mask):
    img = adaptive_gamma_correction.agc_dials(img)

    img_dial_blurred = cv2.blur(img, (3, 3))  # Smooth the image
    img_pointer = mask * img_dial_blurred

    img_pointer_hls = cv2.cvtColor(img_pointer, cv2.COLOR_BGR2HLS)  # Convert to the HLS color space
    channels = cv2.split(img_pointer_hls)
    img_pointer_lightness_raw = channels[1]
    lightness_factor = img_pointer_lightness_raw * .6

    channels_bgr = cv2.split(img_pointer)
    r = np.ma.masked_where(channels_bgr[2] == 0, channels_bgr[2]) - lightness_factor
    r *= 255.0 / r.max()
    th_3 = np.mean(r) - (1.5 * np.std(r))
    r = np.asarray(r, dtype=np.uint8)
    binary = cv2.inRange(r, 0, th_3)
    binary *= cv2.split(mask)[0]

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=1)

    return binary, img_pointer


def determine_final_pointer(contours, img, estimated_max_pointersize, img_org):
    possible_pointers = []

    height, width = img.shape[:2]
    mask_result = np.zeros((height, width), np.uint8)

    for c in contours:
        if (estimated_max_pointersize * .1) < cv2.contourArea(c) < estimated_max_pointersize:
            mask = np.zeros((height, width), np.uint8)
            mask_inner = create_mask_inner(img)
            mask_outer = create_mask_outer(img)

            cv2.drawContours(mask, [c], -1, 255, -1)

            mask_inner = mask_inner * mask
            mask_outer = mask_outer * mask

            pixels_inner = np.count_nonzero((mask_inner == [255]))
            pixels_outer = np.count_nonzero((mask_outer == [255]))

            if pixels_outer is not 0 and pixels_inner / pixels_outer >= 1.4:
                possible_pointers.append(c)
                cv2.drawContours(mask_result, [c], -1, 255, -1)


    if len(possible_pointers) == 1:
        return possible_pointers[0]
    elif len(possible_pointers) > 1:
        pointer = determine_best_candidate(possible_pointers, img_org)
        return pointer
    else:
        return None


def determine_best_candidate(pointers, img):
    dic = {}

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    channels = cv2.split(img_yuv)
    u = cv2.equalizeHist(channels[1])

    for index, pointer in enumerate(pointers):
        H, W = img.shape[:2]
        mask = np.zeros((H, W), np.uint8)

        cv2.drawContours(mask, [pointer], -1, 1, -1)
        u_copy = u.copy()
        u_copy *= mask

        mean_value = np.mean(np.ma.masked_where(u_copy == 0, u_copy))
        dic[index] = mean_value

    index = max(dic, key=dic.get)

    return pointers[index]


def create_mask_pointer(img_dial):
    height, width, depth = img_dial.shape

    center = (int(width / 2), int(height / 2))
    r = int(width / 2)

    mask = np.zeros((height, width, depth), np.uint8)
    cv2.circle(mask, center, int(r * OUTER_THRESHOLD), (1, 1, 1), -1)
    cv2.circle(mask, center, int(r * INNER_THRESHOLD), (0, 0, 0), -1)

    box = [center, (int(width * .6), height), (width, height), (width, int(height * .6))]
    box = np.array(general_functions.order_rectangle_coordinates(box))  # Remove part of the dial that has no measurements anyway
    cv2.drawContours(mask, [box], 0, (0, 0, 0), -1)

    return mask


def create_mask_contour(img_dial, box):
    height, width, depth = img_dial.shape

    mask = np.zeros((height, width, depth), np.uint8)
    cv2.drawContours(mask, [box], 0, (1, 1, 1), -1)

    return mask


def create_mask_mask(img):
    height, width = img.shape

    center = (int(width / 2), int(height / 2))
    r = int(width / 2)

    mask = np.zeros((height, width), np.uint8)
    cv2.circle(mask, center, int(r * OUTER_THRESHOLD), 1, -1)
    cv2.circle(mask, center, int(r * INNER_THRESHOLD + .01), 0, -1)

    return mask


def create_mask_inner(img):
    height, width = img.shape[:2]

    center = (int(width / 2), int(height / 2))
    r = int(width / 2)

    mask = np.zeros((height, width), np.uint8)
    cv2.circle(mask, center, int(r * BREAKPOINT), 1, -1)
    cv2.circle(mask, center, int(r * INNER_THRESHOLD), 0, -1)

    return mask


def create_mask_edge(height, width):
    center = (int(width / 2), int(height / 2))
    r = int(width / 2)

    mask = np.zeros((height, width), np.uint8)
    cv2.circle(mask, center, int(r * (INNER_THRESHOLD + 0.01)), 1, -1)
    cv2.circle(mask, center, int(r * INNER_THRESHOLD), 0, -1)

    return mask


def create_mask_edge_yuv(height, width):
    center = (int(width / 2), int(height / 2))
    r = int(width / 2)

    mask = np.zeros((height, width), np.uint8)
    cv2.circle(mask, center, int(r * (INNER_THRESHOLD + 0.06)), 1, -1)
    cv2.circle(mask, center, int(r * INNER_THRESHOLD), 0, -1)

    return mask


def create_mask_outer(img):
    height, width = img.shape[:2]

    center = (int(width / 2), int(height / 2))
    r = int(width / 2)

    mask = np.zeros((height, width), np.uint8)
    cv2.circle(mask, center, int(r * OUTER_THRESHOLD), 1, -1)
    cv2.circle(mask, center, int(r * BREAKPOINT), 0, -1)

    return mask
