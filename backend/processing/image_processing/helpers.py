import math
import imutils


def rotate_dial(img_dial, x_cap, y_cap, x_pointer, y_pointer):
    """Rotates the image of the dial based on the position of the red pointer"""
    try:
        height, width = img_dial.shape[:2]

        sideA = math.sqrt((abs(y_cap - y_pointer) ** 2) + (abs(x_cap - x_pointer) ** 2))
        sideB = abs(y_cap - height)
        sideC = math.sqrt((abs(x_cap - x_pointer) ** 2) + (abs(y_pointer - height) ** 2))
        cosine_alfa = ((sideC ** 2) - (sideA ** 2) - (sideB ** 2)) / (-2 * sideA * sideB)
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


def crop_image(image, mask_dial, x_cap, y_cap, r_dial, mask_cap):
    """Crops the original image and mask to the dial and adjusts the center coordinates of the dial"""

    img_dial = image * mask_dial
    img_dial = crop_to_circle(x_cap, y_cap, r_dial, img_dial)
    mask_cap = crop_to_circle(x_cap, y_cap, r_dial, mask_cap)
    height, width = img_dial.shape[:2]
    x_cap = int(width / 2)
    y_cap = int(height / 2)

    return img_dial, mask_cap, x_cap, y_cap
