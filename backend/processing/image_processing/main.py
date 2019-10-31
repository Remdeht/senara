# Main Algorithm - Dial Reading
# Necessary libraries
import cv2

# Modules used in the detection
from .dial import dial, pointer, pointer_red, tally_counter, zero
from . import helpers


# Main function for the execution of the detection

def detect(image, fd, n, flag_knn, flag_templates):
    try:
        pointer_value, tally_values = main(image, fd, n, flag_knn, flag_templates)
    except Exception as e:
        raise
    else:
        return pointer_value, tally_values


def main(image_name, fd, n, flag_knn, flag_templates):
    """Main function that initiates all other processes in the image processing algorithm"""
    image = cv2.imread(image_name)  # Reads in the image

    # Step 1: Identify the dial in the input image and return a cropped image containing only the dial.
    mask_dial, mask_cap, x_cap, y_cap, r_dial = dial.detect_dial(image)

    if mask_dial is None:
        raise ValueError("No Dial was detected in the Image")

    img_dial, mask_cap, x_cap, y_cap = helpers.crop_image(image, mask_dial, x_cap, y_cap, r_dial, mask_cap)

    # Step 2: Identify the red pointer and determine it's center for the estimation of the dial position.
    center_pointer = pointer_red.get_center_pointer(img_dial, mask_cap)

    # Step 3: Based on the position of the red dial and the center of the dial a rotation is performed.
    img_dial = helpers.rotate_dial(img_dial, x_cap, y_cap, center_pointer[0], center_pointer[1])

    # Step 4: Read the Tally Counter
    img_dial, center_tally, tally_values = tally_counter.extract_counter(img_dial, fd, n, flag_knn, flag_templates)

    if img_dial is None or center_tally is None:
        raise ValueError("No Dial was detected in the Image")

    # Step 5: Detect the location of the 0 point
    zero_coordinates = zero.main_process(img_dial, center_tally)

    # Step 6: Identify the pointer and read the value based on it's position.
    pointer_value = pointer.read_pointer(img_dial, zero_coordinates)

    return pointer_value, tally_values
