# Main Program - Dial Reading
# ibraries
import os
import cv2
from pathlib import Path

# Root directory
from constants import ROOT_DIR

# Sripts used in the detection
from dial import dial, pointer, pointer_red, tally_counter, zero
import helpers


# Main function for the execution of the detection
def detect(image, fd, n, flag_knn, flag_templates, name=None):
    try:
        pointer_value, tally_values = main(image_loc=image, fd=fd, n=n, flag_knn=flag_knn,
                                           flag_templates=flag_templates, name=name)
    except Exception as e:
        print(e)
        raise
    else:
        return pointer_value, tally_values


def main(image_loc, fd, n, flag_knn, flag_templates, name=None):
    """Main Function that initiates all other processes in the image processing algorithm"""
    image = cv2.imread(image_loc)  # Reads in the image

    # Step 1: Identify the dial in the input image
    mask_dial, mask_cap, x_cap, y_cap, r_dial = dial.detect_dial(image, name)

    if mask_dial is None:
        raise ValueError("No Dial was detected in the Image")

    # Step 1.5: cropping of the image based on the center of the dial and the radius
    img_dial, mask_cap, x_cap, y_cap = helpers.crop_image(image, mask_dial, x_cap, y_cap, r_dial, mask_cap, name)

    # Step 2: Identify the red pointer and determine it's center for the estimation of the dial position.
    center_pointer = pointer_red.get_center_pointer(img_dial, mask_cap, name)

    # Step 3: Based on the position of the red dial and the center of the dial a rotation is performed.
    img_dial = helpers.rotate_dial(img_dial, x_cap, y_cap, center_pointer[0], center_pointer[1])

    # Step 4: Read the Tally Counter
    img_dial, center_tally, tally_values = tally_counter.extract_counter(img_dial, fd, n,
                                                                         flag_knn, flag_templates, name)

    if img_dial is None or center_tally is None:
        raise ValueError("No Dial was detected in the Image")

    # Step 5: Detect the location of the 0 point
    zero_coordinates = zero.main_process(img_dial, center_tally)

    # Step 6: Identify the pointer and read the value based on it's position.
    pointer_value = pointer.read_pointer(img_dial, zero_coordinates, name)

    if name is not None:
        helpers.write_results(pointer_value=pointer_value, tally_values=tally_values,
                      fd=fd, n=n, img_dial=img_dial, name=name, flag=flag_templates)

    return pointer_value, tally_values


if __name__ == '__main__':

    DATA_DIR = Path(ROOT_DIR + "\\data\\water_meters")
    water_meters = os.listdir(DATA_DIR)

    fourier_descriptors = [10, 20, 30, 40, 50]  # Fourier Descriptors
    neighbours = [5, 7, 9, 11]  # Number of Neighbours
    flags_templates = [0, 1]  # Template set, 0 for structurally generated templates, 1 for randomly placed templates
    flags_knn = [0, 1]  # Weighting of kNN classification, 0 uniformly weighted and 1 for distance weighted

    for flag_t in flags_templates:
        for flag_k in flags_knn:
            for i in fourier_descriptors:
                for j in neighbours:
                    for index, water_meter in enumerate(water_meters):
                        print("File no. %d" % (index + 1))
                        image_location = str(DATA_DIR.joinpath(water_meter))
                        print(image_location)
                        try:
                            detect(image=image_location, name=water_meter, fd=i, n=j,
                                   flag_knn=flag_k, flag_templates=flag_t)
                        except Exception as e:

                            print(e)

