import numpy as np


def order_rectangle_coordinates(box):
    """ Orders the coordinates of a rectangle to go clockwise"""

    # Code found at: https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
    # the method is based on a function of the imutils library, however some adjustments have made as the method did not
    # function correctly

    box.sort(key=lambda x: x[0])  # sort the points based on their x-coordinates
    xSorted = box

    leftMost = xSorted[0:2]  # select the left-most and right-most points from the sorted x-coordinate points
    rightMost = xSorted[2:4]

    leftMost.sort(key=lambda x: x[1])  # sort the left-most coordinates according based on y
    (tl, bl) = leftMost  # and select the top-left and bottom-left points

    rightMost.sort(key=lambda x: x[1])
    (tr, br) = rightMost

    return np.array([tl, tr, br, bl])  # return the coordinates


def calc_fourier_descriptors(boundary, number_of_descriptors):
    """Calculates the Fourier descriptors for a boundary array"""

    if not number_of_descriptors % 2 == 0 or type(number_of_descriptors) is not int:
        raise ValueError("number of descriptors must be an integer and divisible by two!")

    if len(boundary) < number_of_descriptors:
        return [0], [0]

    boundary_complex = []

    for point in boundary:  # Convert to complex numbers
        point_complex = point[0][0] + point[0][1] * 1j
        boundary_complex.append(point_complex)

    boundary_complex = np.asarray(boundary_complex, dtype=np.complex)  # converts to a ndarray
    fft = np.fft.fft(boundary_complex)  # Applies the Fourier Transform and Shift to the boundary ndarray
    fft = np.fft.fftshift(fft)

    W = len(fft)
    half_d = int(number_of_descriptors / 2)

    fourier_descriptors = fft[int(W / 2) - (half_d + 1):int(W / 2) + half_d + 1]  # selects the Fourier descriptors

    max_loc = np.argmax(np.absolute(fourier_descriptors))
    fourier_descriptors = np.delete(fourier_descriptors, max_loc)  # remove the DC
    fourier_descriptors[:] = [i / max(fourier_descriptors, key=abs) for i in
                              fourier_descriptors]  # Normalize for largest value
    max_loc = np.argmax(np.absolute(fourier_descriptors))
    fourier_descriptors = np.delete(fourier_descriptors, max_loc)

    fourier_x = np.ndarray.tolist(
        np.real(fourier_descriptors))  # Split up the complex numbers and turn them to list
    fourier_y = np.ndarray.tolist(np.imag(fourier_descriptors))

    return fourier_x, fourier_y
