import numpy as np
import cv2

def _determine_contrast(img, t=2):
    """Determines the contrast within the input image"""
    img_value_mean = np.mean(img)
    img_value_std = np.std(img)

    d = (img_value_mean + (2 * img_value_std)) - (img_value_mean - (2 * img_value_std))

    if d <= (1 / t):
        return True
    else:
        return False


def _gamma_correction(img, gamma):
    """Applies the power transform based on the gamma"""
    img = np.power(img, gamma)
    return img


def _calculate_c(img, mean):
    """Calculates the constant c based on the image statistics"""
    a = 1 - img
    h = _heaviside(.5 - mean)

    k = img + (1 - img) * np.mean(img)
    c = 1 / (1 + (h * (k - 1)))

    return c * img


def _heaviside(x):
    """Performs the heaviside function"""
    if x > 0:
        return 1
    else:
        return 0


def _low_contrast_gamma(std):
    """Gamma cirrection for the low contrast images"""
    gamma = -np.log2(std)
    return gamma


def _high_contrast_gamma(mean, std):
    """Gamma cirrection for the high contrast images"""
    gamma = np.exp((1 - (mean + std)) * .5)
    return gamma


def _convert_to_rgb(hue, saturation, value):
    """Converts HSV to RGB"""
    value = np.asarray(value * 255, np.uint8)
    img_hsv = cv2.merge([hue, saturation, value])
    img_result = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)  # Convert to  RGB color space

    return img_result


def agc_1c_masked(img):
    """Adaptive Gamma Correction for a single channel where the 0 values are excluded from the calculation
    of the correction parameters"""
    # This Algorithm is based on "An adaptive gamma correction for image enhancement" by Rahman et al. (2016).

    if img.dtype is not np.float:
        img = np.asarray(img, np.float)/np.max(img)  # select the Value and scale the values between 0-1

    img = np.ma.masked_where(img == 0, img)

    if _determine_contrast(img):
        img_std = np.std(img)  # Standard Deviation
        img_mean = np.mean(img)
        gamma = _low_contrast_gamma(img_std)  # Determine Gamma based on the Standard Deviation
        img_gamma = _gamma_correction(img, gamma)  # Apply gamma correction
        img_gamma = _calculate_c(img_gamma, img_mean)  # Apply the C constant

        return img_gamma

    else:
        img_std = np.std(img)
        img_mean = np.mean(img)
        gamma = _high_contrast_gamma(img_mean, img_std)
        img_gamma = _gamma_correction(img, gamma)
        img_gamma = _calculate_c(img_gamma, img_mean)

        return img_gamma


def agc_1c(img):
    """Adaptive Gamma Correction for a single channel"""

    # This Algorithm is based on "An adaptive gamma correction for image enhancement" by Rahman et al. (2016).

    if img.dtype is not np.float:
        img = np.asarray(img, np.float)/np.max(img)  # select the Value and scale the values between 0-1

    if _determine_contrast(img):
        img_std = np.std(img)  # Standard Deviation
        img_mean = np.mean(img)
        gamma = _low_contrast_gamma(img_std)  # Determine Gamma based on the Standard Deviation
        img_gamma = _gamma_correction(img, gamma)  # Apply gamma correction
        img_gamma = _calculate_c(img_gamma, img_mean)  # Apply the C constant

        return img_gamma

    else:
        img_std = np.std(img)
        img_mean = np.mean(img)
        gamma = _high_contrast_gamma(img_mean, img_std)
        img_gamma = _gamma_correction(img, gamma)
        img_gamma = _calculate_c(img_gamma, img_mean)

        return img_gamma


def agc(img):
    """Adaptive Gamma Correction for a RGB image"""
    # This Algorithm is based on "An adaptive gamma correction for image enhancement" by Rahman et al. (2016).

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV color space
    channels = cv2.split(img_hsv)
    img_value = np.asarray(channels[2], np.float) / 255  # select the Value and scale the values between 0-1

    if _determine_contrast(img_value):
        img_std = np.std(img_value)  # Standard Deviation
        gamma = _low_contrast_gamma(img_std)  # Determine Gamma based on the Standard Deviation
        img_gamma = _gamma_correction(img_value, gamma)  # Apply gamma correction
        img_gamma = _calculate_c(img_gamma, np.mean(img_value))  # Apply the C constant
        img_result = _convert_to_rgb(channels[0], channels[1], img_gamma)

        return img_result
    else:
        img_std = np.std(img_value)
        img_mean = np.mean(img_value)
        gamma = _high_contrast_gamma(img_mean, img_std)
        img_gamma = _gamma_correction(img_value, gamma)
        img_gamma = _calculate_c(img_gamma, img_mean)
        img_result = _convert_to_rgb(channels[0], channels[1], img_gamma)

        return img_result


def agc_dials(img):
    """Adaptive Gamma Correction for RGB image of a cropped dial, 0 values are excluded from the calculation
    of gamma correction parameters"""
    # This Algorithm is based on "An adaptive gamma correction for image enhancement" by Rahman et al. (2016).

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV color space
    channels = cv2.split(img_hsv)
    img_value = np.asarray(channels[2], np.float) / 255  # select the Value and scale the values between 0-1
    img_value = np.ma.masked_where(img_value == 0, img_value)

    if _determine_contrast(img_value):
        img_std = np.std(img_value)  # Standard Deviation
        gamma = _low_contrast_gamma(img_std)  # Determine Gamma based on the Standard Deviation
        img_gamma = _gamma_correction(img_value, gamma)  # Apply gamma correction
        img_gamma = _calculate_c(img_gamma, np.mean(img_value))  # Apply the C constant
        img_result = _convert_to_rgb(channels[0], channels[1], img_gamma)

        return img_result
    else:
        img_std = np.std(img_value)
        img_mean = np.mean(img_value)
        gamma = _high_contrast_gamma(img_mean, img_std)
        img_gamma = _gamma_correction(img_value, gamma)
        img_gamma = _calculate_c(img_gamma, img_mean)
        img_result = _convert_to_rgb(channels[0], channels[1], img_gamma)

        return img_result

