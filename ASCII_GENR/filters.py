import cv2
import numpy as np
from performance import GlobalTimer

@GlobalTimer.time
def difference_of_gaussians(image, sigma=2.0, sigma_scale=1.6, tau=1.0, threshold=0.005):
    """
    Apply Difference of Gaussians (DoG) filter.
    """
    # Ensure image is grayscale and normalized to 0-1
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    sigma2 = sigma * sigma_scale
    gaussian1 = cv2.GaussianBlur(image, (0, 0), sigma)
    gaussian2 = cv2.GaussianBlur(image, (0, 0), sigma2)

    dog = gaussian1 - tau * gaussian2
    edges = (dog >= threshold).astype(np.uint8)

    return edges

@GlobalTimer.time
def sobel_edge_detection_shader_style(image):
    """
    Apply Sobel edge detection matching the Unity shader's separable approach.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    magnitude_normalized = cv2.normalize(
        magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    theta = np.arctan2(gradient_y, gradient_x)
    # Fix: A pixel is only a valid edge if its magnitude is non-zero
    mask = ((magnitude > 0) & (~np.isnan(theta))).astype(np.uint8)

    return magnitude_normalized, theta, mask

@GlobalTimer.time
def quantize_edge_direction(theta, mask):
    """
    Quantize edge angles to 4 discrete directions.
    """
    height, width = theta.shape
    direction_map = np.full((height, width), -1, dtype=np.int8)
    valid_edge_mask = mask > 0

    if not np.any(valid_edge_mask):
        return direction_map

    valid_thetas = theta[valid_edge_mask]
    abs_theta_normalized = np.abs(valid_thetas) / np.pi

    # Direction 0: VERTICAL
    vertical_cond = ((abs_theta_normalized >= 0.0) & (abs_theta_normalized < 0.05)) | \
                    ((abs_theta_normalized > 0.9) & (abs_theta_normalized <= 1.0))
    # Direction 1: HORIZONTAL
    horizontal_cond = (abs_theta_normalized > 0.45) & (abs_theta_normalized < 0.55)
    # Direction 2/3: DIAGONALS
    lower_diag_cond = (abs_theta_normalized > 0.05) & (abs_theta_normalized < 0.45)
    upper_diag_cond = (abs_theta_normalized > 0.55) & (abs_theta_normalized < 0.9)

    quantized_valid = np.full(valid_thetas.shape, -1, dtype=np.int8)
    quantized_valid[vertical_cond] = 0
    quantized_valid[horizontal_cond] = 1
    quantized_valid[lower_diag_cond & (valid_thetas > 0)] = 2
    quantized_valid[lower_diag_cond & (valid_thetas <= 0)] = 3
    quantized_valid[upper_diag_cond & (valid_thetas > 0)] = 3
    quantized_valid[upper_diag_cond & (valid_thetas <= 0)] = 2

    direction_map[valid_edge_mask] = quantized_valid
    return direction_map
