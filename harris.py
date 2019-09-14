import cv2
import numpy as np
from non_maximal_suppression import *
from utils import *

# Harris Params
HARRIS_BLOCK_SIZE = 2  # Neighborhood size - for each pixel HARRIS_BLOCK_SIZE * HARRIS_BLOCK_SIZE pixels are considered
HARRIS_SOBEL_K_SIZE = -1  # Aperture parameter for the Sobel() operator. -1 uses scharr function with 3X3 window.
HARRIS_K = 0.04  # Harris detector free parameter
NMS_RADIUS = 5


def get_harris_locations(patch_gray):
    patch_gray_float32 = np.float32(patch_gray)
    patch_harris = cv2.cornerHarris(patch_gray_float32, HARRIS_BLOCK_SIZE, HARRIS_SOBEL_K_SIZE, HARRIS_K)
    patch_harris_threshold = np.percentile(patch_harris, 99)
    patch_harris_locations = nms(patch_harris, NMS_RADIUS, patch_harris_threshold)
    return patch_harris_locations

