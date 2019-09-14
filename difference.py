import cv2
import numpy as np
from non_maximal_suppression import *
from utils import *
from Parameters import *
from skimage import morphology


def get_diff_locations(diff_patches, tracking_point, bounding_box_old_big_patch_abs_cords):
    # Registration using ECC (Enhanced Correlation Coefficient Maximization) - http://xanthippi.ceid.upatras.gr/people/evangelidis/george_files/PAMI_2008.pdf
    old_patch = diff_patches[0]
    new_patch = diff_patches[1]

    # Find size of image1
    sz = new_patch.shape

    try:
        # Define the motion model
        warp_mode = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Specify the number of iterations.
        number_of_iterations = 10

        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = 1e-10

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC(new_patch, old_patch, warp_matrix, warp_mode, criteria)
        # Use warpAffine for Translation, Euclidean and Affine
        old_patch_aligned = cv2.warpAffine(old_patch, warp_matrix, (sz[1], sz[0]),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    except:
        # print('ECC Failed, continue without registration and diff features')
        old_patch_aligned = new_patch

    # Diff image
    diff_image = abs(np.float64(old_patch_aligned) - np.float64(new_patch))


    tracking_point_local_cords_old_patch = abs_cords_to_local_cords([tracking_point], bounding_box_old_big_patch_abs_cords)
    [ymin, ymax, xmin, xmax] = calc_cropped_patch_params(tracking_point_local_cords_old_patch[0], diff_image, REF_PATCH_SIZE_CROP)
    diff_image = diff_image[ymin:ymax, xmin:xmax]
    bounding_box_diff_abs_cords = [0, 0, 0, 0]
    bounding_box_diff_abs_cords[0] = bounding_box_old_big_patch_abs_cords[0] + ymin
    bounding_box_diff_abs_cords[1] = bounding_box_old_big_patch_abs_cords[0] + ymax
    bounding_box_diff_abs_cords[2] = bounding_box_old_big_patch_abs_cords[2] + xmin
    bounding_box_diff_abs_cords[3] = bounding_box_old_big_patch_abs_cords[2] + xmax
    if diff_image.size == 0:
        interest_points = []
        return interest_points, diff_image, bounding_box_diff_abs_cords
    # threshold - 99.9% percent
    diff_image_threshold = np.percentile(diff_image, 99.9)
    retval, bw_diff_img = cv2.threshold(diff_image, diff_image_threshold, 255, cv2.THRESH_BINARY)

    # remove_small_objects instead of bwareopen() in matlab
    bw_diff_thresholded = morphology.remove_small_objects(bw_diff_img.astype(np.int32), min_size=3, connectivity=8)
    bw_diff_thresholded = np.uint8(bw_diff_thresholded)

    # dilation
    kernel = np.ones((7, 7), np.uint8)
    bw_diff_dilated = cv2.dilate(bw_diff_thresholded, kernel, iterations=1)

    interest_points = []
    try:
        retval, bw_diff_labels, stats, centroids = cv2.connectedComponentsWithStats(bw_diff_dilated)
        if centroids.shape[0] < GROUPS_NUM:
            for blob_num in range(centroids.shape[0]):
                area = stats[blob_num, 4]
                if area > 3 and area < 5000:
                    interest_points.append(centroids[blob_num])
                interest_points = replace_interest_points_cords_xy(interest_points)
        return interest_points, diff_image, bounding_box_diff_abs_cords
    except:
        # print('didnt find connected components')
        return interest_points, diff_image, bounding_box_diff_abs_cords
