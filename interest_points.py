from harris import *
from difference import *
from utils import *
import numpy as np
from Parameters import *
import cv2


def find_interest_points(ref_img_patch, diff_patches, tracking_point, bounding_box_small_patch_abs_cords, bounding_box_old_big_patch_abs_cords):

    # Harris
    # get interest points coordinates in the cropped patch ("local coordinates")
    patch_harris_locations = get_harris_locations(ref_img_patch)
    # get interest points coordinates in the original image ("absolute coordinates")
    harris_abs_locations = local_cords_to_abs_cords(patch_harris_locations, bounding_box_small_patch_abs_cords)

    # Difference
    diff_interest_points, diff_image, bounding_box_diff_abs_cords = get_diff_locations(diff_patches, tracking_point, bounding_box_old_big_patch_abs_cords)
    diff_abs_locations = local_cords_to_abs_cords(diff_interest_points, bounding_box_diff_abs_cords)

    # Fusion
    keypoints_locations = interest_points_fusion_logic(harris_abs_locations, diff_abs_locations)

    # Generate keypoints list from coordinates
    keypoint_list = interest_points_list_to_kepoints_list(keypoints_locations)

    return keypoint_list, diff_image, bounding_box_diff_abs_cords


def compute_descriptors(ref_img_frame, keypoint_list):
    ref_img_frame_uint8 = np.uint8(ref_img_frame)
    # Compute the descriptors with ORB - http://www.willowgarage.com/sites/default/files/orb_final.pdf
    orb = cv2.ORB_create(edgeThreshold=5, patchSize=5)
    keypoint_list, orb_descriptors = orb.compute(ref_img_frame_uint8, keypoint_list)
    return keypoint_list, orb_descriptors


def interest_points_fusion_logic(harris_abs_locations, diff_abs_locations):
    if diff_abs_locations == []:
        interest_points = harris_abs_locations
    else:
        interest_points = diff_abs_locations
        for har_loc in harris_abs_locations:
            new_point = 1
            for dif_loc in interest_points:
                if (abs(har_loc[0] - dif_loc[0]) < 3) and (abs(har_loc[1] - dif_loc[1]) < 3):
                    new_point = 0
                    break
            if new_point == 1:
                interest_points.append(har_loc)
    return interest_points


