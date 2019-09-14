import math
import cv2
import numpy as np


def calc_cropped_patch_params(center_point, frame_to_process, patch_size):
    # This function accepts an image, a center point and the desired size of the image after crop.
    # It returns a list with the coordinates of the bounding box - [ymin, ymax, xmin, xmax]

    # Inputs:
    # center point - list with 2 integers: [center_point_y, center_point_x]
    # frame_to_process - ndarray, usually 2D (grayscaled image)
    # patch_size - list with 2 integers: [cropped_image_height, cropped_image_width]
    # Output:
    # list of 4 integers: [ymin, ymax, xmin, xmax]

    half_patch_size = [math.ceil(patch_size[0] / 2), math.ceil(patch_size[1] / 2)]
    y_half_len = min(half_patch_size[0], center_point[0], frame_to_process.shape[0] - center_point[0])
    x_half_len = min(half_patch_size[1], center_point[1], frame_to_process.shape[1] - center_point[1])

    ymin = center_point[0] - y_half_len
    ymax = center_point[0] + y_half_len
    xmin = center_point[1] - x_half_len
    xmax = center_point[1] + x_half_len

    return [int(ymin), int(ymax), int(xmin), int(xmax)]


def interest_points_list_to_kepoints_list(interest_points_list):
    keypoint_list = []
    for keypoint_num in range(len(interest_points_list)):
        keypoint = interest_points_list[keypoint_num]
        keypoint_list.append(cv2.KeyPoint(keypoint[1], keypoint[0], 31))
    return keypoint_list


def local_cords_to_abs_cords(local_cords, bounding_box_abs_cords):
    if local_cords == []:
        return []
    else:
        [ymin, ymax, xmin, xmax] = bounding_box_abs_cords
        abs_cords = []
        for i in range(len(local_cords)):
            abs_cords_i = [0, 0]
            abs_cords_i[0] = ymin + local_cords[i][0]
            abs_cords_i[1] = xmin + local_cords[i][1]
            abs_cords.append(abs_cords_i)
        return abs_cords


def abs_cords_to_local_cords(abs_cords, bounding_box):
    if abs_cords == []:
        return []
    else:
        [ymin, ymax, xmin, xmax] = bounding_box
        local_cords = []
        for i in range(len(abs_cords)):
            local_cords_i = [0, 0]
            local_cords_i[0] = abs_cords[i][0] - ymin
            local_cords_i[1] = abs_cords[i][1] - xmin
            local_cords.append(local_cords_i)
        return local_cords


def replace_interest_points_cords_xy(patch_interest_points_locations):
    for i in range(len(patch_interest_points_locations)):
        patch_interest_points_locations[i] = (patch_interest_points_locations[i][1], patch_interest_points_locations[i][0])
    return patch_interest_points_locations


def save_img_with_keypoints(image, keypoints_abs_cords):
    # This utility is for keypoint visualisation (for debug purposes)
    img_with_keypoints = cv2.drawKeypoints(image, keypoints_abs_cords, outImage=np.array([]), color=(0, 0, 255))
    cv2.imwrite("debug/img_with_keypoints.png", img_with_keypoints)
    return
