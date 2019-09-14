import numpy as np
import cv2
from UserParams import *
from Parameters import *
from video_info import *
from utils import *
from interest_points import *
from MTT import *
from matplotlib import pyplot
from PIL import Image
import time
import os
from joblib import load

# input movie numbers: list of according to the definition in "video_info.py"
# valid input_movie_numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,17,18,19,20,21,22,23,24,25,26,27]
input_movie_numbers = [1]


# load alexnet serialized model from disk
prototxt_alexnet_fc7 = r"C:\Users\roiei\Desktop\caffe\caffe\models\bvlc_alexnet\deploy_fc7.prototxt"
model_alexnet = r"C:\Users\roiei\Desktop\caffe\caffe\models\bvlc_alexnet\bvlc_alexnet.caffemodel"
alexnet_fc7 = cv2.dnn.readNetFromCaffe(prototxt_alexnet_fc7, model_alexnet)

# load SVM classifier from disk
svm_classifier = load(r'drone_tracker_svm_classifier\alexnet_svm.joblib')

for movie_number in input_movie_numbers:

    # Pre-Proccesing
    print('Processing movie number ' + str(movie_number))
    [start_frame, end_frame, tracking_point, video_filename] = get_video_info(movie_number)
    if start_frame == -1 or start_frame == end_frame:
        print("skipping movie " + str(movie_number) + ". Invalid video info. ")
        continue
    cap = cv2.VideoCapture(video_filename)

    # Output movie setup
    output_video = 'output_videos/movie_' + str(movie_number) + time.strftime("_%d%m%Y_%H%M%S") + '.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, FPS, (1920, 1080))

    # get initial rectangle around the drone and convert to grayscale
    cap.set(1, start_frame)
    retval, first_frame = cap.read()
    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # crop 100X100 patch
    [ymin, ymax, xmin, xmax] = calc_cropped_patch_params(tracking_point, first_frame_gray, REF_PATCH_SIZE_CROP)
    bounding_box_small_patch_abs_cords = [ymin, ymax, xmin, xmax]
    first_patch = first_frame_gray[ymin:ymax, xmin:xmax]

    # save 200X200 image around the drone to get diff features from it
    [ymin, ymax, xmin, xmax] = calc_cropped_patch_params(tracking_point, first_frame_gray, REF_PATCH_SIZE_CROP_2)
    bounding_box_old_big_patch_abs_cords = [ymin, ymax, xmin, xmax]
    old_patch = first_frame_gray[ymin:ymax, xmin:xmax]

    tracks = []
    t_cur = DIFFERENTIAL_TIME

    tic = time.time()
    # main processing loop
    for frame_num in range(start_frame + 1, end_frame):
        # get next frame
        retval, ref_img_frame = cap.read()
        ref_img_frame_clone = ref_img_frame.copy()
        ref_img_frame_gray = cv2.cvtColor(ref_img_frame, cv2.COLOR_BGR2GRAY)

        # 100X100
        [ymin, ymax, xmin, xmax] = calc_cropped_patch_params(tracking_point, ref_img_frame_gray, REF_PATCH_SIZE_CROP)
        bounding_box_small_patch_abs_cords = [ymin, ymax, xmin, xmax]
        ref_img_patch = ref_img_frame_gray[ymin:ymax, xmin:xmax]

        # 200X200
        [ymin, ymax, xmin, xmax] = calc_cropped_patch_params(tracking_point, ref_img_frame_gray, REF_PATCH_SIZE_CROP_2)
        bounding_box_new_big_patch_abs_cords = [ymin, ymax, xmin, xmax]
        new_patch = ref_img_frame_gray[ymin:ymax, xmin:xmax]
        diff_patches = [old_patch, new_patch]

        # Detection
        keypoint_list, diff_image, bounding_box_diff_abs_cords = find_interest_points(ref_img_patch, diff_patches, tracking_point, bounding_box_small_patch_abs_cords, bounding_box_old_big_patch_abs_cords)

        # Compute Descriptors
        keypoint_list, descriptors = compute_descriptors(ref_img_frame, keypoint_list)

        # Fill list for MTT
        array_for_mtt = fill_list_for_mtt(keypoint_list, tracking_point, bounding_box_small_patch_abs_cords, diff_image, bounding_box_diff_abs_cords)

        # MTT
        tracks, label = mtt_tracker(tracks, array_for_mtt, t_cur, frame_num, descriptors, tracking_point, alexnet_fc7, svm_classifier, ref_img_frame)
        t_cur = t_cur + DIFFERENTIAL_TIME

        # Target selection
        if len(tracks) != 0:
            if tracks[0].k > MINIMUM_TRACK_LIFE:
                selected_label_frame = tracks[0].label_frame
                selected_label_group = tracks[0].label_group
                x_centroid = tracks[0].x_centroid_predicted
                y_centroid = tracks[0].y_centroid_predicted
                tracking_point = [(tracking_point[0]+y_centroid)/2, (tracking_point[1] + x_centroid)/2]

        # Tracking image
        locations = []
        for i in range(len(tracks)):
            if tracks[i].track_status[0][0] == 1:
                locations.append([tracks[i].y_centroid_predicted, tracks[i].x_centroid_predicted])
        bounding_box_top_left = (bounding_box_small_patch_abs_cords[2], bounding_box_small_patch_abs_cords[0])
        bounding_box_bottom_right = (bounding_box_small_patch_abs_cords[3], bounding_box_small_patch_abs_cords[1])
        cv2.rectangle(ref_img_frame_clone, bounding_box_top_left, bounding_box_bottom_right, (0, 255, 0), 3)
        if len(locations) > 0:
            loc_y = int(np.round(locations[0][0]))
            loc_x = int(np.round(locations[0][1]))
            cv2.circle(ref_img_frame_clone, (loc_x, loc_y), 3, (0, 0, 255), -1)
            cv2.putText(ref_img_frame_clone, label, (loc_x, loc_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0))
            for i in range(len(locations) - 1):
                loc_y = int(np.round(locations[i+1][0]))
                loc_x = int(np.round(locations[i+1][1]))
                cv2.circle(ref_img_frame_clone, (loc_x, loc_y), 3, (255, 0, 0), -1)

        out.write(ref_img_frame_clone)
        # update patch queue
        old_patch = new_patch
        bounding_box_old_big_patch_abs_cords = bounding_box_new_big_patch_abs_cords
    toc = time.time()
    print(toc-tic)
