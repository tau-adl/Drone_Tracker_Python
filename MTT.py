from Parameters import *
import numpy as np
from utils import *


class Track:

    def __init__(self, label_frame, label_group, x_c, y_c, state_vector, current_state_vector, x_c_predicted,
                 y_c_predicted, t, time, k, q, GR, signal, track_status, features):
        self.label_frame = label_frame
        self.label_group = label_group
        self.x_centroid = [x_c]
        self.y_centroid = [y_c]
        self.state_vector = state_vector
        self.current_state_vector = current_state_vector
        self.x_centroid_predicted = x_c_predicted.tolist()
        self.y_centroid_predicted = y_c_predicted.tolist()
        self.t = t
        self.time = [time]
        self.k = k
        self.q = q
        self.GR = GR
        self.signal = signal
        self.track_status = track_status
        self.features = features


def fill_list_for_mtt(keypoint_list, tracking_point, bounding_box_small_patch_abs_cords, diff_image, bounding_box_diff_abs_cords):
    array_for_mtt = np.zeros((3, len(keypoint_list)))
    [ymin, ymax, xmin, xmax] = bounding_box_small_patch_abs_cords
    index = 0
    for i in range(len(keypoint_list)):
        [x_centroid, y_centroid] = [keypoint_list[i].pt[0], keypoint_list[i].pt[1]]
        if ymin <= y_centroid < ymax and xmin <= x_centroid < xmax:
            array_for_mtt[0, index] = y_centroid
            array_for_mtt[1, index] = x_centroid
            diff_window_size = [11, 11]
            [keypoint_local_cords] = abs_cords_to_local_cords([[y_centroid, x_centroid]], bounding_box_diff_abs_cords)
            [key_ymin, key_ymax, key_xmin, key_xmax] = calc_cropped_patch_params(keypoint_local_cords, diff_image,
                                                                                 diff_window_size)
            keypoint_diff_window = diff_image[key_ymin:key_ymax, key_xmin:key_xmax]
            try:
                array_for_mtt[2, index] = np.max(keypoint_diff_window)
            except:
                array_for_mtt[2, index] = 0
            index = index + 1
        else:
            np.delete(array_for_mtt, array_for_mtt.shape[1]-1, 1)
    return array_for_mtt


def mtt_tracker(tracks, array_for_mtt, t_cur, frame_num, descriptors, tracking_point, alexnet_fc7, svm_classifier, ref_img_frame):
    label = "background"
    if len(tracks) == 0:
        for i in range(array_for_mtt.shape[1]):
            new_track = Track(frame_num, i, array_for_mtt[1, i], array_for_mtt[0, i], np.zeros((1, 6)),
                              np.zeros((1, 6)),
                              array_for_mtt[1, i], array_for_mtt[0, i], 1, t_cur, 1, 0, GATING_RADIUS,
                              array_for_mtt[2, i], np.zeros((1, 4)), descriptors[i, :])
            tracks.append(new_track)
        tracks, label = sort_tracks(tracks, tracking_point, alexnet_fc7, svm_classifier, ref_img_frame)
        return tracks, label

    check_list = np.arange(array_for_mtt.shape[1])
    del_list = np.zeros((1, len(tracks)))

    for i in range(len(tracks)):
        if array_for_mtt.shape[1] == 0:
            break
        cur_track = tracks[i]
        last_x_c = cur_track.x_centroid_predicted
        last_y_c = cur_track.y_centroid_predicted
        start_window = cur_track.GR
        track_features = cur_track.features

        associated_group_index = match_tracks(start_window, array_for_mtt[:, check_list], last_x_c, last_y_c, track_features, descriptors[check_list, :])
        associated_group = check_list[associated_group_index]
        cur_features = descriptors[associated_group]

        if associated_group.size != 0:
            group_features = array_for_mtt[:, associated_group]
            mask = np.ones(len(check_list), dtype=bool)
            mask[associated_group_index] = False
            check_list = check_list[mask]
            cur_track = fill_track(cur_track, group_features, t_cur, 0, cur_features)

        else:
            cur_track = fill_track(cur_track, np.zeros(0), t_cur, 1, cur_features)

        if (cur_track.q >= N_TRK_DEL_FOR_NEW_TRACK and cur_track.track_status[0][0] == 0) or (
                cur_track.q >= N_TRK_DEL and cur_track.track_status[0][0] == 1):
            del_list[0, i] = 1
        else:
            cur_track = smooth_filter_mtt(cur_track)

        tracks[i] = cur_track

    tracks_copy = tracks[:]
    removed = 0
    for i in range(len(tracks_copy)):
        if del_list[0, i] == 1:
            del tracks[i - removed]
            removed = removed + 1

    new_list = array_for_mtt[:, check_list]
    if new_list.size != 0:
        new_features = descriptors[check_list, :]

    unavailable_track_numbers = []
    for i in range(len(tracks)):
        unavailable_track_numbers = unavailable_track_numbers + [tracks[i].label_group]

    if new_list.size != 0:
        track_num_available = max(unavailable_track_numbers) + 1
        list_len = new_list.shape[1]
        for i in range(list_len):
            new_track = Track(frame_num, track_num_available, new_list[1, i], new_list[0, i], np.zeros((1, 6)),
                              np.zeros((1, 6)),
                              new_list[1, i], new_list[0, i], 1, t_cur, 1, 0, GATING_RADIUS,
                              new_list[2, i], np.zeros((1, 4)), new_features[i, :])
            track_num_available = track_num_available + 1
            tracks.append(new_track)

    if len(tracks) != 0:
        tracks, label = sort_tracks(tracks, tracking_point, alexnet_fc7, svm_classifier, ref_img_frame)

    return tracks, label


def sort_tracks(tracks, tracking_point, alexnet_fc7, svm_classifier, ref_img_frame):
    # remove tracks that are outside the patch
    tracks_copy = tracks[:]
    removed = 0
    for i in range(len(tracks_copy)):
        track = tracks_copy[i - removed]
        if abs(track.y_centroid[-1] - tracking_point[0]) > 50 or abs(track.x_centroid[-1] - tracking_point[1]) > 50:
            del tracks[i - removed]
            removed = removed + 1

    # sort tracks by number of frames per tracker, and then by signal
    # secondary key - signal
    tracks = sorted(tracks, key=lambda track: track.signal, reverse=True)
    # primary key - t
    tracks = sorted(tracks, key=lambda track: track.t, reverse=True)

    patch_to_classify_crop_size = [40, 40]
    label = ["background"]
    # run for 5 most durable trackers
    if len(tracks) != 0:
        for track_num in range(min(5, len(tracks))):
            # the NN and SVM goes here
            [ymin, ymax, xmin, xmax] = calc_cropped_patch_params(tracking_point, ref_img_frame, patch_to_classify_crop_size)
            patch_to_classify = ref_img_frame[ymin:ymax, xmin:xmax, :]
            patch_to_classify_resized = cv2.resize(patch_to_classify, (224, 224))
            blob_to_classify = cv2.dnn.blobFromImage(patch_to_classify_resized, 1, (224, 224), (104, 117, 123))
            alexnet_fc7.setInput(blob_to_classify)
            alexnet_fc7_activations = alexnet_fc7.forward()
            label = svm_classifier.predict(alexnet_fc7_activations)
            if label[0] == "drone":
                tracks.insert(0, tracks.pop(track_num))
                break

    number_of_tracks = min(MAX_NUMBER_OF_TRACKS, len(tracks))
    new_tracks = tracks[0:number_of_tracks]
    return new_tracks, label[0]


def match_tracks(window, cg_list, x_p, y_p, track_features, features):
    associated_group = []
    y_cg_list = cg_list[0, :]
    x_cg_list = cg_list[1, :]
    if cg_list.shape[1] >= 1:
        features = features.astype(float)
        track_features = track_features.astype(float)
        dis = np.sqrt((x_cg_list - x_p) ** 2 + (y_cg_list - y_p) ** 2)
        dis_normalized = dis / window
        orb_dis = np.sqrt(np.sum((features - track_features) ** 2, 1))
        orb_dis_normalized = orb_dis / 512
        feature_dis = orb_dis_normalized + dis_normalized
        for i in range(len(dis)):
            if dis[i] > window:
                feature_dis[i] = 999
        min_dist = np.min(feature_dis)
        associated_group = np.argmin(feature_dis)
        if min_dist > 2.0:
            associated_group = []
    return associated_group


def fill_track(cur_track, group_features, t_cur, q_cur, cur_features):
    if group_features.size != 0:
        cur_track.signal = 0.5 * cur_track.signal + group_features[2]
        cur_track.x_centroid = cur_track.x_centroid + [group_features[1]]
        cur_track.y_centroid = cur_track.y_centroid + [group_features[0]]
        cur_track.k = cur_track.k + 1
        cur_track.features = cur_features

    else:
        cur_track.signal = 0.5 * cur_track.signal
        cur_track.x_centroid = cur_track.x_centroid + [cur_track.x_centroid[-1]]
        cur_track.y_centroid = cur_track.y_centroid + [cur_track.y_centroid[-1]]

    cur_track.t = cur_track.t + 1
    cur_track.time = cur_track.time + [t_cur]

    if q_cur > 0:
        cur_track.q = cur_track.q + 1
        cur_track.GR = min(100, cur_track.GR + 20)
    else:
        cur_track.q = 0
        cur_track.GR = max(10, cur_track.GR - 5)

    if cur_track.k >= N_TRK_CONF:
        cur_track.track_status[0] = 1

    return cur_track


def smooth_filter_mtt(track):
    if track.k <= N_TRK_CONF or PREDICTION_ACTIVATE == 0:
        x_centroid = track.x_centroid
        y_centroid = track.y_centroid
        for i in range(len(x_centroid)):
            if x_centroid[i] == NOT_VALID_VALUE or y_centroid[i] == NOT_VALID_VALUE:
                del x_centroid[i]
                del y_centroid[i]
        if len(x_centroid) == 1:
            track.x_centroid_predicted = x_centroid[-1]
            track.y_centroid_predicted = y_centroid[-1]
        else:
            last_point_coeff = 1
            track.x_centroid_predicted = last_point_coeff*x_centroid[-1] + (1 - last_point_coeff) * (2 * x_centroid[-1] - x_centroid[-2])
            track.y_centroid_predicted = last_point_coeff*y_centroid[-1] + (1 - last_point_coeff) * (2 * y_centroid[-1] - y_centroid[-2])
        return track

    if track.q > 0 and PREDICTION_ACTIVATE == 1:
        v_x = track.state_vector(1)
        v_y = track.state_vector(4)
        dt = DIFFERENTIAL_TIME
        track.x_centroid_predicted = track.x_centroid_predicted + v_x*dt
        track.y_centroid_predicted = track.y_centroid_predicted + v_y * dt
        return track

    # TODO: complete unused smoothing logic according to matlab
