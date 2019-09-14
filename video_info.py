def get_video_info(video_number):

    switcher = {  # [start_frame, end_frame, tracking_point, video_filename]
        # GOPR0014.MP4
        1: [10258, 10604, [130, 1739], "D:/MSc_Project/Drone_Movies_Raw/GOPR0014.MP4"],
        2: [10499, 10799, [419, 1265], "D:/MSc_Project/Drone_Movies_Raw/GOPR0014.MP4"],
        3: [15999, 16349, [617, 844], "D:/MSc_Project/Drone_Movies_Raw/GOPR0014.MP4"],
        4: [17099, 17449, [620, 1036], "D:/MSc_Project/Drone_Movies_Raw/GOPR0014.MP4"],
        5: [21419, 21759, [356, 930], "D:/MSc_Project/Drone_Movies_Raw/GOPR0014.MP4"],
        6: [22999, 23335, [632, 912], "D:/MSc_Project/Drone_Movies_Raw/GOPR0014.MP4"],
        7: [17499, 17699, [632, 346], "D:/MSc_Project/Drone_Movies_Raw/GOPR0014.MP4"],
        8: [17569, 17699, [489, 899], "D:/MSc_Project/Drone_Movies_Raw/GOPR0014.MP4"],
        9: [23069, 23312, [644, 894], "D:/MSc_Project/Drone_Movies_Raw/GOPR0014.MP4"],
        10: [28799, 29999, [749, 1010], "D:/MSc_Project/Drone_Movies_Raw/GOPR0014.MP4"],
        11: [17549, 17949, [383, 726], "D:/MSc_Project/Drone_Movies_Raw/GOPR0014.MP4"],
        12: [16799, 17949, [528, 962], "D:/MSc_Project/Drone_Movies_Raw/GOPR0014.MP4"],
        13: [17749, 17949, [639, 954], "D:/MSc_Project/Drone_Movies_Raw/GOPR0014.MP4"],
        14: [38279, 39479, [388, 1025], "D:/MSc_Project/Drone_Movies_Raw/GOPR0014.MP4"],
        # GOPR0010.MP4 - movies 15 and 16 intentionally missing for consistency with matlab code
        17: [22799, 23279, [263, 812], "D:/MSc_Project/Drone_Movies_Raw/GOPR0010.MP4"],
        18: [25379, 25619, [251, 906], "D:/MSc_Project/Drone_Movies_Raw/GOPR0010.MP4"],
        19: [26759, 27419, [318, 1060], "D:/MSc_Project/Drone_Movies_Raw/GOPR0010.MP4"],
        20: [28019, 28379, [191, 809], "D:/MSc_Project/Drone_Movies_Raw/GOPR0010.MP4"],
        21: [29219, 29399, [264, 846], "D:/MSc_Project/Drone_Movies_Raw/GOPR0010.MP4"],
        22: [29819, 30419, [440, 767], "D:/MSc_Project/Drone_Movies_Raw/GOPR0010.MP4"],
        23: [32459, 33119, [250, 946], "D:/MSc_Project/Drone_Movies_Raw/GOPR0010.MP4"],
        24: [33659, 34079, [183, 794], "D:/MSc_Project/Drone_Movies_Raw/GOPR0010.MP4"],
        25: [35699, 36119, [203, 873], "D:/MSc_Project/Drone_Movies_Raw/GOPR0010.MP4"],
        26: [37979, 39719, [673, 1065], "D:/MSc_Project/Drone_Movies_Raw/GOPR0010.MP4"],
        27: [15999, 16349, [349, 1199], "D:/MSc_Project/Drone_Movies_Raw/GOPR0010.MP4"]
    }
    return switcher.get(video_number, [-1, -1, [-1, -1], "Invalid video info"])
