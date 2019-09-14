import cv2
import numpy as np
import time
import os

# This code is a utility for video cropping.
# It allows cropping a movie by setting first frame, initial ROI and last frame.

# Movie to crop and first frame to be displayed
input_video = 'D:\MSc_Project\Drone_Movies_Raw\GOPR0014.mp4'
cap = cv2.VideoCapture(input_video)
start_frame = 10259

# output movie setup
output_video = 'cropped_drone_videos/' + os.path.splitext(input_video)[0] + time.strftime("_%d%m%Y_%H%M%S") + '.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(output_video, fourcc, 20.0, (1920, 1080))

# Text parameters
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 30)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

# get first frame
cap.set(1, start_frame)
retval, first_frame = cap.read()
frame_num = start_frame + 1

# initialize the list of reference points and boolean indicating
# whether cropping ROI is being performed or not
refPt = []
cropping = False


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping, clone, image

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False

        # clone image without rectangle, to allow reset of the ROI
        clone = image.copy()
        # draw a rectangle around the ROI
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)


# load the image, clone it, and setup the mouse callback function
image = first_frame
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

while True:
    # display the image and wait for a keypress
    image_with_text = image.copy()
    cv2.putText(image_with_text, 'Press n for next frame, c to set ROI or r to reset. Draw ROI with the mouse.', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    cv2.imshow("image", image_with_text)
    key = cv2.waitKey(1) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()

    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

    # if the 'n' key is pressed, continue to the next frame
    elif key == ord("n"):
        retval, image = cap.read()
        frame_num = frame_num + 1
        continue

# if there are two reference points, then crop the region of interest
# from the image and display it
if len(refPt) == 2:
    # roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    roi_center = (int(np.floor((refPt[0][0]+refPt[1][0])/2)), int(np.floor((refPt[0][1]+refPt[1][1])/2)))
    output_first_frame_num = frame_num
    print('ROI center coordinates:', roi_center)
    print('first frame number:', output_first_frame_num)
    # draw circle in the middle of ROI and show coordinates and frame number
    cv2.putText(image, 'ROI center coordinates: ' + str(roi_center) + '. first frame number:' + str(frame_num) + '. Press n to continue.' , bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    cv2.circle(image, roi_center, 3, (0, 0, 255), -1)
    cv2.imshow("image", image)
    cv2.waitKey(0)

image = clone
out.write(image)  # start writing frames to output video

while True:
    # display the image and wait for a keypress
    image_with_text = image.copy()
    cv2.putText(image_with_text, 'Press n for next frame or c to set last frame',
                bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    cv2.imshow("image", image_with_text)
    key = cv2.waitKey(1) & 0xFF

    # if the 'c' key is pressed, break from the loop to set as last frame
    if key == ord("c"):
        break

    # if the 'n' key is pressed, continue to the next frame
    elif key == ord("n"):
        retval, image = cap.read()
        out.write(image)
        frame_num = frame_num + 1
        continue

# show last frame number
cv2.putText(image, 'last frame number:' + str(frame_num) + '. Press any key to exit or s to save movie and exit.', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
cv2.imshow("image", image)
output_last_frame_num = frame_num
print('last frame number:', output_last_frame_num)
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        out.release()
        text_file = open("cropped_drone_videos/cropped_movies_data.txt", "a")
        total_out_length = output_last_frame_num - output_first_frame_num
        out_str = 'Filename: ' + output_video + ' Input video: ' + input_video + ' ROI center 1st frame: ' + str(roi_center) + ' First frame number: ' + str(output_first_frame_num) + ' Last frame number: ' + str(output_last_frame_num) + ' Total length: ' + str(total_out_length) + '\n'
        text_file.write(out_str)
        text_file.close()
        break
    elif key != 0xFF:
        out.release()
        try:
            os.remove(output_video)
        finally:
            break

cap.release()
# close all open windows
cv2.destroyAllWindows()
