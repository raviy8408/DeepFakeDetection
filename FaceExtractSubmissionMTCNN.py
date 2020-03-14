import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import math
from tqdm import tqdm
from mtcnn.mtcnn import MTCNN
from os import listdir
from os.path import isfile, join
from support_funcs import increase_brightness, adjust_brightness_and_contrast

base_dir = "C:/Ravi/files/deepfake-detection-challenge/"

# importing face detector
detector = MTCNN()

# file list of submission file
submission_faces = [f for f in listdir(base_dir + 'test_videos/') if isfile(join(base_dir + 'test_videos/', f))]

# extract faces from the video frames
# Train Data Creation
for i in tqdm(submission_faces):
    # if i == "eppyqpgewp.mp4":
    count = 0
    videoFile = i
    dir_path = base_dir + 'test_videos/'
    cap = cv2.VideoCapture(dir_path + videoFile)  # capturing the video from the given path
    frameRate = cap.get(5)  # frame rate
    x = 1
    while cap.isOpened():
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if not ret:
            break
        if frameId % math.floor(frameRate) == 0:
            # storing the frames in a new folder named train_mtcnn
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detect_faces(frame)
            if len(faces):
                for (x, y, w, h) in [faces[0]['box']]:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    # Save faces in a folder
                    roi_color = frame[y:y + h, x:x + w]
                    roi_shape = np.shape(roi_color)
                    if roi_shape[0]*roi_shape[1]*roi_shape[2] != 0:
                        count += 1
                        cv2.imwrite(base_dir + 'submission_mtcnn/' + videoFile + "_" + "frame%d.jpg" % count, roi_color)
            else:
                # frame_brightened = increase_brightness(frame, value=30)
                frame_brightened = adjust_brightness_and_contrast(frame, alpha=2.5, beta=80, gamma=1.5)
                faces_bright = detector.detect_faces(frame_brightened)
                if len(faces_bright):
                    print("Face brightening worked")
                    for (x, y, w, h) in [faces_bright[0]['box']]:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                        # Save faces in a folder
                        roi_color = frame[y:y + h, x:x + w]
                        roi_shape = np.shape(roi_color)
                        if roi_shape[0]*roi_shape[1]*roi_shape[2] != 0:
                            count += 1
                            cv2.imwrite(base_dir + 'submission_mtcnn/' + videoFile + "_" + "frame%d.jpg" % count, roi_color)
                else:
                    print(videoFile)

    cap.release()