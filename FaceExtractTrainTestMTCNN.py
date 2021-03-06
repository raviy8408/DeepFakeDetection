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

# create train and test video split
pd_dt = pd.read_json(base_dir + "train_sample_videos/metadata.json").T.reset_index()
pd_dt.rename(columns={"index": "video"}, inplace=True)
print("Train data info:\n")
print(pd_dt.info())
print(pd_dt.head())
X_train, X_test, y_train, y_test = train_test_split(pd_dt["video"].values, pd_dt["label"].values,
                                                    test_size=0.2, random_state=42, stratify=pd_dt["label"].values)

# extract faces from the video frames
# Train Data Creation
for i in tqdm(range(X_train.shape[0])):
    count = 0
    videoFile = X_train[i]
    dir_path = base_dir + 'train_sample_videos/'
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
                    count += 1
                    cv2.imwrite(base_dir + 'train_mtcnn/' + videoFile + "_" + "frame%d.jpg" % count, roi_color)
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
                        count += 1
                        cv2.imwrite(base_dir + 'train_mtcnn/' + videoFile + "_" + "frame%d.jpg" % count, roi_color)
                else:
                    print(videoFile)

    cap.release()

# Test Data Creation
for i in tqdm(range(X_test.shape[0])):
    count = 0
    videoFile = X_test[i]
    dir_path = base_dir + 'train_sample_videos/'
    cap = cv2.VideoCapture(dir_path + videoFile)  # capturing the video from the given path
    frameRate = cap.get(5)  # frame rate
    x = 1
    while cap.isOpened():
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if not ret:
            break
        if frameId % math.floor(frameRate) == 0:
            # storing the frames in a new folder named test_mtcnn
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detect_faces(frame)
            if len(faces):
                for (x, y, w, h) in [faces[0]['box']]:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    # Save faces in a folder
                    roi_color = frame[y:y + h, x:x + w]
                    count += 1
                    cv2.imwrite(base_dir + 'test_mtcnn/' + videoFile + "_" + "frame%d.jpg" % count, roi_color)
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
                        count += 1
                        cv2.imwrite(base_dir + 'test_mtcnn/' + videoFile + "_" + "frame%d.jpg" % count, roi_color)
                else:
                    print(videoFile)

    cap.release()

# create face image tag for frame, to be used in baseline model
# train data tags
train_faces = [f for f in listdir(base_dir + 'train_mtcnn/') if isfile(join(base_dir + 'train_mtcnn/', f))]
train_faces_df = pd.DataFrame({'file_name': train_faces})
train_faces_df["video"] = train_faces_df["file_name"].apply(lambda x: x.split("_")[0])
train_faces_tag_df = pd.merge(left=train_faces_df, left_on=["video"], right=pd_dt[["video", "label"]],
                              right_on=["video"], how="left")
train_faces_tag_df.to_csv(base_dir + 'train_mtcnn/' + "train_faces_tag_df.csv", index=False, header=True)
print(train_faces_tag_df.head())

# test data tags
test_faces = [f for f in listdir(base_dir + 'test_mtcnn/') if isfile(join(base_dir + 'test_mtcnn/', f))]
test_faces_df = pd.DataFrame({'file_name': test_faces})
test_faces_df["video"] = test_faces_df["file_name"].apply(lambda x: x.split("_")[0])
test_faces_tag_df = pd.merge(left=test_faces_df, left_on=["video"], right=pd_dt[["video", "label"]],
                             right_on=["video"], how="left")
test_faces_tag_df.to_csv(base_dir + 'test_mtcnn/' + "test_faces_tag_df.csv", index=False, header=True)
print(test_faces_tag_df.head())




