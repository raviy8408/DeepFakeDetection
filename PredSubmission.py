import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.models import load_model

base_dir = "C:/Ravi/files/deepfake-detection-challenge/"

submission_faces = [f for f in listdir(base_dir + 'submission_yolo/') if isfile(join(base_dir + 'submission_yolo/', f))]

submission_image = []

# for loop to read and store frames
for i in tqdm(submission_faces):
    # loading the image and keeping the target size as (224,224,3)
    img = image.load_img(base_dir + 'submission_yolo/' + i, target_size=(224, 224, 3))
    # converting it to array
    img = image.img_to_array(img)
    # normalizing the pixel value
    img = img / 255
    # appending the image to the train_image list
    submission_image.append(img)

# converting the list to numpy array
X_submission = np.array(submission_image)

# shape of the array
print("Shape of submission data:")
print(X_submission.shape)

# creating the base model of pre-trained InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False)
# extracting features for submission frames
# X_submission = base_model.predict(X_submission)
# np.save(base_dir + 'X_submission_encoded.npy', X_submission)
X_submission = np.load(base_dir + 'X_submission_encoded.npy')
print("Shape of submission data after prediction with pre-trained model")
print(X_submission.shape)

# flattening the X_train
X_submission = X_submission.reshape(X_submission.shape[0], 5*5*2048)
# # normalizing the pixel values
max_elem = np.loadtxt(base_dir + "max_elem.txt").item(0)
X_submission = X_submission / max_elem

# shape of images
print("Shape of submission data after flattening:")
print(X_submission.shape)

# loading model
model = load_model(base_dir + "model.h5")

# prediction on submission video frames
y_submission_pred_prob = [elem[0] for elem in model.predict(X_submission)]
y_submission_pred_class = [1 if elem > 0.5 else 0 for elem in y_submission_pred_prob]

submission_image_df = pd.DataFrame()
submission_image_df["frames"] = submission_faces
submission_image_df["filename"] = submission_image_df["frames"].apply(lambda x: x.split("_")[0])
submission_image_df["label"] = y_submission_pred_class

submission_video_df = submission_image_df.groupby("filename").apply(lambda x: pd.Series({'label': x["label"]
                                                                                     .value_counts().index[0]}))\
    .reset_index()

print("\n Final submission file:\n")
print(submission_video_df.head())

submission_video_df.to_csv(base_dir + "submission.csv", index=False)







