import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score, \
        recall_score, precision_score, f1_score

base_dir = "C:/Ravi/files/deepfake-detection-challenge/"

# data load
train_data = pd.read_csv(base_dir + 'train_yolo/' + "train_faces_tag_df.csv")
test_data = pd.read_csv(base_dir + 'test_yolo/' + "test_faces_tag_df.csv")

# creating an empty list
train_image = []
test_image = []

# for loop to read and store frames
for i in tqdm(range(train_data.shape[0])):
    # loading the image and keeping the target size as (224,224,3)
    img = image.load_img(base_dir + 'train_yolo/' + train_data['file_name'][i], target_size=(224, 224, 3))
    # converting it to array
    img = image.img_to_array(img)
    # normalizing the pixel value
    img = img / 255
    # appending the image to the train_image list
    train_image.append(img)

for i in tqdm(range(test_data.shape[0])):
    # loading the image and keeping the target size as (224,224,3)
    img = image.load_img(base_dir + 'test_yolo/' + test_data['file_name'][i], target_size=(224, 224, 3))
    # converting it to array
    img = image.img_to_array(img)
    # normalizing the pixel value
    img = img / 255
    # appending the image to the test_image list
    test_image.append(img)

# converting the list to numpy array
X_train = np.array(train_image)
X_test = np.array(test_image)

# shape of the array
print("Shape of train data:")
print(X_train.shape)

print("Shape of test data:")
print(X_test.shape)

# creating dummies of target variable for train and validation set
y_train = pd.get_dummies(train_data["label"])
y_test = pd.get_dummies(test_data["label"])
print("Shape of y_train")
print(y_train.shape)
print("Shape of y_test:")
print(y_test.shape)
# creating the base model of pre-trained InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False)
# extracting features for training frames
# X_train = base_model.predict(X_train)
# np.save(base_dir + 'X_train_encoded.npy', X_train)
X_train = np.load(base_dir + 'X_train_encoded.npy')
print("Shape of train data after prediction with pre-trained model")
print(X_train.shape)
# X_test = base_model.predict(X_test)
# np.save(base_dir + 'X_test_encoded.npy', X_test)
X_test = np.load(base_dir + 'X_test_encoded.npy')
print("Shape of test data after prediction with pre-trained model")
print(X_test.shape)

# flattening the X_train
X_train = X_train.reshape(X_train.shape[0], 5*5*2048)
X_test = X_test.reshape(X_test.shape[0], 5*5*2048)
# normalizing the pixel values
max_elem = X_train.max()
np.savetxt(base_dir + "max_elem.txt", [max_elem])
X_train = X_train / max_elem
X_test = X_test/ max_elem
# shape of images
print("Shape of train data after flattening:")
print(X_train.shape)
print("Shape of test data after flattening:")
print(X_test.shape)

# defining the model architecture
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# defining a function to save the weights of best model
mcp_save = ModelCheckpoint(base_dir + 'weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')

# compiling the model
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# training the model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=[mcp_save], batch_size=128)
model.save(base_dir + "model.h5")

y_test_pred_prob = [elem[0] for elem in model.predict(X_test)]
y_test_pred_class = [1 if elem > 0.5 else 0 for elem in y_test_pred_prob]

print("\n#######--Model performance at face level--#########\n")
print("Accuracy:\n")
print(accuracy_score(y_test["FAKE"].values, y_test_pred_class))
print("\nConfusion Matrix:\n")
print(pd.crosstab(pd.Series(y_test["FAKE"].values, name='Actual'), pd.Series(y_test_pred_class, name='Predicted')))
print("\nClassification Report:\n")
print(classification_report(y_test["FAKE"].values, y_test_pred_class))
print("\nCohen Kappa:\n")
print(cohen_kappa_score(y_test["FAKE"].values, y_test_pred_class))

print("\n#######--Model performance at video level--#########\n")
test_data_pred = test_data.copy()
test_data_pred["pred_label"] = y_test_pred_class

test_video_pred = test_data_pred.groupby("video").apply(lambda x: pd.Series({'label': x["label"].unique()[0],
                                                                             'pred_label': x["pred_label"]
                                                                            .value_counts().index[0]}))\
    .reset_index()

test_video_pred["act_label"] = [1 if elem == 'FAKE' else 0 for elem in test_video_pred["label"].values]
print("Accuracy:\n")
print(accuracy_score(test_video_pred["act_label"].values, test_video_pred["pred_label"].values))
print("\nConfusion Matrix:\n")
print(pd.crosstab(pd.Series(test_video_pred["act_label"].values, name='Actual'), pd.Series(test_video_pred["pred_label"]
                                                                                           .values, name='Predicted')))
print("\nClassification Report:\n")
print(classification_report(test_video_pred["act_label"].values, test_video_pred["pred_label"].values))
print("\nCohen Kappa:\n")
print(cohen_kappa_score(test_video_pred["act_label"].values, test_video_pred["pred_label"].values))










