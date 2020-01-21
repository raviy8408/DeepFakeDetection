import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# create train and test video split
pd_dt = pd.read_json(r'C:\Ravi\files\deepfake-detection-challenge\train_sample_videos\metadata.json').T.reset_index()
pd_dt.rename(columns={"index": "video"}, inplace=True)
print("Train data info:\n")
print(pd_dt.info())
print(pd_dt.head())
X_train, X_test, y_train, y_test = train_test_split(pd_dt["video"].values, pd_dt["label"].values,
                                                    test_size=0.2, random_state=42, stratify=pd_dt["label"].values)

# extract faces from the video frames


