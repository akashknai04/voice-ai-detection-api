import os
import librosa
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATASET_PATH = "dataset"

X = []
y = []

print("Loading dataset...")

for label_name, label_value in [("human", 0), ("ai", 1)]:
    folder = os.path.join(DATASET_PATH, label_name)

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)

        try:
            audio, sr = librosa.load(file_path, sr=16000, mono=True)

            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfcc_mean = np.mean(mfcc.T, axis=0)

            X.append(mfcc_mean)
            y.append(label_value)

        except Exception as e:
            print("Error loading:", file_path)

X = np.array(X)
y = np.array(y)

print("Training model...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

joblib.dump(model, "voice_classifier.pkl")

print("Model saved successfully!")
