import os
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load wav2vec2
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

dataset_path = "dataset"

X = []
y = []

def extract_embedding(file_path):
    y_audio, sr = librosa.load(file_path, sr=16000)
    inputs = processor(y_audio, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding


for label, folder in enumerate(["human", "ai"]):
    folder_path = os.path.join(dataset_path, folder)

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        print("Processing:", file_path)

        embedding = extract_embedding(file_path)

        X.append(embedding)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Training classifier...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, "voice_classifier.pkl")

print("Model saved as voice_classifier.pkl")
