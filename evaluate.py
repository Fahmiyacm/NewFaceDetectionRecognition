import os
import cv2
import pickle
import face_recognition
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

data = pickle.load(open("encodings.pkl", "rb"))

test_path = "dataset/test"

y_true = []
y_pred = []

for person in os.listdir(test_path):
    person_path = os.path.join(test_path, person)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        image = cv2.imread(img_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        encodings = face_recognition.face_encodings(rgb)

        if len(encodings) == 0:
            continue

        encoding = encodings[0]

        matches = face_recognition.compare_faces(
            data["encodings"], encoding
        )

        face_distances = face_recognition.face_distance(
            data["encodings"], encoding
        )

        best_match = np.argmin(face_distances)

        if matches[best_match]:
            pred = data["names"][best_match]
        else:
            pred = "Unknown"

        y_true.append(person)
        y_pred.append(pred)

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, average='macro'))
print("Recall:", recall_score(y_true, y_pred, average='macro'))