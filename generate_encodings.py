import os
import cv2
import pickle
import face_recognition

train_path = "dataset/train"

known_encodings = []
known_names = []

for person in os.listdir(train_path):
    person_path = os.path.join(train_path, person)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        image = cv2.imread(img_path)
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        encodings = face_recognition.face_encodings(rgb)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person)

data = {
    "encodings": known_encodings,
    "names": known_names
}

with open("encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print("Encodings created successfully!")