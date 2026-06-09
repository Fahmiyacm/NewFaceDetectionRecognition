import os
import numpy as np
import face_recognition


def get_embeddings(split_path):
    X = []
    y = []

    for person in os.listdir(split_path):

        person_path = os.path.join(split_path, person)

        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):

            img_path = os.path.join(person_path, img_name)

            if not os.path.isfile(img_path):
                continue

            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            try:
                image = face_recognition.load_image_file(img_path)

                faces = face_recognition.face_locations(image)
                encodings = face_recognition.face_encodings(image, faces)

                if len(encodings) > 0:
                    X.append(encodings[0])
                    y.append(person)

            except Exception as e:
                print("Skipping:", img_path)

    return np.array(X), np.array(y)