import os
import pickle
import numpy as np
import face_recognition
import json

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import learning_curve

# ---------------- ROOT PATH FIX ----------------


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR, "dataset", "train")
TEST_PATH = os.path.join(BASE_DIR, "dataset", "test")

MODEL_PATH = os.path.join(BASE_DIR, "model", "svm_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
CURVE_PATH = os.path.join(BASE_DIR, "model", "curve.json")

os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)


def get_embeddings(path):
    X, y = [], []

    for person in os.listdir(path):
        person_path = os.path.join(path, person)

        if not os.path.isdir(person_path):
            continue

        for img in os.listdir(person_path):
            img_path = os.path.join(person_path, img)

            if not img.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            try:
                image = face_recognition.load_image_file(img_path)
                faces = face_recognition.face_locations(image)
                encodings = face_recognition.face_encodings(image, faces)

                if len(encodings) > 0:
                    X.append(encodings[0])
                    y.append(person)
            except:
                continue

    return np.array(X), np.array(y)


print("Loading data...")
X_train, y_train = get_embeddings(TRAIN_PATH)
X_test, y_test = get_embeddings(TEST_PATH)

print("Train:", len(X_train), "Test:", len(X_test))


# ---------------- TRAIN MODEL ----------------
model = SVC(
    kernel="linear",
    probability=True
)
model.fit(X_train, y_train)


# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")

print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)


# ---------------- SAVE MODEL ----------------
pickle.dump(model, open(MODEL_PATH, "wb"))


# ---------------- LEARNING CURVE ----------------
train_sizes, train_scores, test_scores = learning_curve(
    model,
    X_train,
    y_train,
    cv=3,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 5)
)

curve = {
    "train_sizes": train_sizes.tolist(),
    "train_acc": np.mean(train_scores, axis=1).tolist(),
    "test_acc": np.mean(test_scores, axis=1).tolist()
}

json.dump(curve, open(CURVE_PATH, "w"))


# ---------------- SAVE METRICS ----------------
metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall)
}

json.dump(metrics, open(METRICS_PATH, "w"))

print("Training completed successfully ✔")