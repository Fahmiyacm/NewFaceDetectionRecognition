import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image
import mediapipe as mp
import face_recognition
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

# -----------------------------
# Load encodings
# -----------------------------
data = pickle.load(open("encodings.pkl", "rb"))

# -----------------------------
# MediaPipe
# -----------------------------
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(min_detection_confidence=0.5)

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Face Recognition System", layout="centered")

st.title("🔍 Face Recognition System")

menu = st.sidebar.radio("Select Option", ["Predict", "Evaluation Dashboard"])

# =========================================================
# 🔹 1. PREDICTION SECTION
# =========================================================
if menu == "Predict":

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:

        image = Image.open(uploaded_file)
        image = np.array(image)

        rgb = image.copy()

        results = detector.process(rgb)

        if results.detections:

            for detection in results.detections:

                bbox = detection.location_data.relative_bounding_box

                h, w, _ = image.shape

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                w_box = int(bbox.width * w)
                h_box = int(bbox.height * h)

                x = max(0, x)
                y = max(0, y)

                face = rgb[y:y+h_box, x:x+w_box]

                name = "Unknown"

                if face.size != 0 and face.shape[0] > 20 and face.shape[1] > 20:

                    face = cv2.resize(face, (160, 160))

                    encodings = face_recognition.face_encodings(face)

                    if len(encodings) > 0:

                        encoding = encodings[0]

                        matches = face_recognition.compare_faces(
                            data["encodings"],
                            encoding
                        )

                        face_distances = face_recognition.face_distance(
                            data["encodings"],
                            encoding
                        )

                        best_match = np.argmin(face_distances)

                        if matches[best_match]:
                            name = data["names"][best_match]

                cv2.rectangle(image, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
                cv2.putText(image, name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        st.image(image, caption="Result", use_container_width=True)

# =========================================================
# 🔹 2. EVALUATION DASHBOARD (FIXED + SAFE)
# =========================================================
elif menu == "Evaluation Dashboard":

    st.subheader("📊 Model Evaluation")

    test_path = "dataset/test"

    if not os.path.exists(test_path):
        st.error("Dataset folder not found!")
    else:

        y_true = []
        y_pred = []

        for person in os.listdir(test_path):
            person_path = os.path.join(test_path, person)

            if not os.path.isdir(person_path):
                continue

            for img_name in os.listdir(person_path):

                img_path = os.path.join(person_path, img_name)

                image = cv2.imread(img_path)
                if image is None:
                    continue

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                results = detector.process(rgb)

                pred = "Unknown"

                if results.detections:

                    for detection in results.detections:

                        bbox = detection.location_data.relative_bounding_box

                        h, w, _ = image.shape

                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        w_box = int(bbox.width * w)
                        h_box = int(bbox.height * h)

                        face = rgb[y:y+h_box, x:x+w_box]

                        if face.size != 0 and face.shape[0] > 20:

                            face = cv2.resize(face, (160, 160))

                            encodings = face_recognition.face_encodings(face)

                            if len(encodings) > 0:

                                encoding = encodings[0]

                                face_distances = face_recognition.face_distance(
                                    data["encodings"],
                                    encoding
                                )

                                best_match = np.argmin(face_distances)
                                pred = data["names"][best_match]

                y_true.append(person)
                y_pred.append(pred)

        # -----------------------------
        # Metrics (SAFE)
        # -----------------------------
        if len(y_true) == 0:
            st.warning("No test data found")
        else:
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

            st.write("### Accuracy:", round(acc, 2))
            st.write("### Precision:", round(prec, 2))
            st.write("### Recall:", round(rec, 2))

            fig, ax = plt.subplots()
            ax.bar(["Accuracy", "Precision", "Recall"], [acc, prec, rec])
            ax.set_ylim(0, 1)
            st.pyplot(fig)