import os
import json
import pickle
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import face_recognition

# ---------------- PATHS ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "svm_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
CURVE_PATH = os.path.join(BASE_DIR, "model", "curve.json")

# ---------------- LOAD FILES ----------------

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

with open(CURVE_PATH, "r") as f:
    curve = json.load(f)

# ---------------- PAGE ----------------

st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 End-to-End Face Detection & Recognition System")

# ---------------- TABS ----------------

tab1, tab2 = st.tabs(
    ["🔍 Face Recognition", "📊 Model Performance"]
)

# =====================================================
# TAB 1 : FACE RECOGNITION
# =====================================================

with tab1:

    st.header("Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        image_np = np.array(image)

        rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(
            rgb,
            face_locations
        )

        if len(face_locations) == 0:
            st.warning("No face detected.")
        else:

            results = []

            for (top, right, bottom, left), encoding in zip(
                    face_locations,
                    face_encodings):

                # Prediction
                prediction = model.predict([encoding])[0]

                confidence = "N/A"

                # Try probability if available
                if hasattr(model, "predict_proba"):
                    prob = np.max(model.predict_proba([encoding]))
                    confidence = f"{prob * 100:.2f}%"

                # Draw box
                cv2.rectangle(
                    image_np,
                    (left, top),
                    (right, bottom),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image_np,
                    prediction,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                results.append(
                    {
                        "Name": prediction,
                        "Confidence": confidence
                    }
                )

            st.image(
                image_np,
                caption="Recognition Result",
                width=500
            )

            st.subheader("Recognition Results")

            for r in results:
                st.write(
                    f"**Name:** {r['Name']} | "
                    f"**Confidence:** {r['Confidence']}"
                )

# =====================================================
# TAB 2 : MODEL PERFORMANCE
# =====================================================

with tab2:

    st.header("Model Evaluation")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{metrics['accuracy']:.4f}"
    )

    col2.metric(
        "Precision",
        f"{metrics['precision']:.4f}"
    )

    col3.metric(
        "Recall",
        f"{metrics['recall']:.4f}"
    )

    st.markdown("---")

    st.subheader("Learning Curve")

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        curve["train_sizes"],
        curve["train_acc"],
        marker="o",
        label="Train Accuracy"
    )

    ax.plot(
        curve["train_sizes"],
        curve["test_acc"],
        marker="o",
        label="Validation Accuracy"
    )

    ax.set_xlabel("Training Samples")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Learning Curve")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    st.markdown("---")

    st.write(
        """
        **Accuracy**: Overall percentage of correctly recognized faces.

        **Precision**: How many predicted identities were correct.

        **Recall**: How many actual identities were successfully recognized.
        """
    )