import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Custom loss function for loading model
from tensorflow.keras.losses import MeanAbsoluteError

# Load trained model with custom objects
custom_objects = {
    "mae": MeanAbsoluteError()
}
model = load_model("age_gender_model.h5", custom_objects=custom_objects)

# Define gender labels
gender_dict = {0: "👨 Male", 1: "👩 Female"}

# Streamlit UI Setup
st.set_page_config(page_title="Age & Gender Prediction System", page_icon="🔍", layout="centered")

st.title("🔍 Age & Gender Prediction")
st.write("📷 **Upload a face image to predict age and gender.**")

st.sidebar.info(
    "**This application is a Semester Project for the course Artificial Neural Networks (ANN).** "
    "It is designed for educational purposes only. Predictions may not be 100% accurate."
)


st.sidebar.markdown("**Developed by Ali Zia 👨‍💻**")  
st.sidebar.markdown("---")

# Sidebar for Image Upload
st.sidebar.header("📂 Upload Image")

# Preserve uploaded image using session state
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

# If new file is uploaded, update session state
if uploaded_file is not None:
    st.session_state.uploaded_image = uploaded_file

# Display uploaded image (remains even after prediction)
if st.session_state.uploaded_image is not None:
    st.sidebar.image(st.session_state.uploaded_image, caption="🖼️ Uploaded Image", width=150)

# Prediction logic
if st.session_state.uploaded_image is not None:
    with st.spinner("⏳ Processing..."):
        # Load and preprocess the image
        image = Image.open(st.session_state.uploaded_image).convert("L")  # Convert to grayscale
        image = image.resize((128, 128))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)  # Reshape for model input

        # Make Predictions
        gender_pred, age_pred = model.predict(image)
        gender_label = gender_dict[int(np.round(gender_pred[0][0]))]
        predicted_age = round(age_pred[0][0])

    # Display Predictions
    st.subheader("🔮 Predictions")
    st.success(f"🧑 Predicted Gender: **{gender_label}**")
    st.info(f"🎂 Predicted Age: **{predicted_age}**")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Developed by Ali Zia**👨‍💻 ")
st.sidebar.markdown("---")
