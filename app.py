import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load trained model
model = tf.keras.models.load_model("mnist_cnn_model.keras")

st.set_page_config(page_title="Handwritten Digit Recognition")

st.title("✍️ Handwritten Digit Recognition")

st.write("Upload a single handwritten digit (0-9) and the CNN model will predict it.")

uploaded_file = st.file_uploader(
    "Choose a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Display Original Image
    original_image = Image.open(uploaded_file)

    st.image(
        original_image,
        caption="Original Image",
        width=200
    )

    # -------------------------------
    # Image Preprocessing
    # -------------------------------

    image = original_image.convert("L")      # Convert to grayscale
    image = ImageOps.invert(image)           # Invert colors
    image = image.resize((28, 28))           # Resize

    img_array = np.array(image)

    img_array = img_array.astype("float32") / 255.0

    img_array = img_array.reshape(1, 28, 28, 1)

    # Show Processed Image
    st.image(
        image,
        caption="Processed Image",
        width=200
    )

    # -------------------------------
    # Prediction
    # -------------------------------

    prediction = model.predict(img_array, verbose=0)

    predicted_digit = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    st.success(f"Predicted Digit: {predicted_digit}")

    st.info(f"Confidence: {confidence:.2f}%")