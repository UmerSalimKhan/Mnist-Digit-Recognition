import streamlit as st
import tensorflow as tf  
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("mnist_model.keras")  

st.title("MNIST Digit Recognition")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img = image.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img)  # Convert to NumPy array

    # Check image type and channels
    if img_array.dtype != np.uint8:
        st.error("Image type should be uint8")
    if len(img_array.shape) != 2:
        st.error("Image should be grayscale (2D)") # If the image is already 2D, no need to add a channel dimension.
    else:
        img_array = img_array.reshape(1, 28, 28, 1) / 255.0  # Add channel dimension and normalize

        if st.button("Predict"):
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            st.write(f"Predicted Digit: {predicted_class}")

            # Display probabilities 
            st.write("Probabilities:")
            for i, prob in enumerate(predictions[0]):
                st.write(f"{i}: {prob:.4f}")