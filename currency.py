from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import streamlit as st
import os

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the Keras model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Function to preprocess image for Keras model
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array

# Streamlit UI
st.title('Fake Currency Detection')

img_file = st.file_uploader('Upload currency', type=['png', 'jpg', 'jpeg'])

if img_file is not None:
    file_details = {
        'name': img_file.name,
        'size': img_file.size,
        'type': img_file.type
    }
    st.write(file_details)
    st.image(img_file, width=255)

    # Save the uploaded image
    with open(os.path.join('uploads', 'src.jpg'), 'wb') as f:
        f.write(img_file.getbuffer())

    st.success('Image Saved')

    # Preprocess the image for Keras model
    processed_image = preprocess_image('uploads/src.jpg')

    # Create the array of the right shape to feed into the Keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = processed_image

    # Predict using Keras model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    predicted_class = class_names[index].strip()

    st.info('Keras Model Prediction:')
    st.write("Class:", predicted_class)
    st.write("Confidence Score:", prediction[0][index])
