import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/plant_village_model.h5"


model = tf.keras.models.load_model(model_path)

class_indices_path = os.path.join(working_dir, "class_indices-1.json")
class_indices = json.load(open(class_indices_path))


def load_preprocess_image(image_path, target_size = (256,256)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image_arr = np.array(image) / 255.0
    image_arr = np.expand_dims(image, axis = 0)
    image_arr = image_arr.astype('float32') / 255
    return image_arr

def predict_image_class(model,image_path,class_indices):
    preprocessed_img = load_preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(prediction, axis = -1)[0]
    predicted_class = class_indices[str(predicted_class_index)]
    return predicted_class


#STREAMLIT APPP

st.title('PLANT DISEASE CLASSIFIER - 2025')
upload_image = st.file_uploader("Upload an image...",type=['jpg','png','jpeg'])

if upload_image is not None:
    image = Image.open(upload_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150,150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            prediction = predict_image_class(model, upload_image, class_indices)
            st.success(f"Prediction: {str(prediction)}")



