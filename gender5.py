import tensorflow as tf
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image

# Load the saved model
model = tf.keras.models.load_model(r'C:\Users\thota\OneDrive\Desktop\project_git\gender3.h5')

# Function to preprocess and resize the image
def preprocess_image(img):
    img = cv2.resize(img, (64, 64))  # Resize to 64x64
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize if required
    return img_array

# Custom VideoProcessor class
class VideoProcessor:
    def __init__(self):
        self.class_label = "Unknown"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Predict gender
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)
        self.class_label = "Female" if prediction < 0.5 else "Male"

        # Overlay text
        cv2.putText(img, f"Gender: {self.class_label}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit app
def app():
    st.title("Gender Detection using Streamlit WebRTC")
    webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)

    if webrtc_ctx.video_processor:
        st.write(f"Predicted Gender: {webrtc_ctx.video_processor.class_label}")

app()
