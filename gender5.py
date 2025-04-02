import tensorflow as tf
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import numpy as np
import cv2

# Load the saved model with error handling
try:
    model = tf.keras.models.load_model(r'C:\Users\thota\OneDrive\Desktop\project_git\gender3.h5')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Function to preprocess the image
def preprocess_image(img):
    try:
        img = cv2.resize(img, (64, 64))  # Resize to model's input size
        img_array = np.asarray(img, dtype=np.float32) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Define Video Processor Class
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        processed_img = preprocess_image(img)
        
        if processed_img is not None:
            prediction = model.predict(processed_img)
            gender = "Male" if prediction[0][0] >= 0.5 else "Female"
            
            # Display Gender Label
            cv2.putText(img, f"Gender: {gender}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(img, "Error in Processing", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("Real-time Gender Detection")
st.write("This app uses a deep learning model to detect gender in real-time.")
st.info("Ensure your webcam is enabled and positioned correctly.")

webrtc_streamer(key="example", 
                video_processor_factory=VideoProcessor,
                mode=WebRtcMode.SENDRECV, 
                media_stream_constraints={"video": True, "audio": False})
