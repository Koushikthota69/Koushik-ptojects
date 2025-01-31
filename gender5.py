import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Load the saved model
model = tf.keras.models.load_model(r'C:\Users\thota\OneDrive\Desktop\project_git\gender3.h5')

# Function to preprocess and resize the image
def preprocess_image(img):
    # Resize the image to 64x64 as per the model's expected input size
    img = cv2.resize(img, (64, 64))  # Resize to 64x64
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize if required
    return img_array

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from webcam.")
        break

    # Preprocess the frame
    processed_img = preprocess_image(frame)

    # Make predictions
    prediction = model.predict(processed_img)

    # Determine gender based on prediction
    if prediction < 0.5:
        gender = "Female"
    else:
        gender = "Male"

    # Display the gender on the frame
    cv2.putText(frame, f"Gender: {gender}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Gender Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows() ##match this code with the below # Display the live video feed and the classification results
def app():
    webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)

    if webrtc_ctx.video_processor:
        class_label = st.empty()
        elapsed_time = st.empty()

        while True:
            class_label.write(f"Predicted class: {webrtc_ctx.video_processor.class_label}")
            elapsed_time.write(f"Time taken: {webrtc_ctx.video_processor.elapsed_time:.4f} seconds")
            time.sleep(0.1)

app()