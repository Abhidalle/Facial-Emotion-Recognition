#IMPORTS FIRST

import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
from tensorflow.keras.layers import Dense

original_dense_init = Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_dense_init(self, *args, **kwargs)
Dense.__init__ = patched_dense_init


st.set_page_config(page_title="Emotion Detector AI", layout="centered")

# SIDEBAR
with st.sidebar:
    st.header("About Emotion Detector AI")
    st.markdown("Emotion Detector AI is a real-time facial expression tracking AI built with deep learning and CNN.")
    st.markdown("---")
    st.subheader("The Tools Required")
    st.markdown("* Base Model: MobileNetV2\n* Custom Layers: Global Avg Pooling + Dense (256) + Dropout (0.5)\n* Performance: 60%+ Validation Accuracy")
    st.markdown("---")
    st.subheader("The Making Process")
    st.markdown("This project started as a raw 39% accurate model. After retraining, balancing datasets, and fighting library conflicts, my CNN Project was born.")
    st.markdown("---")
    st.caption("Privacy First: 100% local processing. No data is ever saved.")


st.title("Emotion Detector AI")
st.markdown("Real-time facial expression tracking powered by Deep Learning.")
st.info("Privacy Guarantee: All video processing happens purely in your device's active memory. **Zero data is stored.**")


@st.cache_resource
def get_model():
    return load_model("best_model_v2.keras")

emotion_model = get_model()
emotion_labelings = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 4, minSize=(80,80))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Crop and Preprocess
            face_crop = cv2.resize(img[y:y+h, x:x+w], (224, 224))
            face_crop = preprocess_input(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB).astype('float32'))
            # Predict
            pred = emotion_model.predict(np.expand_dims(face_crop, axis=0), verbose=0)
            emotion = emotion_labelings[np.argmax(pred)]
            cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return img

st.markdown("### 🎥 Camera Controls")
picture = st.camera_input("Take a picture")

if picture:
    bytes_data = picture.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Process the image
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4, minSize=(80,80))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(cv2_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Crop and Preprocess
        face_crop = cv2.resize(cv2_img[y:y+h, x:x+w], (224, 224))
        face_crop = preprocess_input(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB).astype('float32'))
        
        # Predict
        pred = emotion_model.predict(np.expand_dims(face_crop, axis=0), verbose=0)
        emotion = emotion_labelings[np.argmax(pred)]
        
        cv2.putText(cv2_img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    st.image(cv2_img, channels="BGR")