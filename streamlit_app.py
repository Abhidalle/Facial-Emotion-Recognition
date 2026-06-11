
# Import the necessary libraries for this streamlit app

import streamlit as st
import cv2
import numpy as np 
from keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import keras
from keras.layers import Dense

original_dense_init = Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)  # Deletes the Colab keyword
    original_dense_init(self, *args, **kwargs)
Dense.__init__ = patched_dense_init

# Add the basic stufs first
st.set_page_config(page_title="Emotion Detector AI", layout="centered")

# BEfore we do anything lets make an sidebar that has teh detailes about the code and what teh app does in general 
with st.sidebar:
    st.header("About Emotion Detector AI")
    st.markdown("Emotion Detector AI is a real-time facial expression tracking AI that is  built with deep learning and CNN(Convolution Neural Networks)")
    st.markdown("---")
    
    st.subheader("The Tools Required")
    st.markdown("""
    Base Model: MobileNetV2
    Custom Layers: Global Average Pooling + Dense (256) + Dropout (0.5)
    Performance: 60%+ Validation Accuracy
    """)
    
    st.markdown("---")
    
    st.subheader("The Making Process")
    st.markdown("""
    I wanted to quit only building my regular AI systems that were either text based or data based. It is my first project i built using the Deep Learning. This project started as a raw 39% accurate model. After completely re training the neural network, balancing the dataset mathematically by using the balanced function, and fighting through some massive library version conflicts, My first CNN Project was born. 
    """)



    st.markdown("---")

    # A tiny version of your privacy guarantee for the sidebar for our users
    st.caption("Privacy First: 100% local processing. No data is ever saved to the cloud.")



#Then with the basic UI components like titles and descriptions
st.title("Emotion Detector AI")

st.markdown("Real-time facial expression tracking powered by Deep Learning.")

#. Add the info thing that would tell the users that we do not store data in cloud or smt shady haha.
st.info(" Privacy Guarantee: All video processing happens purely in your device's active memory. **Zero data is stored.** No photos, videos, or personal data are ever recorded, saved, or sent to any database.")

#Use the cache_resource 
@st.cache_resource 
def load_my_brain():
    return load_model("best_model_v2.keras")

emotion_model = load_my_brain()
emotion_labelings = ["Anger", "Disgust", "Fear", "Happy","Neutral", "Sad", "Surprise"] 
#This is only used to find teh faces and it was that same pre trained model i was taking about from Open CV
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
#Add Simple checkbox to open camera
st.markdown("### 🎥 Camera Controls")
run_camera = st.checkbox("Turn Camera ON")


#Add empty box that will hold teh video frames on teh web
video_placeholder = st.image([])
 
if run_camera:
    cap = cv2.VideoCapture(0)
    #Add tehse variables so that we can make it lag free(Only taking into account what we see in the 5 frames as 1)
    frame_count = 0
    skip_frames = 5 
    final_emotion = "Scanning..."
    while run_camera: 
        ret, frame = cap.read()
        if not ret:
            st.write("Camera does not seem to be working")
            break 
        #Just add this for Zero to slighty mini nal lag
        frame_count += 1
 
        #Make the images of gray color (SINCE WE TRAINED ON B&W IMAGES) so the model can find it easily 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Find the faces by ignoring the backgrounds
        last_faces = []
        detected = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80,80))
        if len(detected) > 0:
            last_faces = detected
        faces = last_faces

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y), (x+w, y+h), (0,255,0), 2)
            if frame_count % skip_frames == 0:
                 # Crop the face and then finally reize them as well
                face_crop = frame[y:y+h, x:x+w]
                
                face_crop = cv2.resize(face_crop, (224, 224))
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

                # Normalize using MobileNetV2's exact method (scales to -1 to 1, not 0 to 1)
        
                face_crop = face_crop.astype('float32')
                face_crop = np.expand_dims(face_crop, axis=0)
                face_crop = preprocess_input(face_crop)


                # Get the prediction from the AI
                prediction = emotion_model.predict(face_crop,  verbose=0)
                max_index = int(np.argmax(prediction))
                final_emotion = emotion_labelings[max_index]

            #Put the text right above teh green box as well trh emotioan indicator
            cv2.putText(frame, final_emotion, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        # WE shall flip the colors so we dont look blue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb)

    #FINALLY Clean up the camera after uncheking the box
    cap.release()