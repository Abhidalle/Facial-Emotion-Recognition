# Import the necessary libraries for this streamlit app
import streamlit as st
import cv2
import numpy as np 
from keras.models import load_model

#Start with the basic UI components like titles and descriptions
st.title("AI Emotion Analyzer")
st.write("Check the box below for turning your webcam on")

#Use the cache_resource 
@st.cache_resource
def load_my_brain():
    return load_model("best_model.h5")

emotion_model = load_my_brain() 
emotion_labelings = ["Anger", "Disgust", "Fear", "Happy","Neutral", "Sad", "Surprise"]

#This is only used to find teh faces and it was that same pre trained model i was taking about from Open CV
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

#Add Simple checkbox to open camera
run_camera = st.checkbox("Turn Camera ON")

#Add empty box that will hold teh video frames on teh web
video_placeholder = st.image([])
 
if run_camera:
    cap = cv2.VideoCapture(0)
    
    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.write("Camera does not seem to be working")
            break

        #Make the images of gray color (SINCE WE TRAINED ON B&W IMAGES) so the model can find it easily 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Find the faces by ignoring the backgrounds
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(120,120))

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y), (x+w, y+h), (0,255,0), 2)

            # Crop the face and then finally reize them as well
            face_crop = frame[y:y+h, x:x+w]
            face_crop = cv2.resize(face_crop, (224, 224))
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            #Normalize the numbers now
            face_crop = face_crop.astype('float32') / 255.0
            face_crop = np.expand_dims(face_crop, axis=0)

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
