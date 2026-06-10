import cv2 
import numpy as np
import json
import h5py
import matplotlib.pyplot as plt
from keras.models import load_model

new_model = load_model("best_model.h5")
frame = cv2.imread("trial.jpeg")
#plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#this is a variaition from the actual Open CV code but does it pretty similarly
faces = faceCascade.detectMultiScale(gray, 1.1, 4)

if len(faces) == 0:
    print("Face Not Detected(Add some light &/OR )")
else:
    for x, y, w, h in faces: 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        cv2.rectangle(frame, (x,y), (x+w,y+h),(255,0,0), 2)
        
        backtorgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        
        #Now changing that cropped file and resize
        final_image = cv2.resize(backtorgb, (224, 224))
        final_image = np.expand_dims(final_image, axis=0)
        
    

        # we have to normalize it as well 
        final_image = final_image/255.0
        predictions = new_model.predict(final_image) 
        print(predictions[0])
        