# We have imported an pre trained ml model famously from haarcascade. Its in xml file format and we will use that for the face detection to make things a LOT easier for us.
import cv2 

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
print("Webcam does work")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to bring it on")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0), 2)

    cv2.imshow("Live Face Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

#Finally clein the camera resources
cap.release()
cv2.destroyAllWindows()

