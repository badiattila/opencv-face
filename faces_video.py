import numpy as np
import cv2

cap = cv2.VideoCapture(0)
haar_face = cv2.CascadeClassifier("face-haar.xml")
haar_eye = cv2.CascadeClassifier("eye-haar.xml")

while(True):
    # Capture frame-by-frame
    ret, faces = cap.read()

    # Our operations on the frame come here
    faces_gray = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)

    detected_faces = haar_face.detectMultiScale(faces_gray, 1.1, 10)

    for (x, y, w, h) in detected_faces:
        cv2.rectangle(faces, (x, y), (x+w, y+h), (255,0,0), 2)
        eye_color=faces[y:y+h, x:x+h]
        eye_gray=faces_gray[y:y+h, x:x+h]
        detected_eye = haar_eye.detectMultiScale(eye_gray, 1.1, 10)
        for (xm, ym, wm, hm) in detected_eye:
            cv2.rectangle(eye_color, (xm, ym), (xm+wm, ym+hm), (0,255,0), 2)

    # Display the resulting frame
    cv2.imshow('frame',faces)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()