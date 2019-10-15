import cv2

faces = cv2.imread("faces.png")
faces_gray = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)


haar_face = cv2.CascadeClassifier("face-haar.xml")
haar_eye = cv2.CascadeClassifier("eye-haar.xml")
detected_faces = haar_face.detectMultiScale(faces_gray)

for (x, y, w, h) in detected_faces:
    cv2.rectangle(faces, (x, y), (x+w, y+h), (255,0,0), 2)
    eye_color=faces[y:y+h, x:x+h]
    eye_gray=faces_gray[y:y+h, x:x+h]
    detected_eye = haar_eye.detectMultiScale(eye_gray, 1.2, 10)
    for (xm, ym, wm, hm) in detected_eye:
            cv2.rectangle(eye_color, (xm, ym), (xm+wm, ym+hm), (0,255,0), 2)


cv2.imshow("Title", faces)
cv2.imshow("gray", faces_gray)

cv2.waitKey(0)
cv2.destroyAllWindows()