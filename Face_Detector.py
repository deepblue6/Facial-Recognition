import cv2
from random import randrange

# Loading pre-trained data into a variable
trained_face_data = cv2.CascadeClassifier('haarscade.xml')

# Choose an image to detect faces in

# img = cv2.imread('WhatsApp Image 2023-06-13 at 9.58.47 AM.jpeg')

# Capturing video from webcam
webcam = cv2.VideoCapture(0)

while True:

    ## Read current frame
    succesful_frame_read, frame = webcam.read()

    # Must make the image grayscale
    gs_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(gs_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 15)

    cv2.imshow('Program...', frame)
    cv2.waitKey(1)

    if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty('Program...', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release Video Camera
webcam.release()
