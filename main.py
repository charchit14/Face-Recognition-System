import csv
import cv2
from datetime import datetime
import face_recognition
import numpy as np

# 0 denotes the first webcam, 1 denotes the second, and so on... (Depends on your PC and webcam)
video_capture = cv2.VideoCapture(0)

# Loading known faces and encoding them
# Encoding converts a face/image into a number so that it is easier to compare 
erling_face = face_recognition.load_image_file("faces/erling.jpg")
erling_encoding = face_recognition.face_encodings(erling_face)[0] # 0 since it returns a list, and I need the first image. (For example, if written 5, it returns a list of 5 images/faces)`

kdb_face = face_recognition.load_image_file("faces/kdb.jpg")
kdb_encoding = face_recognition.face_encodings(kdb_face)[0]

ruben_face = face_recognition.load_image_file("faces/ruben.jpg")
ruben_encoding = face_recognition.face_encodings(ruben_face)[0]

known_face_encodings = [erling_encoding, kdb_encoding, ruben_encoding]
known_face_names = ["Erling Haaland", "Kevin De Bruyne", "Ruben Dias"]

# List of expected people
people = known_face_names.copy()

face_locations = []
face_encodings = []

# Getting the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

name = ""  # Initializing name outside of the loop

while True:
    _, frame = video_capture.read()  # video_capture has two arguments: the first is whether video capture was true or not, and the second is the frame. So, '_' is the first argument
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognizing faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)  # This will tell us how close the face is with the saved face (comparison)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Showing the name in text if the person is present/detected
        if name in people:  # Check if the name is in the list of expected people
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomleftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, "This is " + name, bottomleftCornerOfText, font, fontScale, fontColor, thickness, lineType)

            people.remove(name)
            current_time = now.strftime("%H-%M%S")
            lnwriter.writerow([name, current_time])

    cv2.imshow("Face Detection Window", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
