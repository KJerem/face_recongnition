import cv2
import numpy as np
import os
import sqlite3

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
camera = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer/trainingdata.yml")
# id = 0
# font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,1,1,0,1)


def getProfile(id):
    db_connection = sqlite3.connect("sqlite.db")
    # cmd="SELECT * FROM Peoples WHERE ID="+str(id)
    cursor = db_connection.execute("SELECT * FROM Peoples WHERE id=?", (id,))
    profile = None
    for row in cursor:
        profile = row
    db_connection.close()
    return profile


while True:
    ret, img = camera.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, conf = recognizer.predict(gray[y : y + h, x : x + w])
        profile = getProfile(id)
        print(profile)
        if profile != None:
            cv2.putText(
                img,
                "Name : " + str(profile[1]),
                (x, y + h + 20),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 127),
                2,
            )
            cv2.putText(
                img,
                "Age : " + str(profile[2]),
                (x, y + h + 45),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 127),
                2,
            )
            cv2.putText(
                img,
                "Gender : " + str(profile[3]),
                (x, y + h + 70),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 127),
                2,
            )
            # cv2.putText(img, output_text, (50,50),cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)

    cv2.imshow("Face", img)
    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
