import cv2  # opencv camera
import numpy as np # numpy array
import sqlite3  # sqlite databaseee

faceDetect = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)  # to detect the faces in camera

camera = cv2.VideoCapture(0)
# 0 is for web camera


def insertorupdate(Id, Name, age):  # function is for sqlite database
    db_connection = sqlite3.connect("sqlite.db")  # connect to database
    cmd = "SELECT * FROM STUDENTS WHERE ID=" + str(Id)
    cursor = db_connection.execute(cmd)  # cursor to execute statement
    isRecordExist = 0  # assume there is no record in the table
    for row in cursor:
        isRecordExist = 1

    if isRecordExist == 1:
        db_connection.execute(
            "UPDATE STUDENTS SET Name=? WHERE Id=?",
            (
                Name,
                Id,
            ),
        )
        db_connection.execute(
            "UPDATE STUDENTS SET age=? WHERE Id=?",
            (
                age,
                Id,
            ),
        )
    else:
        db_connection.execute(
            "INSERT INTO STUDENTS(Id, Name, age) VALUES(?,?,?)", (Id, Name, age)
        )

    db_connection.commit()
    db_connection.close()


# insert user defined values into table
Id = input("Enter User Id: ")
Name = input("Enter User Name: ")
age = input("Enter User Age: ")

insertorupdate(Id, Name, age)

# detect face in web camera coding
sampleNum = 0
while True:
    ret, img = camera.read()  # OPEN CAMERA
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # IMAGE CONVERT INTO BGRGRAY COLOR
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)  # scale face
    for x, y, w, h in faces:
        sampleNum = sampleNum + 1
        cv2.imwrite(
            "dataset/user." + str(Id) + "." + str(sampleNum) + ".jpg",
            gray[y : y + h, x : x + w],
        )
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.waitKey(100)  # delay time
    cv2.imshow("Face", img)
    cv2.waitKey(1)
    if sampleNum > 20:  # if the dataset is > 20 break
        break
camera.release()
cv2.destroyAllWindows() # quit
