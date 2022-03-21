import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv
import argparse
import pandas as pd


# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-f", "--file", help = "Attendance Csv File", required=True)
parser.add_argument("-i", "--images", help = "Training images", required=True)

# Read arguments from command line
args = vars(parser.parse_args())

ATTENDANCE_PATH = args["file"]
TRAINING_IMAGES_PATH = args["images"]


df = pd.DataFrame(columns=('id', 'Name', 'Time'))
df.to_csv(ATTENDANCE_PATH, index=False)


student_info = {}
data = pd.read_csv(TRAINING_IMAGES_PATH)


def findEncodings():
    encodeList = []

    for index, item in data.iterrows():
        img = cv2.imread(item["image_url"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


encodeListKnown = findEncodings()


def markAttendance(id,name):
    with open(ATTENDANCE_PATH, 'a+') as f:
        myDataList = f.readlines()
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        print(name, dtString)
        f.writelines(f'{id},{name},{dtString}\n')


print('Encoding Complete')

cap = cv2.VideoCapture(0)
AttendanceList = {}

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = data["name"][matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            if AttendanceList.get(data["id"][matchIndex]) == None:
                markAttendance(data["id"][matchIndex],name)
                AttendanceList[data["id"][matchIndex]] = name


    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
