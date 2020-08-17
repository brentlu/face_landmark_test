import cv2
import numpy as np
import dlib


def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture("20200429_2B.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

second = 10
total_frames = second * fps

print('width='+str(width)+' height='+str(height)+' fps='+str(fps)+' fourcc='+decode_fourcc(fourcc))

# keep the same size and fps
# always use mp4
writer = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

frames_array = []

# read the first frame
ret, frame = cap.read()

while ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)


    #frameS = cv2.resize(frame, (960, 540)) 
    #cv2.imshow("Frame", frameS)
    #cv2.imshow("Frame", frame)

    #height, width, layers = frame.shape
    #size = (width, height)
    #frames_array.append(frame)
    writer.write(frame)

    total_frames -= 1
    if total_frames == 0:
        break

    # read next frame
    ret, frame = cap.read()

writer.release()
cap.release()
