import cv2
import numpy as np
import dlib


source_video_name = '20200429_2B.mp4'
destination_video_name = 'test.mp4'

def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

def draw_rect(img, rect, color):
    # draw rectangle
    x1 = rect.left()
    y1 = rect.top()
    x2 = rect.right()
    y2 = rect.bottom()
    if color == 'green':
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    elif color == 'red':
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    #print(f'draw_rect: ({x1}, {y1}), ({x2}, {y2}), {color}, {(x2 - x1) * (y2 - y1)}')

def draw_landmarks(img, landmarks):
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(img, (x, y), 6, (255, 0, 0), -1)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(source_video_name)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

handled_frames = 0
output_frames = -1

print('Source video: ' + source_video_name)
print('  width  = ' + str(width))
print('  height = ' + str(height))
print('  fps    = ' + str(fps))
print('  fourcc = ' + str(fourcc))

# keep the same size and fps
# always use mp4
writer = cv2.VideoWriter(destination_video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# read the first frame
ret, frame = cap.read()

while ret:
    print('Progress[' + str(handled_frames) + '/' + str(frame_count) + ']', end="\r")
    #print(f'Processing frame: {handled_frames + 1}')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # init
    area_max = 0

    # get rectangle of front faces
    faces = detector(gray)
    faces_num = len(faces)
    for face in faces:
        if faces_num == 1:
            # draw a green rect on the face
            draw_rect(frame, face, 'green')

            # get landmarks
            landmarks = predictor(gray, face)
            draw_landmarks(frame, landmarks)
        else:
            # draw a red rect on every faces
            draw_rect(frame, face, 'red')

            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            area = (x2 - x1) * (y2 - y1)
            if area > area_max:
                area_max = area
                face_max = face
                #print(f'face_max: ({face_max.left()}, {face_max.top()}), ({face_max.right()}, {face_max.bottom()}), {(face_max.right() - face_max.left()) * (face_max.bottom() - face_max.top())}')

    if faces_num > 1:
        # draw a green rect on the face
        draw_rect(frame, face_max, 'green')

        # get landmarks
        landmarks = predictor(gray, face_max)
        draw_landmarks(frame, landmarks)

    #print(str(faces_num) + ' faces found\n')
    #frameS = cv2.resize(frame, (960, 540)) 
    #cv2.imshow("Frame", frameS)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imshow("Frame", frame)

    #height, width, layers = frame.shape
    #size = (width, height)
    #frames_array.append(frame)

    writer.write(frame)
    handled_frames += 1

    if output_frames > 0:
        if handled_frames >= output_frames:
            break;

    # read next frame
    ret, frame = cap.read()

print('\nWork done, total ' + str(handled_frames) + ' frames written to ' + destination_video_name)

writer.release()
cap.release()
