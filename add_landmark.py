import cv2
import numpy as np
import dlib


source_video_name = '20200429_2B.mp4'
destination_video_name = 'test.mp4'

def show_the_img(img):
    frame = cv2.resize(img, (960, 540))
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)

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

def draw_biggest_face(img, faces):
    # init
    area_max = 0

    faces_num = len(faces)
    for face in faces:
        if faces_num == 1: # fast-pass
            # draw a green rect on the only-one face
            draw_rect(img, face, 'green')

            return face
        else:
            # draw a red rect on every faces first
            draw_rect(img, face, 'red')

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
        # draw a green rect on the biggest face
        draw_rect(img, face_max, 'green')

        return face_max

    return None

def draw_center_face(img, faces):
    threshold_left = width * 0.4
    threshold_right = width * 0.6
    faces_center = []

    for face in faces:
        x1 = face.left()
        x2 = face.right()

        if x1 < threshold_right and x2 > threshold_left:
            faces_center.append(face)
        else:
            # draw a red rect on faces not in the center
            draw_rect(img, face, 'red')

    faces_center_num = len(faces_center)

    if faces_center_num > 1:
        # more than one face in the center, select the biggest one
        #print('draw_center_face: more than one face in the center\n')

        biggest = draw_biggest_face(img, faces_center)
        #if biggest == None:
        #    print(f'draw_center_face: fail to draw biggest face\n')

        #show_the_img(img)
        return biggest

    elif faces_center_num == 1:
        # unique center face found, draw a green rect on the face
        draw_rect(img, faces_center[0], 'green')

        return faces_center[0]

    return None

#face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(source_video_name)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print('Source video: ' + source_video_name)
print('  width  = ' + str(width))
print('  height = ' + str(height))
print('  fps    = ' + str(fps))
print('  fourcc = ' + str(fourcc))

# keep the same size and fps
# always use mp4
writer = cv2.VideoWriter(destination_video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# init for video
handled_frames = 0
output_frames = 30

hog_detector = dlib.get_frontal_face_detector()
cnn_detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame = cap.read()
    if ret == False:
        # no frame to process
        break;

    # init for frame
    frame_state = 'init'
    use_cnn = True

    while True:
        if frame_state == 'init':
            #print('Progress[' + str(handled_frames) + '/' + str(frame_count) + ']', end="\r")
            print(f'Processing frame: {handled_frames + 1}')

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_state = 'hog_detect'
        elif frame_state == 'hog_detect':
            # get rectangle of front faces
            faces = hog_detector(gray)
            faces_num = len(faces)

            if faces_num == 0:
                print('no face found by hog detector')

                if use_cnn != False:
                    frame_state = 'cnn_detect'
                else:
                    frame_state = 'next_frame'
            else:
                # we've got some faces
                frame_state = 'draw_face_rectangles'
        elif frame_state == 'cnn_detect':
            # get rectangle of front faces
            faces = cnn_detector(gray)
            faces_num = len(faces)

            if faces_num == 0:
                print('no face found by cnn detector')
                frame_state = 'next_frame'
            else:
                # we've got some faces
                frame_state = 'draw_face_rectangles'
        elif frame_state == 'draw_face_rectangles':
            face = draw_center_face(frame, faces)
            if face == None:
                # all faces found are not in the center position
                print('fail to draw center face\n')
                #show_the_img(frame)

                if use_cnn != False:
                    frame_state = 'cnn_detect'
                    use_cnn = False
                else:
                    frame_state = 'next_frame'
            else:
                frame_state = 'draw_landmarks'
        elif frame_state == 'draw_landmarks':
            # get landmarks
            landmarks = predictor(gray, face)
            draw_landmarks(frame, landmarks)
            frame_state = 'next_frame'
        elif frame_state == 'next_frame':
            # next frame
            writer.write(frame)
            handled_frames += 1
            break;

    #print(str(faces_num) + ' faces found\n')

    #height, width, layers = frame.shape
    #size = (width, height)
    #frames_array.append(frame)

    # don't want to process entire video
    if output_frames > 0:
        if handled_frames >= output_frames:
            break;

print('\nWork done, total ' + str(handled_frames) + ' frames written to ' + destination_video_name)

writer.release()
cap.release()
