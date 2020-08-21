
from scipy.spatial import distance as dist
#from imutils import face_utils
import cv2
import dlib
#import imutils
import matplotlib.pyplot as plt
import numpy as np


def show_the_img(img, caption):
    height, width, layers = img.shape
    size = (int(width / 4), int(height / 4))

    img_resize = cv2.resize(img, size)
    cv2.imshow(caption, img_resize)
    cv2.waitKey(1)

def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

def auto_detect_rotation(video_path, detector):
    degrees = [-1, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    text = ['none', '90c', '180', '90cc']
    counts = [0, 0, 0, 0]

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if ret == False:
            # no frame to process
            break;

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for i in range(len(degrees)):
            if degrees[i] != -1:
                rotate = cv2.rotate(gray, degrees[i])
                #show_the_img(frame_rotate, f'Degree index {i}')
                faces = detector(rotate)
            else:
                #show_the_img(frame, 'no rotation')
                faces = detector(gray)

            faces_num = len(faces)
            if faces_num != 0:
                counts[i] += faces_num
                #print('auto_detect_rotation: %d faces found for index %d' % (faces_num, i))

            if counts[i] >= 5:
                if degrees[i] != -1:
                    print('Rotation detected: need to rotate %s' % (text[i]))
                cap.release()
                return degrees[i]

def find_biggest_face(faces):
    # init
    area_max = 0

    faces_num = len(faces)
    for face in faces:
        if faces_num == 1: # fast-pass
            return face
        else:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            area = (x2 - x1) * (y2 - y1)
            if area > area_max:
                area_max = area
                face_max = face

    if faces_num > 1:
        return face_max

    return None

def find_target_face(img, faces):
    height, width, layers = img.shape
    threshold_left = width * 0.4
    threshold_right = width * 0.6
    faces_center = []

    for face in faces:
        x1 = face.left()
        x2 = face.right()

        if x1 < threshold_right and x2 > threshold_left:
            faces_center.append(face)

    faces_center_num = len(faces_center)

    if faces_center_num > 1:
        # more than one face in the center, select the biggest one
        #print('find_target_face: more than one face in the center')

        biggest = find_biggest_face(faces_center)
        if biggest == None:
            # should never happen
            print('find_target_face: fail to find biggest face')

        #show_the_img(img, 'Biggest face found')
        return biggest
    elif faces_center_num == 1:
        # unique center face found, draw a green rect on the face
        return faces_center[0]

    # all faces are not in the center
    return None

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
    else:
        print('draw_rect: unknown color %s' % (color))

def draw_face_rectangles(img, faces, target):
    for face in faces:
        draw_rect(img, face, 'red')

    if target != None:
        draw_rect(img, target, 'green')

def draw_landmarks(img, landmarks, part, marker):
    for n in range(0, 68):
        if part == 'left-eye':
            if n < 42 or n >= 48:
                continue
        elif part != 'all':
            print('draw_landmarks: unknown part %s' % (part))

        x = landmarks.part(n).x
        y = landmarks.part(n).y
        if marker == 'circle':
            cv2.circle(img, (x, y), 6, (255, 0, 0), -1)
        elif marker == 'text':
            cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
            cv2.putText(img, str(n), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 1, 8, False);
        else:
            print('draw_landmarks: unknown marker %s' % (marker))

def calculate_ear_value(landmarks, eye):
    if eye == 'left':
        # euclidean distances between the two sets of vertical eye landmarks
        A = dist.euclidean((landmarks.part(43).x, landmarks.part(43).y),
                           (landmarks.part(47).x, landmarks.part(47).y))
        B = dist.euclidean((landmarks.part(44).x, landmarks.part(44).y),
                           (landmarks.part(46).x, landmarks.part(46).y))
        # euclidean distance between the horizontal eye landmark
        C = dist.euclidean((landmarks.part(42).x, landmarks.part(42).y),
                           (landmarks.part(45).x, landmarks.part(45).y))
    elif eye == 'right':
        # euclidean distances between the two sets of vertical eye landmarks
        A = dist.euclidean((landmarks.part(38).x, landmarks.part(38).y),
                           (landmarks.part(40).x, landmarks.part(40).y))
        B = dist.euclidean((landmarks.part(37).x, landmarks.part(37).y),
                           (landmarks.part(41).x, landmarks.part(41).y))
        # euclidean distance between the horizontal eye landmark
        C = dist.euclidean((landmarks.part(39).x, landmarks.part(39).y),
                           (landmarks.part(36).x, landmarks.part(36).y))
    else:
        print('calculate_ear_value: unknown eye %s' % (eye))

    ear = (A + B) / (2.0 * C)

    return ear

def process_one_frame(frame, hog_detector, cnn_detector, predictor, use_cnn, osd_enable):
    # init for frame
    frame_state = 'init'
    ret, ear_left, ear_right = False, 0.0, 0.0

    while True:
        if frame_state == 'init':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_state = 'hog_detect'
        elif frame_state == 'hog_detect':
            # get rectangle of front faces
            faces = hog_detector(gray)
            faces_num = len(faces)

            if faces_num == 0:
                print('process_one_frame: no face found by hog detector')

                if use_cnn != False:
                    # try very slow cnn detector
                    frame_state = 'cnn_detect'
                else:
                    break;
            else:
                # we've got some faces
                frame_state = 'draw_face_rectangles'
        elif frame_state == 'cnn_detect':
            # get rectangle of front faces
            faces = cnn_detector(gray)
            faces_num = len(faces)

            if faces_num == 0:
                print('process_one_frame: no face found by cnn detector')
                break;
            else:
                # we've got some faces
                # TODO: fix the rect
                frame_state = 'draw_face_rectangles'
        elif frame_state == 'draw_face_rectangles':
            target = find_target_face(frame, faces)

            if target == None:
                # all faces found are not in the center position
                print('process_one_frame: fail to find target face')
                #show_the_img(frame, 'no center face')

                if use_cnn != False:
                    frame_state = 'cnn_detect'
                    use_cnn = False
                else:
                    if osd_enable != False:
                        draw_face_rectangles(frame, faces, target)
                    break;
            else:
                if osd_enable != False:
                    draw_face_rectangles(frame, faces, target)
                frame_state = 'draw_landmarks'
        elif frame_state == 'draw_landmarks':
            # get landmarks
            landmarks = predictor(gray, target)
            if osd_enable != False:
                draw_landmarks(frame, landmarks, 'left-eye', 'circle')

            #show_the_img(frame, 'landmarks drawn')

            frame_state = 'calculate_ear'
        elif frame_state == 'calculate_ear':
            # finally everything is done!!
            ear_left = calculate_ear_value(landmarks, 'left')
            ear_right = calculate_ear_value(landmarks, 'right')
            ret = True
            break;
        else:
            print('process_one_frame: unknown state %s' % (frame_state))

    return ret, ear_left, ear_right

def process_one_video(video_path, hog_detector, cnn_detector, predictor, output_path, output_frames):
    # init for video
    frame_index = 0
    frame_fail = 0
    times = []
    ears = []

    cap = cv2.VideoCapture(video_path)

    if cap.isOpened() == False:
        print('Fail to open source video %s' % (video_path))
        return times, ears

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print('Source video:   %s' % (video_path))
    print('  width       = %d' % (width))
    print('  height      = %d' % (height))
    print('  fps         = %d' % (fps))
    print('  fourcc      = %s' % (decode_fourcc(fourcc)))
    print('  frame_count = %d' % (frame_count))

    rotation = auto_detect_rotation(video_path, hog_detector)

    if rotation == cv2.ROTATE_90_CLOCKWISE or rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
        temp = width
        width = height
        height = temp

    if output_path != None:
        # keep the same size and fps
        # always use mp4
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (width, height))

    while True:
        ret, frame = cap.read()
        if ret == False:
            # no frame to process
            break;

        frame_index += 1

        print('Processing frame: %d / %d: ' % (frame_index, frame_count), end = '')

        if rotation != -1:
            frame = cv2.rotate(frame, rotation)

        ret, ear_left, ear_right = \
            process_one_frame(frame, hog_detector, cnn_detector, predictor,
                              False, # try again with cnn if hog fails
                              output_path != None) # draw osd info on the frame
        if ret == False:
            show_the_img(frame, 'Failed Frame')
            frame_fail += 1
        else:
            time_stamp = frame_index / fps
            times.append(time_stamp)
            ears.append(ear_left)
            print('ts: %4.3f, ear: (%4.3f, %4.3f)' % (time_stamp, ear_left, ear_right))

        if output_path != None:
            writer.write(frame)

        # don't want to process entire video
        if output_frames > 0:
            if frame_index >= output_frames:
                break;

    print('Video done, %d frames written to %s' % (frame_index, output_path))
    print('  %d frames (%3.2f%%) failed to process' % (frame_fail, frame_fail * 100.0 / frame_count))

    # clean-up
    if output_path != None:
        writer.release()
    cap.release()

    return times, ears

def main():
    source_video_path = '20200528_1AB.mp4' # rotation test
    #source_video_path = '20200429_2B.mp4'
    destination_video_path = 'test.mp4'

    hog_detector = dlib.get_frontal_face_detector()
    cnn_detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    times, ears = process_one_video(source_video_path, hog_detector, cnn_detector, predictor,
                                    destination_video_path, # None to disable output
                                    -1)                     # -1 to process all frames

    if len(times) != 0:
        plt.plot(times, ears, "r--")
        plt.title("EAR-time", fontsize = 20)
        plt.xlabel("time", fontsize = 12)
        plt.ylabel("EAR", fontsize = 12)

        plt.show()

if __name__ == '__main__':
    main()
