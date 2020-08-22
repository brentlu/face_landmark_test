
from scipy.spatial import distance as dist
#from imutils import face_utils
import csv
import cv2
import dlib
import hashlib
#import imutils
import matplotlib.pyplot as plt
import numpy as np
import os


def show_the_img(img, caption):
    height, width, layers = img.shape
    size = (int(width / 4), int(height / 4))

    img_resize = cv2.resize(img, size)
    cv2.imshow(caption, img_resize)
    cv2.waitKey(1)

    return

def get_data_path(directory):
    if directory == 'root':
        return './data'
    elif directory == 'video':
        return './data/video'
    elif directory == 'csv':
        return './data/csv'
    else:
        print('get_data_path: unknown directory %s' % (directory))

    # should not get here
    return ''

def create_data_directory():
    # create basic layout of data directory
    print('Check data directory:')

    data_path = get_data_path('root')
    video_path = get_data_path('video')
    csv_path = get_data_path('csv')

    abs_data_path = os.path.abspath(data_path)
    abs_video_path = os.path.abspath(video_path)
    abs_csv_path = os.path.abspath(csv_path)

    pathes = (abs_data_path, abs_video_path, abs_csv_path)

    for path in pathes:
        if os.path.isdir(path) == False:
            try:
                print('  creating %s' % (path))
                os.mkdir(path)
            except OSError:
                print('  fail to create %s directory' % (path))
                return False

    return True

def get_md5_digest(path, size):
    block_size = 4096
    remain_size = size

    file_hash = hashlib.md5()
    with open(path, 'rb') as f:
        while remain_size > block_size:
            block = f.read(block_size)
            file_hash.update(block)
            remain_size -= block_size

        if remain_size != 0:
            block = f.read(remain_size)
            file_hash.update(block)

    return file_hash.hexdigest()

def prepare_one_video(input_video_path):
    if os.path.isfile(input_video_path) == False:
        print('  file %s does not exist' % (input_video_path))
        return False, None, None

    # first 64KB should be sufficient
    file_hash = get_md5_digest(input_video_path, 64 * 1024)

    # remove directory part
    _, file_name = os.path.split(input_video_path)

    # remove ext part
    file_name, _ = os.path.splitext(file_name)

    # generate file name of output video and csv file
    output_video_name = file_name + '-' + str(file_hash) + '.mp4'
    output_csv_name = file_name + '-' + str(file_hash) + '.csv'

    output_video_path = os.path.join(get_data_path('video'), output_video_name)
    output_csv_path = os.path.join(get_data_path('csv'), output_csv_name)

    output_video_path = os.path.abspath(output_video_path)
    output_csv_path = os.path.abspath(output_csv_path)

    if os.path.isfile(output_video_path):
        print('  video data already exists')

    if os.path.isfile(output_csv_path):
        print('  csv data already exists')

    return True, output_video_path, output_csv_path

def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

def auto_detect_rotation(video_path, detector):
    degrees = [-1, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    text = ['none', '90 degree clockwise', '180 degree', '90 degree counter clockwise']
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
                    print('Auto-rotation info:')
                    print('  need to rotate %s' % (text[i]))
                cap.release()
                return degrees[i]

    # should not get here
    return -1

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
        return biggest, faces_center_num
    elif faces_center_num == 1:
        # unique center face found, draw a green rect on the face
        return faces_center[0], faces_center_num

    # all faces are not in the center
    return None, 0

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

    return

def draw_face_rectangles(img, faces, target):
    for face in faces:
        draw_rect(img, face, 'red')

    if target != None:
        draw_rect(img, target, 'green')

    return

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

    return

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

def process_one_frame(frame, hog_detector, cnn_detector, predictor, frame_result, use_cnn, osd_enable):
    # init for frame
    state = 'init'

    while True:
        if state == 'init':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # first try hog detector
            state = 'hog_detect'
        elif state == 'hog_detect':
            # get rectangle of front faces
            faces = hog_detector(gray)
            faces_num = len(faces)

            if faces_num == 0:
                print('process_one_frame: no face found by hog detector')

                # try very slow cnn detector to find more faces
                if use_cnn != False:
                    state = 'cnn_detect'
                    continue

                return False

            # we've got some faces
            frame_result['detector'] = 'h'
            frame_result['total_face_num'] = faces_num

            state = 'process_faces'
        elif state == 'cnn_detect':
            # get rectangle of front faces
            faces = cnn_detector(gray)
            faces_num = len(faces)

            if faces_num == 0:
                print('process_one_frame: no face found by cnn detector')
                return False

            # we've got some faces
            # TODO: fix the rect
            frame_result['detector'] = 'c'
            frame_result['total_face_num'] = faces_num

            state = 'process_faces'
        elif state == 'process_faces':
            # find the target face
            target, center_num = find_target_face(frame, faces)

            frame_result['center_face_num'] = center_num

            if target == None:
                # all faces found are not in the center position
                print('process_one_frame: fail to find target face')

                # try very slow cnn detector to find more faces
                if use_cnn != False:
                    use_cnn = False

                    state = 'cnn_detect'
                    continue

                # draw red rectangles before leaving
                if osd_enable != False:
                    draw_face_rectangles(frame, faces, target)

                return False

            frame_result['target_left'] = target.left()
            frame_result['target_top'] = target.top()
            frame_result['target_right'] = target.right()
            frame_result['target_bottom'] = target.bottom()

            if osd_enable != False:
                draw_face_rectangles(frame, faces, target)

            # get landmarks of the target face
            landmarks = predictor(gray, target)

            if osd_enable != False:
                draw_landmarks(frame, landmarks, 'left-eye', 'circle')

            # calculate ear values for both eyes
            frame_result['ear_left'] = calculate_ear_value(landmarks, 'left')
            frame_result['ear_right'] = calculate_ear_value(landmarks, 'right')

            return True
        else:
            print('process_one_frame: unknown state %s' % (state))
            return False

    # should not get here
    return False

def process_one_video(input_video_path, hog_detector, cnn_detector, predictor, output_video_path, output_csv_path, output_frames):
    # init for video
    frame_index = 0
    frame_fail_count = 0
    csv_fields = ['index', 'detector', 'total_face_num', 'center_face_num', 'target_left', 'target_top', 'target_right', 'target_bottom', 'time_stamp', 'ear_left', 'ear_right']
    times = []
    ears = []

    cap = cv2.VideoCapture(input_video_path)

    if cap.isOpened() == False:
        print('Fail to open source video %s' % (input_video_path))
        return times, ears

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print('Path info:')
    print('  input video  = %s' % (input_video_path))
    print('  output video = %s' % (output_video_path))
    print('  output csv   = %s' % (output_csv_path))
    print('Input video info:')
    print('  width       = %d' % (width))
    print('  height      = %d' % (height))
    print('  fps         = %d' % (fps))
    print('  fourcc      = %s' % (decode_fourcc(fourcc)))
    print('  frame_count = %d' % (frame_count))

    rotation = auto_detect_rotation(input_video_path, hog_detector)

    if rotation == cv2.ROTATE_90_CLOCKWISE or rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
        temp = width
        width = height
        height = temp

    if output_video_path != None:
        # always use mp4
        output_size = (int(width / 4), int(height / 4))
        video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                       fps, output_size)

    if output_csv_path != None:
        csvfile = open(output_csv_path, 'w', newline='')
        csv_writer = csv.DictWriter(csvfile, fieldnames = csv_fields)
        csv_writer.writeheader()

    print('Processing video:')

    while True:
        ret, frame = cap.read()
        if ret == False:
            # no frame to process
            break;

        # prepare an empty dict
        frame_result = {
            'index': 0,
            'detector': 'h', # 'h' for hog and 'c' for cnn
            'total_face_num' : 0,
            'center_face_num' : 0,
            'target_left' : 0,
            'target_top' : 0,
            'target_right' : 0,
            'target_bottom' : 0,
            'time_stamp' : 0.0,
            'ear_left': 0.0,
            'ear_right': 0.0,
        }

        frame_index += 1
        frame_result['index'] = frame_index

        print('  frame: (%3d/%d), ' % (frame_index, frame_count), end = '')

        if rotation != -1:
            frame = cv2.rotate(frame, rotation)

        ret = process_one_frame(frame, hog_detector, cnn_detector, predictor, frame_result,
                                False, # try again with cnn if hog fails
                                output_video_path != None) # draw osd info on the frame
        if ret == False:
            show_the_img(frame, 'Failed Frame')
            frame_fail_count += 1
        else:
            time_stamp = frame_index / fps
            frame_result['time_stamp'] = time_stamp

            times.append(time_stamp)
            ears.append(frame_result['ear_left'])
            print('ts: %4.3f, ear: (%4.3f,%4.3f)' % (time_stamp, frame_result['ear_left'], frame_result['ear_right']))

            if output_csv_path != None:
                csv_writer.writerow(frame_result)

        if output_video_path != None:
            frame = cv2.resize(frame, output_size)
            video_writer.write(frame)

        # don't want to process entire video
        if output_frames > 0:
            if frame_index >= output_frames:
                break;

    print('Statistic info:')
    print('  total %d frames' % (frame_count))
    print('  %d frames processed' % (frame_index))
    print('  %d frames (%3.2f%%) failed' % (frame_fail_count, frame_fail_count * 100.0 / frame_index))

    # clean-up
    if output_video_path != None:
        video_writer.release()
    if output_csv_path != None:
        csvfile.close()
    cap.release()

    return times, ears


def main():

    input_video_path = '20200528_1AB.mp4' # rotation test

    # init for all videos
    hog_detector = dlib.get_frontal_face_detector()
    cnn_detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # check data directory
    if create_data_directory() == False:
        print('fail to create data directory')
        return

    # translate to abs path
    input_video_path = os.path.abspath(input_video_path)

    ret, output_video_path, output_csv_path = prepare_one_video(input_video_path)
    if ret == False:
        print('fail to prepare video params')
        return

    times, ears = process_one_video(input_video_path, hog_detector, cnn_detector, predictor,
                                    output_video_path, # None to disable output
                                    output_csv_path,
                                    -1)                # -1 to process all frames

    return
    if len(times) != 0:
        plt.plot(times, ears, 'r--')
        plt.title('EAR-time', fontsize = 20)
        plt.xlabel('time', fontsize = 12)
        plt.ylabel('EAR', fontsize = 12)

        plt.show()

if __name__ == '__main__':
    main()
