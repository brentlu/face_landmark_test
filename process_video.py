#!/usr/bin/python3

import csv
import cv2
import dlib
import hashlib
import magic
import numpy as np
import os
import re
import subprocess
import time


logger_started = 0

def show_the_img(img, caption):
    height, width, layers = img.shape
    size = (int(width / 4), int(height / 4))

    img_resize = cv2.resize(img, size)
    cv2.imshow(caption, img_resize)
    cv2.waitKey(1)

    return

def logger_start():
    global logger_started
    global logger_file

    logger_print('Start logger:')

    # get current time
    now = time.gmtime()

    timestamp = time.strftime('%Y-%m%d-%H%M', now)

    log_path_base = os.path.join(get_data_path('log'), timestamp)
    log_path_base = os.path.abspath(log_path_base)

    for n in range(1, 100):
        log_path = '%s-%s.log' % (log_path_base, str(n))
        if os.path.exists(log_path) == False:
            break

    logger_file = open(log_path, 'w')
    logger_started = 1

    logger_print('  log path = %s' % (log_path))

    return

def logger_print(string, end = '\n'):
    global logger_started
    global logger

    print(string, end = end)

    if end == '\r':
        end = '\n'

    if logger_started != 0:
        logger_file.write('%s%s' % (string, end))

    return

def logger_stop():
    global logger_started
    global logger

    if logger_started != 0:
        logger_file.close()

    return

def get_data_path(directory):
    if directory == 'data':
        return './data'
    elif directory == 'csv':
        return './data/csv'
    elif directory == 'log':
        return './data/log'
    elif directory == 'video':
        return './data/video'
    else:
        logger_print('get_data_path: unknown directory %s' % (directory))

    # should not get here
    return ''

def check_data_directory():
    # create basic layout of data directory
    logger_print('Check data directory:')

    data_path = get_data_path('data')
    data_path = os.path.abspath(data_path)

    csv_path = get_data_path('csv')
    csv_path = os.path.abspath(csv_path)

    log_path = get_data_path('log')
    log_path = os.path.abspath(log_path)

    video_path = get_data_path('video')
    video_path = os.path.abspath(video_path)

    pathes = (data_path, csv_path, log_path, video_path)

    for path in pathes:
        if os.path.isdir(path) == False:
            try:
                logger_print('  create %s directory' % (path))
                os.mkdir(path)
            except OSError:
                logger_print('  fail to create %s directory' % (path))
                return False

    logger_print('  all good')

    return True

def calculate_md5_digest(path, size):
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

def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

def auto_detect_rotation(video_path, detector):
    degrees = [-1, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    text = ['none', '90 degree clockwise', '180 degree', '90 degree counter clockwise']
    counts = [0, 0, 0, 0]

    cap = cv2.VideoCapture(video_path)

    logger_print('  orientation detection:')

    while True:
        ret, frame = cap.read()
        if ret == False:
            # no frame to process
            break;

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for i in range(len(degrees)):
            if degrees[i] != -1:
                rotate = cv2.rotate(gray, degrees[i])
                faces = detector(rotate)
            else:
                faces = detector(gray)

            faces_num = len(faces)
            if faces_num != 0:
                counts[i] += faces_num
                #logger_print('auto_detect_rotation: %d faces found for %s' % (faces_num, text[i]))

            if counts[i] >= 5:
                logger_print('    need to rotate %s' % (text[i]))
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
        biggest = find_biggest_face(faces_center)
        if biggest == None:
            # should never happen
            logger_print('find_target_face: fail to find biggest face')

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
        logger_print('draw_rect: unknown color %s' % (color))

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
            logger_print('draw_landmarks: unknown part %s' % (part))

        x = landmarks.part(n).x
        y = landmarks.part(n).y
        if marker == 'circle':
            cv2.circle(img, (x, y), 6, (255, 0, 0), -1)
        elif marker == 'text':
            cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
            cv2.putText(img, str(n), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 1, 8, False);
        else:
            logger_print('draw_landmarks: unknown marker %s' % (marker))

    return

def process_one_frame(frame, hog_detector, cnn_detector, predictor, frame_result, options):
    # init for frame
    state = 'init'
    use_cnn = options['use_cnn_when_fail']

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
                logger_print('process_one_frame: no face found by hog detector')

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
                logger_print('process_one_frame: no face found by cnn detector')
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
                logger_print('process_one_frame: fail to find target face')

                # try very slow cnn detector to find more faces
                if use_cnn != False:
                    use_cnn = False

                    state = 'cnn_detect'
                    continue

                # draw red rectangles before leaving
                if options['output_video'] != False:
                    draw_face_rectangles(frame, faces, target)

                return False

            frame_result['target_left'] = target.left()
            frame_result['target_top'] = target.top()
            frame_result['target_right'] = target.right()
            frame_result['target_bottom'] = target.bottom()

            if options['output_video'] != False:
                draw_face_rectangles(frame, faces, target)

            # get landmarks of the target face
            landmarks = predictor(gray, target)

            if options['output_video'] != False:
                draw_landmarks(frame, landmarks, 'left-eye', 'circle')

            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y

                frame_result['mark_%d_x' % (n)] = x
                frame_result['mark_%d_y' % (n)] = y

            logger_print('process_one_frame: success', end = '\r')
            return True
        else:
            logger_print('process_one_frame: unknown state %s' % (state))
            return False

    # should not get here
    return False

def process_one_video_internal(input_video_path, input_csv_path, hog_detector, cnn_detector, predictor, options):
    # init for video
    csv_index = 0
    frame_index = 0
    frame_fail_count = 0
    csv_fields = ['index', 'detector', 'total_face_num', 'center_face_num', 'target_left', 'target_top', 'target_right', 'target_bottom', 'time_stamp', \
                  'mark_0_x', 'mark_0_y', 'mark_1_x', 'mark_1_y', 'mark_2_x', 'mark_2_y', 'mark_3_x', 'mark_3_y', 'mark_4_x', 'mark_4_y', 'mark_5_x', 'mark_5_y', 'mark_6_x', 'mark_6_y', 'mark_7_x', 'mark_7_y', 'mark_8_x', 'mark_8_y', 'mark_9_x', 'mark_9_y', \
                  'mark_10_x', 'mark_10_y', 'mark_11_x', 'mark_11_y', 'mark_12_x', 'mark_12_y', 'mark_13_x', 'mark_13_y', 'mark_14_x', 'mark_14_y', 'mark_15_x', 'mark_15_y', 'mark_16_x', 'mark_16_y', 'mark_17_x', 'mark_17_y', 'mark_18_x', 'mark_18_y', 'mark_19_x', 'mark_19_y', \
                  'mark_20_x', 'mark_20_y', 'mark_21_x', 'mark_21_y', 'mark_22_x', 'mark_22_y', 'mark_23_x', 'mark_23_y', 'mark_24_x', 'mark_24_y', 'mark_25_x', 'mark_25_y', 'mark_26_x', 'mark_26_y', 'mark_27_x', 'mark_27_y', 'mark_28_x', 'mark_28_y', 'mark_29_x', 'mark_29_y', \
                  'mark_30_x', 'mark_30_y', 'mark_31_x', 'mark_31_y', 'mark_32_x', 'mark_32_y', 'mark_33_x', 'mark_33_y', 'mark_34_x', 'mark_34_y', 'mark_35_x', 'mark_35_y', 'mark_36_x', 'mark_36_y', 'mark_37_x', 'mark_37_y', 'mark_38_x', 'mark_38_y', 'mark_39_x', 'mark_39_y', \
                  'mark_40_x', 'mark_40_y', 'mark_41_x', 'mark_41_y', 'mark_42_x', 'mark_42_y', 'mark_43_x', 'mark_43_y', 'mark_44_x', 'mark_44_y', 'mark_45_x', 'mark_45_y', 'mark_46_x', 'mark_46_y', 'mark_47_x', 'mark_47_y', 'mark_48_x', 'mark_48_y', 'mark_49_x', 'mark_49_y', \
                  'mark_50_x', 'mark_50_y', 'mark_51_x', 'mark_51_y', 'mark_52_x', 'mark_52_y', 'mark_53_x', 'mark_53_y', 'mark_54_x', 'mark_54_y', 'mark_55_x', 'mark_55_y', 'mark_56_x', 'mark_56_y', 'mark_57_x', 'mark_57_y', 'mark_58_x', 'mark_58_y', 'mark_59_x', 'mark_59_y', \
                  'mark_60_x', 'mark_60_y', 'mark_61_x', 'mark_61_y', 'mark_62_x', 'mark_62_y', 'mark_63_x', 'mark_63_y', 'mark_64_x', 'mark_64_y', 'mark_65_x', 'mark_65_y', 'mark_66_x', 'mark_66_y', 'mark_67_x', 'mark_67_y']

    logger_print('Process video:')

    cap = cv2.VideoCapture(input_video_path)

    if cap.isOpened() == False:
        logger_print('  fail to open %s' % (input_video_path))
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger_print('  video property:')
    logger_print('    width       = %d' % (width))
    logger_print('    height      = %d' % (height))
    logger_print('    fps         = %d' % (fps))
    logger_print('    fourcc      = %s' % (decode_fourcc(fourcc)))
    logger_print('    frame_count = %d' % (frame_count))

    rotation = auto_detect_rotation(input_video_path, hog_detector)

    if rotation == cv2.ROTATE_90_CLOCKWISE or rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
        temp = width
        width = height
        height = temp

    if options['output_video'] != False:
        # always use mp4
        video_writer = cv2.VideoWriter(options['output_video_path'], cv2.VideoWriter_fourcc(*'mp4v'),
                                       fps, (width, height))

    if options['input_csv'] != False:
        csv_file_read = open(input_csv_path, 'r', newline = '')
        csv_reader = csv.DictReader(csv_file_read)

        try:
            row = next(csv_reader)
            csv_index = int(row['index'])
        except StopIteration:
            csv_index = frame_count + 1

    if options['output_csv'] != False:
        csv_file_write = open(options['output_csv_path'], 'w', newline='')
        csv_writer = csv.DictWriter(csv_file_write, fieldnames = csv_fields)
        csv_writer.writeheader()

    while True:
        ret, frame = cap.read()
        if ret == False:
            # no frame to process
            break;

        # prepare an empty dict
        frame_result = {}

        frame_index += 1

        logger_print('  frame: (%3d/%d), ' % (frame_index, frame_count), end = '')

        if options['input_csv'] != False:
            if frame_index == csv_index:
                # copy dict entry
                logger_print('copy from csv file', end = '\r')
                if options['output_csv'] != False:
                    csv_writer.writerow(row)
                try:
                    row = next(csv_reader)
                    csv_index = int(row['index'])
                except StopIteration:
                    csv_index = frame_count + 1
                continue

        if rotation != -1:
            frame = cv2.rotate(frame, rotation)

        ret = process_one_frame(frame, hog_detector, cnn_detector, predictor, frame_result, options)
        if ret == False:
            frame_fail_count += 1
        else:
            if options['output_csv'] != False:
                time_stamp = frame_index / fps
                frame_result['time_stamp'] = time_stamp
                frame_result['index'] = frame_index

                csv_writer.writerow(frame_result)

        if options['output_video'] != False:
            video_writer.write(frame)

        # don't want to process entire video
        if options['frame_index_max'] > 0:
            if frame_index >= options['frame_index_max']:
                break;

    logger_print('Statistic:')
    logger_print('  total %d frames' % (frame_count))
    logger_print('  %d frames processed' % (frame_index))
    logger_print('  %d frames (%3.2f%%) failed' % (frame_fail_count, frame_fail_count * 100.0 / frame_index))

    # clean-up
    if options['output_video'] != False:
        video_writer.release()
    if options['input_csv'] != False:
        csv_file_read.close()
    if options['output_csv'] != False:
        csv_file_write.close()
    cap.release()

    return True

def compress_one_video(video_path):
    directory, _ = os.path.split(video_path)
    tmp_path = os.path.join(directory, 'tmp.mp4')

    # delete the tmp video if already exist
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    os.rename('%s' % video_path, '%s' % tmp_path)

    # prepare ffmpeg command
    cmd = ['ffmpeg', '-i', tmp_path, '-c:v', 'libx264', '-preset', 'veryslow', '-crf', '28', '-c:a', 'copy', video_path]

    logger_print('Compress output video:')

    p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE, universal_newlines = True)
    outs, errs = p.communicate()

    if p.returncode != 0:
        logger_print('  compression fail')
        logger_print('  ffmpeg output:')
        logger_print(errs)

        if os.path.exists(video_path):
            os.remove(video_path)

        os.rename('%s' % tmp_path, '%s' % video_path)
        return False

    logger_print('  compression success')
    logger_print('  ffmpeg output:')
    logger_print(outs)

    # delete tmp video before leaving
    os.remove(tmp_path)

    return True

def process_one_video(input_video_path, hog_detector, cnn_detector, predictor, options):
    input_csv_path = ''

    # first 64KB should be sufficient
    file_hash = calculate_md5_digest(input_video_path, 64 * 1024)

    # remove directory part
    _, file_name = os.path.split(input_video_path)

    # remove ext part
    file_name, _ = os.path.splitext(file_name)

    logger_print('Check process parameters:')
    logger_print('  input video path  = %s' % (input_video_path))

    # generate file name of output video and csv file
    if options['output_csv'] != False:
        output_csv_name = '%s-%s.csv' % (file_name, str(file_hash))
        output_csv_path = os.path.join(get_data_path('csv'), output_csv_name)
        output_csv_path = os.path.abspath(output_csv_path)
        options['output_csv_path'] = output_csv_path

        logger_print('  output csv path   = %s' % (output_csv_path))

        if os.path.isfile(output_csv_path) != False:
            if options['update_csv'] != False:
                # partial update csv file
                logger_print('  csv data will be updated')

                directory, _ = os.path.split(output_csv_path)
                input_csv_path = os.path.join(directory, 'tmp.csv')

                # delete the tmp csv if already exist
                if os.path.exists(input_csv_path):
                    os.remove(input_csv_path)

                os.rename('%s' % output_csv_path, '%s' % input_csv_path)
                options['input_csv'] = True

            elif options['overwrite_csv'] == False:
                logger_print('  csv data exists, abort')

                return False

            else:
                logger_print('  csv data will be overwritten')
    else:
        output_csv_path = None

    if options['output_video'] != False:
        output_video_name = '%s-%s.mp4' % (file_name, str(file_hash))
        output_video_path = os.path.join(get_data_path('video'), output_video_name)
        output_video_path = os.path.abspath(output_video_path)
        options['output_video_path'] = output_video_path

        logger_print('  output video path = %s' % (output_video_path))

        if os.path.isfile(output_video_path) != False:
            if options['overwrite_video'] == False:
                logger_print('  video data exists, abort')
                return False
            logger_print('  video data will be overwritten')
    else:
        output_video_path = None

    ret = process_one_video_internal(input_video_path, input_csv_path,
                                     hog_detector, cnn_detector, predictor,
                                     options)
    if ret != False and \
       options['output_video'] != False and options['compress_video'] != False:
        ret = compress_one_video(output_video_path)

    if ret == False:
        logger_print('Failed to process video file %s\n' % (input_video_path))
    else:
        logger_print('Success to process video file %s\n' % (input_video_path))

    if options['input_csv'] != False:
        os.remove(input_csv_path)

    return ret


def main():
    #input_video_path = '/media'
    input_video_path = '/media/Temp_AIpose20200811'

    # translate to abs path
    input_video_path = os.path.abspath(input_video_path)

    # check data directory
    if check_data_directory() == False:
        logger_print('  fail to check data directory')
        return

    # start the logger
    logger_start()

    # init for all videos
    hog_detector = dlib.get_frontal_face_detector()
    cnn_detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    logger_print('')

    if os.path.isfile(input_video_path) != False:

        options = {
            # update the csv if found
            'output_csv': True,
            'update_csv': True,
            'overwrite_csv': False,

            # do not output video
            'output_video': False,
            'overwrite_video': False,

            # debug options
            'frame_index_max': -1,
            'use_cnn_when_fail': False,
            'compress_video': False,

            # internal use
            'input_csv': False,
        }

        ret = process_one_video(input_video_path, hog_detector, cnn_detector, predictor, options)

    elif os.path.isdir(input_video_path):

        # looking for any video file which name ends with a 'B' character
        prog = re.compile(r'.*B\..+')

        mime = magic.Magic(mime=True)
        for root, dirs, files in os.walk(input_video_path):
            for file in files:
                file_path = os.path.join(root, file)

                file_mine = mime.from_file(file_path)
                if file_mine.find('video') == -1:
                    continue

                if prog.match(file) == None:
                    continue

                options = {
                    # only allows generate a new one if not available
                    'output_csv': True,
                    'update_csv': False,
                    'overwrite_csv': False,

                    # do not output video
                    'output_video': False,
                    'overwrite_video': False,

                    # debug options
                    'frame_index_max': -1,
                    'use_cnn_when_fail': False,
                    'compress_video': False,

                    # internal use
                    'input_csv': False,
                }

                ret = process_one_video(file_path, hog_detector, cnn_detector, predictor, options)

    else:
        logger_print('Fail to process input path %s' % (input_video_path))

    # stop the logger
    logger_stop()

    return

def get_csv_data_file(video_path):
    # translate to abs path
    video_path = os.path.abspath(video_path)

    file_hash = calculate_md5_digest(video_path, 64 * 1024)

    # remove directory part
    _, file_name = os.path.split(video_path)
    # remove ext part
    file_name, _ = os.path.splitext(file_name)

    csv_name = '%s-%s.csv' % (file_name, str(file_hash))
    csv_path = os.path.join(get_data_path('csv'), csv_name)
    csv_path = os.path.abspath(csv_path)

    if os.path.isfile(csv_path) != False:
        # already in data folder
        return csv_path

    options = {
        # only allows generate a new one if not available
        'output_csv': True,
        'update_csv': False,
        'overwrite_csv': False,

        # do not output video
        'output_video': False,
        'overwrite_video': False,

        # debug options
        'frame_index_max': -1,
        'use_cnn_when_fail': False,
        'compress_video': False,
    }

    # check data directory
    if check_data_directory() == False:
        return None

    # start the logger
    logger_start()

    # init for video
    hog_detector = dlib.get_frontal_face_detector()
    cnn_detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    logger_print('')

    ret = process_one_video(video_path, hog_detector, cnn_detector, predictor, options)

    # stop the logger
    logger_stop()

    if ret == False:
        return None

    return csv_path

if __name__ == '__main__':
    main()
