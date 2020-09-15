#!/usr/bin/python3

from facial_video import FacialVideo
from tempfile import NamedTemporaryFile
import argparse
import csv
import cv2
import magic
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import time


def log_start(data_path, input_video_path):

    # remove directory part
    _, file_name = os.path.split(input_video_path)

    # remove ext part
    file_name, _ = os.path.splitext(file_name)

    while True:
        # get current time (local time)
        now = time.localtime()

        timestamp = time.strftime('%Y-%m%d-%H%M', now)

        file_name = '%s-%s.log' % (file_name, timestamp)
        log_path = os.path.join(data_path, file_name)
        log_path = os.path.abspath(log_path)

        if os.path.exists(log_path) == False:
            break

    log_file = open(log_path, 'w')

    return log_file, timestamp

def log_stop(log_file):
    log_file.close()

    return

def log_print(log_file, string, end = '\n'):
    # print to screen directly
    print(string, end = end)

    if end == '\r':
        end = '\n'

    log_file.write('%s%s' % (string, end))

    return

def process_one_video(input_video_path, data_path, start_frame, end_frame):
    frame_index = 0
    frame_no_landmarks = 0

    log_file, timestamp = log_start(data_path, input_video_path)

    log_print(log_file, 'Process video:')
    log_print(log_file, '  input video path: %s' % (input_video_path))
    log_print(log_file, '  start frame: %d' % (start_frame))
    log_print(log_file, '  end frame: %d' % (end_frame))

    # remove directory part
    _, file_name = os.path.split(input_video_path)

    # remove ext part
    file_name, _ = os.path.splitext(file_name)

    file_name = '%s-%s.mp4' % (file_name, timestamp)
    output_video_path = os.path.join(data_path, file_name)
    output_video_path = os.path.abspath(output_video_path)

    log_print(log_file, '  output video path: %s' % (output_video_path))

    fv = FacialVideo(input_video_path)

    if fv.init() == False:
        print('  fail to init engine')
        return False, 0.0

    ret = fv.update_statistic_data(start_frame, end_frame)

    print('Statistic data:')

    if ret == False:
        print('  fail')
        return False, 0.0

    em_min = fv.get_eye_to_mouth_length(fv.MIN)
    em_avg = fv.get_eye_to_mouth_length(fv.AVG)
    em_max = fv.get_eye_to_mouth_length(fv.MAX)

    print('  eye to mouth(left):      min %.3f, avg %.3f, max %.3f' % (em_min[fv.LEFT_EYE], em_avg[fv.LEFT_EYE], em_max[fv.LEFT_EYE]))
    print('  eye to mouth(right):     min %.3f, avg %.3f, max %.3f' % (em_min[fv.RIGHT_EYE], em_avg[fv.RIGHT_EYE], em_max[fv.RIGHT_EYE]))

    # calculate the output video dimensions
    face_rect = fv.find_face_rect(start_frame, end_frame)
    p1 = face_rect[0] # left, top
    p2 = face_rect[1] # right, bottom

    log_print(log_file, '  face: %s, %s' % (str(p1), str(p2)))

    width = p2[0] - p1[0]
    height = p2[1] - p1[1]

    # always use mp4
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                   fv.fps, (width, height))

    log_print(log_file, 'Process frame:')

    em_prev = np.zeros((2, 1), dtype = float)
    delta_total = np.zeros((2, 1), dtype = float)

    while True:
        frame_index += 1

        if frame_index < start_frame:
            # don't decode this frame to speed up
            ret, _ = fv.read(True)
            continue
        elif frame_index > end_frame:
            break
        else:
            # decode this frame
            ret, frame = fv.read()

        if ret == False:
            # no frame to process
            break;

        if frame_index != fv.get_frame_index():
            print('  expect frame %d but got %d' %(frame_index, fv.get_frame_index()))
            break

        if fv.available() != False:
            time_stamp = (frame_index - start_frame) / fv.fps
            landmarks = fv.get_landmarks()
            rect = fv.get_rect()

            em = np.array(fv.calculate_eye_to_mouth_length())

            if em_prev[0] != 0.0 or em_prev[1] != 0.0:
                delta = em - em_prev
            else:
                delta = np.zeros((2, 1), dtype = float)

            em_prev = em

            #length_left = em[fv.LEFT_EYE] * 100.0 / fv.eye_to_mouth[fv.LEFT_EYE][fv.MAX]
            #length_right = em[fv.RIGHT_EYE] * 100.0 / fv.eye_to_mouth[fv.RIGHT_EYE][fv.MAX]

            log_print(log_file, '  frame: %3d, time: %.3f, length: %.3f %.3f, delta %+.3f %+.3f' % (frame_index, time_stamp, em[fv.LEFT_EYE], em[fv.RIGHT_EYE], delta[0], delta[1]), end = '')
            log_print(log_file, '')

            delta_total[0] = delta_total[0] + abs(delta[0])
            delta_total[1] = delta_total[1] + abs(delta[1])

        else:
            log_print(log_file, '  frame: %3d, no landmarks' % (frame_index))
            frame_no_landmarks += 1

        crop = frame[p1[1]:p2[1], p1[0]:p2[0]]
        video_writer.write(crop)

    video_writer.release()

    log_print(log_file, 'Statistic:')
    log_print(log_file, '  total %d frames processed' % (end_frame - start_frame + 1))
    log_print(log_file, '  %d frames (%3.2f%%) has no landmarks' % (frame_no_landmarks, (frame_no_landmarks * 100.0) / (end_frame - start_frame + 1)))
    log_print(log_file, '  total delta %.3f %.3f' % (delta_total[0], delta_total[1]))
    log_print(log_file, 'Process complete')

    log_stop(log_file)

    return True, (delta_total[0] / em_max[fv.LEFT_EYE]) + (delta_total[1] / em_max[fv.RIGHT_EYE])

def process_training_csv(csv_path, data_path):
    csv_fields = ['blink', 'm2e', 'date', 'pid', 'type', 'start_frame', 'end_frame', 'duration', 'width_diff', 'data_blink', 'data_m2e', 'pd_stage']

    _, filename = os.path.split(csv_path)
    print('Process training csv: %s' % (filename))

    # update the record in csv file if found
    temp_file = NamedTemporaryFile(mode = 'w', delete = False)

    with open(csv_path, 'r') as csv_file, temp_file:
        csv_reader = csv.DictReader(csv_file)
        csv_writer = csv.DictWriter(temp_file, fieldnames = csv_fields)
        csv_writer.writeheader()

        for row in csv_reader:
            if row['m2e'] != 'yes':
                # copy the rows
                csv_writer.writerow(row)
                continue

            start_frame = int(row['start_frame'])
            if start_frame == 0:
                # copy the rows
                csv_writer.writerow(row)
                continue

            video_path = '/media/Temp_AIpose%s/SJCAM/%s_%s%s.mp4' % (row['date'], row['date'], row['pid'], row['type'])
            end_frame = int(row['end_frame'])

            ret, delta = process_one_video(video_path, data_path, start_frame, end_frame)
            if ret == False:
                print('  fail')
                return False

            row['data_m2e'] = '%.3f' % (delta)

            # copy the rows
            csv_writer.writerow(row)

    shutil.move(temp_file.name, csv_path)

    print('  success')
    return True

def main():
    start_time = 0.0
    duration = 0.0

    # check data directory first
    data_path = os.path.abspath('./m2e_data')
    if os.path.isdir(data_path) == False:
        try:
            print('create data directory')
            os.mkdir(data_path)
        except OSError:
            print('fail to create data directory')
            return False

    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help = 'path to a video file or a training recipe file')

    parser.add_argument('-s', '--start_time', help = 'start time (sec)')
    parser.add_argument('-d', '--duration', help = 'duration (sec)')

    args = parser.parse_args()

    input_path = args.input_path

    if args.start_time != None:
        start_time = float(args.start_time)

    if args.duration != None:
        duration = float(args.duration)

    print('User input:')
    print('  input path: %s' % (input_path))
    print('  start time: %.3f' % (start_time))
    print('  duration:   %.3f' % (duration))

    _, ext = os.path.splitext(input_path)

    if ext == '.csv':
        if args.start_time != None:
            print('  ignore start time')
        if args.duration != None:
            print('  ignore duration')

        # could be a training recipe
        ret = process_training_csv(input_path, data_path)

    elif os.path.isfile(input_path) != False:
        mime = magic.Magic(mime=True)

        file_mine = mime.from_file(input_path)
        if file_mine.find('video') == -1:
            print('  not a video file')
            return

        fv = FacialVideo(input_path)

        if fv.init() == False:
            print('  fail to init engine')
            return

        start_frame = int(start_time * fv.fps)
        if duration == 0.0:
            end_frame = fv.frame_count
        else:
            end_frame = start_frame + int(duration * fv.fps) - 1

        ret, _ = process_one_video(input_path, data_path, start_frame, end_frame)

    else:
        print('Unrecognized path')

    return

if __name__ == '__main__':
    main()
