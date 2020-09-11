#!/usr/bin/python3

from facial_video import FacialVideo
from numpy import ndarray
import argparse
import csv
import cv2
import magic
import matplotlib.pyplot as plt
import os
import time


# a test program to
#   1. draw face landmarks
#   2. draw a green rectangle on the face if a blink is detected

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

def test_blink_fixed_delta(buffer, ear):
    delta = 0.0
    delta_max = 1.0
    ret = False

    buffer_num = len(buffer)

    for n in range(0, buffer_num):
        delta = ear - buffer[n]
        if delta < -0.05:
            ret = True

        if delta < delta_max:
            delta_max = delta

    if ret != True:
        for n in range(1, buffer_num):
            buffer[n - 1] = buffer[n]

        buffer[buffer_num - 1] = ear
    else:
        for n in range(0, buffer_num):
            buffer[n] = 0.0

    return ret, delta_max

def process_one_video(input_video_path, data_path, start_frame, end_frame):
    draw_plot = False

    frame_index = 0
    frame_no_landmarks = 0

    draw_rect = 0
    draw_text = 0
    blink_count = 0

    blink_window = [0, 0]

    times = []
    deltas = []

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

    ret = fv.update_statistic_data(start_frame, end_frame)

    print('Statistic data:')

    if ret == False:
        print('  fail')
        return False

    ear_min = fv.get_eye_aspect_ratio(fv.MIN)
    ear_avg = fv.get_eye_aspect_ratio(fv.AVG)
    ear_max = fv.get_eye_aspect_ratio(fv.MAX)

    ew_min = fv.get_eye_width(fv.MIN)
    ew_avg = fv.get_eye_width(fv.AVG)
    ew_max = fv.get_eye_width(fv.MAX)

    print('  eye aspect ratio(left):  min %.3f, avg %.3f, max %.3f' % (ear_min[fv.LEFT_EYE], ear_avg[fv.LEFT_EYE], ear_max[fv.LEFT_EYE]))
    print('  eye aspect ratio(right): min %.3f, avg %.3f, max %.3f' % (ear_min[fv.RIGHT_EYE], ear_avg[fv.RIGHT_EYE], ear_max[fv.RIGHT_EYE]))
    print('  eye width(left):         min %.3f, avg %.3f, max %.3f' % (ew_min[fv.LEFT_EYE], ew_avg[fv.LEFT_EYE], ew_max[fv.LEFT_EYE]))
    print('  eye width(right):        min %.3f, avg %.3f, max %.3f' % (ew_min[fv.RIGHT_EYE], ew_avg[fv.RIGHT_EYE], ew_max[fv.RIGHT_EYE]))

    # 0.1 sec buffering
    #buffer_len = 0.1 * fv.fps
    #ears_buffer = ndarray((buffer_len,), float)
    #print('buffer_len: %d' % (buffer_len))
    buffer_left = [0.0] * 3
    buffer_right = [0.0] * 3

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

            ear = [0.0, 0.0]
            ew = [0.0, 0.0]

            blink = [False, False]
            delta = [0.0, 0.0]
            blink_overlap = False

            ear = fv.calculate_eye_aspect_ratio()
            ew = fv.calculate_eye_width()

            #blink = test_blink_fixed_threshold(threshold, frame_index, ear_left, log_file)
            blink[0], delta[0] = test_blink_fixed_delta(buffer_left, ear[0])
            blink[1], delta[1] = test_blink_fixed_delta(buffer_right, ear[1])

            log_print(log_file, '  frame: %3d, time: %.3f, ear: %.3f %.3f, width: %3.2f%% %3.2f%%, delta %+.3f %+.3f' % (frame_index, time_stamp, ear[fv.LEFT_EYE], ear[fv.RIGHT_EYE], ew[fv.LEFT_EYE] * 100.0 / fv.eye_width[fv.LEFT_EYE][fv.MAX], ew[fv.RIGHT_EYE] * 100.0 / fv.eye_width[fv.RIGHT_EYE][fv.MAX], delta[0], delta[1]), end = '')

            if blink[0] != False:
                blink_window[0] = 2
            if blink[1] != False:
                blink_window[1] = 2

            if blink_window[0] != 0 and blink_window[1] != 0:
                # reset
                blink_window[0] = 0
                blink_window[1] = 0

                blink_overlap = True

            if blink_window[0] != 0:
                blink_window[0] -= 1
            if blink_window[1] != 0:
                blink_window[1] -= 1

            if blink_overlap == False:
                log_print(log_file, '')
            elif draw_rect > 0:
                # this one is false blink
                blink_overlap = False
                log_print(log_file, ', false blink')

            # save data for plot
            if draw_plot != False:
                times.append(time_stamp)
                deltas.append(delta[0])

            # draw landmarks
            for n in range(0, 68):
                cv2.circle(frame, landmarks[n], 6, (255, 0, 0), -1)

            # draw rect if blink found
            if blink_overlap != False:
                draw_rect = 3 # draw the rect for three frames
                draw_text = 6 # draw the text for six frames
                blink_count += 1
                log_print(log_file, ', blink found, count %d' % (blink_count))

            if draw_rect > 0:
                cv2.rectangle(frame, rect[0], rect[1], (0, 255, 0), 3)
                draw_rect -= 1

            if draw_text > 0:
                text = 'Blink: %d' % (blink_count)
                cv2.putText(frame, str(text), (p1[0] + 10, p1[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (255, 0, 0), 2, cv2.LINE_AA, False);
                draw_text -= 1

        else:
            log_print(log_file, '  frame: %3d, no landmarks' % (frame_index))
            frame_no_landmarks += 1

            buffer_num = len(buffer_left)

            for n in range(1, buffer_num):
                buffer_left[n - 1] = buffer_left[n]

            buffer_left[buffer_num - 1] = 0.0

            buffer_num = len(buffer_right)

            for n in range(1, buffer_num):
                buffer_right[n - 1] = buffer_right[n]

            buffer_right[buffer_num - 1] = 0.0

            if blink_window[0] != 0:
                blink_window[0] -= 1
            if blink_window[1] != 0:
                blink_window[1] -= 1
            if draw_rect > 0:
                draw_rect -= 1
            if draw_text > 0:
                draw_text -= 1

        crop = frame[p1[1]:p2[1], p1[0]:p2[0]]
        video_writer.write(crop)

    video_writer.release()

    log_print(log_file, 'Statistic:')
    log_print(log_file, '  total %d frames processed' % (end_frame - start_frame + 1))
    log_print(log_file, '  %d frames (%3.2f%%) has no landmarks' % (frame_no_landmarks, (frame_no_landmarks * 100.0) / (end_frame - start_frame + 1)))
    log_print(log_file, '  %d blinks found' % (blink_count))
    log_print(log_file, 'Process complete')

    if draw_plot != False:
        if len(times) != 0:
            plt.plot(times, deltas, 'r--')
            plt.title('ear delta - time', fontsize = 20)
            plt.xlabel('time', fontsize = 12)
            plt.ylabel('ear delta', fontsize = 12)

            plt.show()

    log_stop(log_file)

    return True

def process_training_csv(csv_path, data_path):

    _, filename = os.path.split(csv_path)
    print('Process training csv: %s' % (filename))

    with open(csv_path, 'r', newline = '') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            video_path = row['file_name']
            start_frame = int(row['start_frame'])
            end_frame = int(row['end_frame'])

            ret = process_one_video(video_path, data_path, start_frame, end_frame)
            if ret == False:
                print('  fail')
                return False

    print('  success')
    return True

def main():
    start_time = 0.0
    duration = 0.0

    # check data directory first
    data_path = os.path.abspath('./blink_data')
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
            return False

        fv = FacialVideo(input_path)

        start_frame = int(start_time * fv.fps)
        if duration == 0.0:
            end_frame = fv.frame_count
        else:
            end_frame = start_frame + int(duration * fv.fps) - 1

        ret = process_one_video(input_path, data_path, start_frame, duration)

    else:
        print('Unrecognized path')

    return

if __name__ == '__main__':
     main()
