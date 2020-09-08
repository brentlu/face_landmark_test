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

skip_next_timestamp = False

def log_print(log_file, string, end = '\n'):
    global skip_next_timestamp

    # print to screen directly
    print(string, end = end)

    if end == '\r':
        end = '\n'

    if skip_next_timestamp != False:
        timestamp = ''
        skip_next_timestamp = False
    else:
        # get current time (local time)
        now = time.localtime()
        timestamp = time.strftime('%Y-%m-%dT%H:%M:%S%z', now)

    log_file.write('%s %s%s' % (timestamp, string, end))

    if end == '':
        skip_next_timestamp = True

    return

state = 'open'
close_index = 0

# not used
def test_blink_fixed_threshold(threshold, frame_index, ear, log_file):
    length = 2

    global state
    global close_index

    if state == 'open':
        if ear < threshold:
            close_index = frame_index

            state = 'closing'
            log_print(log_file, 'blink: %d' % (frame_index - close_index + 1))
            return False
    elif state == 'closing':
        if ear < threshold:
            # still closing
            if frame_index >= (close_index + length -1):
                # long enough
                blink_found = True

                state = 'closed'
                log_print(log_file, 'blink: True')
                return True
            else:
                log_print(log_file, 'blink: %d' % (frame_index - close_index + 1))
                return False
        else:
            state = 'open'
    elif state == 'closed':
        if ear >= threshold:
            state = 'open'
        else:
            log_print(log_file, 'blink: %d' % (frame_index - close_index + 1))
            return False

    log_print(log_file, 'blink: False')
    return False

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

def process_one_video(input_video_path, data_path, start_time = 0.0, end_time = 0.0):
    draw_plot = False

    frame_no_landmarks = 0

    draw_rect = 0
    draw_text = 0
    blink_count = 0

    times = []
    deltas = []

    log_file, timestamp = log_start(data_path, input_video_path)

    log_print(log_file, 'Configuration:')
    log_print(log_file, '  input video path:  %s' % (input_video_path))

    # remove directory part
    _, file_name = os.path.split(input_video_path)

    # remove ext part
    file_name, _ = os.path.splitext(file_name)

    file_name = '%s-%s.mp4' % (file_name, timestamp)
    output_video_path = os.path.join(data_path, file_name)
    output_video_path = os.path.abspath(output_video_path)

    log_print(log_file, '  output video path: %s' % (output_video_path))

    fv = FacialVideo(input_video_path)

    start_frame = int(start_time * fv.fps)
    if end_time == 0.0:
        end_frame = fv.frame_count
    else:
        end_frame = int(end_time * fv.fps)

    ret = fv.update_static_data(start_frame, end_frame)

    if ret != False:
        log_print(log_file, '  eye aspect ratio(left):  min %.3f, avg %.3f, max %.3f' % (fv.min_ear[0], fv.avg_ear[0], fv.max_ear[0]))
        log_print(log_file, '  eye aspect ratio(right): min %.3f, avg %.3f, max %.3f' % (fv.min_ear[1], fv.avg_ear[1], fv.max_ear[1]))
        log_print(log_file, '  eye width(left):         min %.3f, avg %.3f, max %.3f' % (fv.min_ew[0], fv.avg_ew[0], fv.max_ew[0]))
        log_print(log_file, '  eye width(right):        min %.3f, avg %.3f, max %.3f' % (fv.min_ew[1], fv.avg_ew[1], fv.max_ew[1]))

    #threshold = min_ear * 0.7 + max_ear * 0.3
    #print('  fixed threshold: %f' % (threshold))

    # 0.1 sec buffering
    #buffer_len = 0.1 * fv.fps
    #ears_buffer = ndarray((buffer_len,), float)
    #print('buffer_len: %d' % (buffer_len))
    ears_buffer = [0.0] * 3

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

    log_print(log_file, 'Process video:')

    while True:
        ret, frame = fv.read()
        if ret == False:
            # no frame to process
            break;

        frame_index = fv.get_frame_index()
        if frame_index < start_frame:
            continue
        elif frame_index >= end_frame:
            break

        if fv.available() != False:
            time_stamp = fv.get_time_stamp()
            landmarks = fv.get_landmarks()
            rect = fv.get_rect()

            ear_left = fv.get_eye_aspect_ratio('left')
            ear_right = fv.get_eye_aspect_ratio('right')
            ew_left = fv.get_eye_width('left')
            ew_right = fv.get_eye_width('right')

            log_print(log_file, '  frame: %3d, time: %.3f, ear: %.3f %.3f, width: %3.2f%% %3.2f%% ' % (frame_index, time_stamp, ear_left, ear_right, ew_left * 100.0 / fv.max_ew[0], ew_right * 100.0 / fv.max_ew[1]), end = '')

            #blink = test_blink_fixed_threshold(threshold, frame_index, ear_left, log_file)
            blink, delta = test_blink_fixed_delta(ears_buffer, ear_left)

            if blink == False:
                log_print(log_file, 'delta %+.3f' % (delta))
            elif draw_rect > 0:
                # this one is false blink
                blink = False
                log_print(log_file, 'delta %+.3f, blink found, false blink)' % (delta))

            # save data for plot
            if draw_plot != False:
                times.append(time_stamp)
                deltas.append(delta)

            # draw landmarks
            for n in range(0, 68):
                cv2.circle(frame, landmarks[n], 6, (255, 0, 0), -1)

            # draw rect if blink found
            if blink != False:
                draw_rect = 3 # draw the rect for three frames
                draw_text = 6 # draw the text for six frames
                blink_count += 1
                log_print(log_file, 'delta %+.3f, blink found, count %d' % (delta, blink_count))

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

        crop = frame[p1[1]:p2[1], p1[0]:p2[0]]
        video_writer.write(crop)

    video_writer.release()

    log_print(log_file, 'Statistic:')
    log_print(log_file, '  total %d frames processed' % (end_frame - start_frame))
    log_print(log_file, '  %d frames (%3.2f%%) has no landmarks' % (frame_no_landmarks, (frame_no_landmarks * 100.0) / (end_frame - start_frame)))
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
    print('process_training_csv: %s' % (filename))

    with open(csv_path, 'r', newline = '') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            video_path = row['file_name']
            start_time = float(row['start_time'])
            end_time = float(row['end_time'])

            ret = process_one_video(video_path, data_path, start_time, end_time)
            if ret == False:
                return False

    print('  success')
    return True

def main():
    start_time = 0.0
    end_time = 0.0


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
    parser.add_argument('path', help = 'path to a video file, a directory, or a training recipe file')

    parser.add_argument('-s', '--start_time', help = 'start time (sec)')
    parser.add_argument('-e', '--end_time', help = 'end time (sec)')


    args = parser.parse_args()

    input_video_path = args.path

    print('User input:')
    print('  input path: %s' % (input_video_path))

    if args.start_time != None:
        start_time = float(args.start_time)
        print('  start time: %.3f' % (start_time))

    if args.end_time != None:
        end_time = float(args.end_time)
        print('  end time:   %.3f' % (end_time))

    _, ext = os.path.splitext(input_video_path)

    if ext == '.csv':
        if args.start_time != None:
            print('  ignore start time')
        if args.end_time != None:
            print('  ignore end time')

        # could be a training recipe
        ret = process_training_csv(input_video_path, data_path)

    elif os.path.isfile(input_video_path) != False:
        mime = magic.Magic(mime=True)

        file_mine = mime.from_file(input_video_path)
        if file_mine.find('video') == -1:
            print('  not a video file')
            return False

        ret = process_one_video(input_video_path, data_path, start_time, end_time)

    elif os.path.isdir(input_video_path):
        if args.start_time != None:
            print('  ignore start time')
        if args.end_time != None:
            print('  ignore end time')

        # looking for any video file which name ends with a 'A' or 'B' character
        prog = re.compile(r'.*[AB]\..+')

        mime = magic.Magic(mime=True)
        for root, dirs, files in os.walk(input_video_path):
            for file in files:
                file_path = os.path.join(root, file)

                file_mine = mime.from_file(file_path)
                if file_mine.find('video') == -1:
                    continue

                if prog.match(file) == None:
                    continue

                ret = process_one_video(input_video_path, data_path)

    else:
        print('Unrecognized path')

    return

if __name__ == '__main__':
     main()
