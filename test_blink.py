#!/usr/bin/python3

from facial_video import FacialVideo
from numpy import ndarray
import cv2
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

    return log_file

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

def test_blink_fixed_delta(buffer, ear, log_file):
    delta = 0.0
    delta_max = 1.0
    ret = False
    end = '\n'

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

    if ret != False:
        end = ''

    log_print(log_file, 'delta %+.6f, blink: %s' % (delta_max, str(ret)), end = end)

    return ret, delta_max

def process_one_video(input_video_path, data_path, start_time, end_time):
    draw_plot = False

    draw_blink = 0
    blink_count = 0

    times = []
    deltas = []

    log_file = log_start(data_path, input_video_path)

    log_print(log_file, 'Configuration:')
    log_print(log_file, '  input video path : %s' % (input_video_path))

    # remove directory part
    _, file_name = os.path.split(input_video_path)

    # remove ext part
    file_name, _ = os.path.splitext(file_name)

    file_name = '%s-blink.mp4' % (file_name)
    output_video_path = os.path.join(data_path, file_name)
    output_video_path = os.path.abspath(output_video_path)

    log_print(log_file, '  output video path: %s' % (output_video_path))

    fv = FacialVideo(input_video_path)

    start_frame = int(start_time * fv.fps)
    end_frame = int(end_time * fv.fps)

    min_ear, avg_ear, max_ear = fv.calculate_min_avg_max_ear('left', start_frame, end_frame)

    log_print(log_file, '  ear(left):  min %f, avg %f, max %f' % (min_ear, avg_ear, max_ear))

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

            ear_left = fv.get_ear_value('left')

            log_print(log_file, '  frame: %3d, time_stamp: %.3f, ear(left): %.6f, ' % (frame_index, time_stamp, ear_left), end = '')

            #blink = test_blink_fixed_threshold(threshold, frame_index, ear_left, log_file)
            blink, delta = test_blink_fixed_delta(ears_buffer, ear_left, log_file)

            if draw_blink > 0 and blink != False:
                # this one is false blink
                blink = False
                log_print(log_file, ', false blink')

            # save data for plot
            if draw_plot != False:
                times.append(time_stamp)
                deltas.append(delta)

            # draw landmarks
            for n in range(0, 68):
                cv2.circle(frame, landmarks[n], 6, (255, 0, 0), -1)

            # draw rect if blink found
            if blink != False:
                draw_blink = 3 # draw three frames
                blink_count += 1
                log_print(log_file, '')

            if draw_blink > 0:
                cv2.rectangle(frame, rect[0], rect[1], (0, 255, 0), 3)
                draw_blink -= 1
        else:
            log_print(log_file, '  frame: %3d, no landmarks' % (frame_index))

        crop = frame[p1[1]:p2[1], p1[0]:p2[0]]
        video_writer.write(crop)

    video_writer.release()

    log_print(log_file, 'Total %d blinks found' % (blink_count))
    log_print(log_file, 'Process complete')

    if draw_plot != False:
        if len(times) != 0:
            plt.plot(times, deltas, 'r--')
            plt.title('ear delta - time', fontsize = 20)
            plt.xlabel('time', fontsize = 12)
            plt.ylabel('ear delta', fontsize = 12)

            plt.show()

    log_stop(log_file)

    return

def main():
    input_video_path = '/media/Temp_AIpose20200806/SJCAM/20200806_1AB.mp4'

    data_path = os.path.abspath('./blink_data')
    if os.path.isdir(data_path) == False:
        # create data directory
        try:
            print('create data directory')
            os.mkdir(data_path)
        except OSError:
            print('fail to create data directory')
            return False

    start_time = 40.0
    end_time = 70.0

    process_one_video(input_video_path, data_path, start_time, end_time)

    return

if __name__ == '__main__':
     main()
