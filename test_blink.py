#!/usr/bin/python3

from facial_video import FacialVideo
from numpy import ndarray
import cv2
import matplotlib.pyplot as plt


# a test program to
#   1. draw face landmarks
#   2. draw a green rectangle on the face if a blink is detected

state = 'open'
close_index = 0

# not used
def test_blink_fixed_threshold(threshold, frame_index, ear):
    length = 2

    global state
    global close_index

    if state == 'open':
        if ear < threshold:
            close_index = frame_index

            state = 'closing'
            print('blink: %d' % (frame_index - close_index + 1))
            return False
    elif state == 'closing':
        if ear < threshold:
            # still closing
            if frame_index >= (close_index + length -1):
                # long enough
                blink_found = True

                state = 'closed'
                print('blink: True')
                return True
            else:
                print('blink: %d' % (frame_index - close_index + 1))
                return False
        else:
            state = 'open'
    elif state == 'closed':
        if ear >= threshold:
            state = 'open'
        else:
            print('blink: %d' % (frame_index - close_index + 1))
            return False

    print('blink: False')
    return False

def test_blink_fixed_delta(buffer, ear):
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

    print('delta %+.6f, blink: %s' % (delta_max, str(ret)), end = end)

    return ret, delta_max

def main():
    input_video_path = '/media/Temp_AIpose20200806/SJCAM/20200806_1AB.mp4'
    output_video_path = './test.mp4'

    draw_plot = False

    start_time = 40.0
    end_time = 70.0

    draw_blink = 0
    blink_count = 0

    times = []
    deltas = []

    fv = FacialVideo(input_video_path)

    start_frame = int(start_time * fv.fps)
    end_frame = int(end_time * fv.fps)

    min_ear, avg_ear, max_ear = fv.calculate_min_avg_max_ear('left', start_frame, end_frame)
    print('ear(left):  min %f, avg %f, max %f' % (min_ear, avg_ear, max_ear))

    #threshold = min_ear * 0.7 + max_ear * 0.3
    #print('threshold %f' % (threshold))

    # 0.1 sec buffering
    #buffer_len = 0.1 * fv.fps
    #ears_buffer = ndarray((buffer_len,), float)
    #print('buffer_len: %d' % (buffer_len))
    ears_buffer = [0.0] * 3

    # always use mp4
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                   fv.fps, (fv.width, fv.height))

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

            print('  frame: %3d, time_stamp: %.3f, ear(left): %.6f, ' % (frame_index, time_stamp, ear_left), end = '')

            #blink = test_blink_fixed_threshold(threshold, frame_index, ear_left)
            blink, delta = test_blink_fixed_delta(ears_buffer, ear_left)

            if draw_blink > 0 and blink != False:
                # this one is false blink
                blink = False
                print(', false blink')

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
                print('')

            if draw_blink > 0:
                cv2.rectangle(frame, rect[0], rect[1], (0, 255, 0), 3)
                draw_blink -= 1
        else:
            print('  frame: %3d, no landmarks' % (frame_index))

        video_writer.write(frame)

    video_writer.release()

    print('Total %d blinks found' % (blink_count))

    if draw_plot != False:
        if len(times) != 0:
            plt.plot(times, deltas, 'r--')
            plt.title('ear delta - time', fontsize = 20)
            plt.xlabel('time', fontsize = 12)
            plt.ylabel('ear delta', fontsize = 12)

            plt.show()

    return

if __name__ == '__main__':
     main()
