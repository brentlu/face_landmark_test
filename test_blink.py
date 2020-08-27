#!/usr/bin/python3

from scipy.spatial import distance as dist
import csv
import cv2
import matplotlib.pyplot as plt
#import os
import process_video as pv

from facial_video import FacialVideo

# a test program to
#   1. draw face landmarks
#   2. draw a green rectangle on the face if a blink is detected



state = 'open'
close_index = 0

def test_blink(min_ear, avg_ear, frame_index, ear_value):
    threshold = (min_ear + avg_ear) * 0.4
    length = 3 # 0.1 sec
    blink_found = False

    global state
    global close_index

    if state == 'open':
        if ear_value < threshold:
            state = 'closing'
            close_index = frame_index
    elif state == 'closing':
        if ear_value < threshold:
            # still closing
            if frame_index >= (close_index + length -1):
                # long enough
                blink_found = True

                state = 'closed'
        else:
            state = 'open'
    elif state == 'closed':
        if ear_value >= threshold:
            state = 'open'

    return blink_found

def calculate_ear_value(landmarks, eye):
    if len(landmarks) != 68:
        print('incomplete landmark')

    if eye == 'left':
        # euclidean distances between the two sets of vertical eye landmarks
        A = dist.euclidean(landmarks[43], landmarks[47])
        B = dist.euclidean(landmarks[44], landmarks[46])
        # euclidean distance between the horizontal eye landmark
        C = dist.euclidean(landmarks[42], landmarks[45])
    elif eye == 'right':
        # euclidean distances between the two sets of vertical eye landmarks
        A = dist.euclidean(landmarks[38], landmarks[40])
        B = dist.euclidean(landmarks[37], landmarks[41])
        # euclidean distance between the horizontal eye landmark
        C = dist.euclidean(landmarks[39], landmarks[36])
    else:
        print('calculate_ear_value: unknown eye %s' % (eye))

    ear = (A + B) / (2.0 * C)

    return ear

def main():
    input_video_path = './20200429_2B.mp4'
    output_video_path = './test2.mp4'

    times = []
    ears = []

    fv = FacialVideo(input_video_path)

    min_ear, avg_ear = fv.calculate_min_avg_ear('left')
    print('min ear %f, avg ear %f' % (min_ear, avg_ear))

    # always use mp4
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                   fv.fps, (fv.width, fv.height))

    while True:
        ret, frame = fv.read()
        if ret == False:
            # no frame to process
            break;

        if fv.available() != False:
            time_stamp = fv.get_time_stamp()
            landmarks = fv.get_landmarks()
            rect = fv.get_rect()

            ear_left = calculate_ear_value(landmarks, 'left')
            ear_right = calculate_ear_value(landmarks, 'right')

            blink = test_blink(min_ear, avg_ear, fv.get_frame_index(), ear_left)

            print('  frame: %3d, time_stamp: %4.3f, ear: (%4.3f,%4.3f), blink: %s' % (fv.get_frame_index(), time_stamp, ear_left, ear_right, str(blink)))

            times.append(time_stamp)
            ears.append(ear_left)

            # draw landmarks
            for n in range(0, 68):
                #if n >= 42 or n < 48:
                cv2.circle(frame, landmarks[n], 6, (255, 0, 0), -1)

            # draw rect if blink found
            if blink != False:
                cv2.rectangle(frame, rect[0], rect[1], (0, 255, 0), 3)

        video_writer.write(frame)

    video_writer.release()

    if len(times) != 0:
        plt.plot(times, ears, 'r--')
        plt.title('EAR-time', fontsize = 20)
        plt.xlabel('time', fontsize = 12)
        plt.ylabel('EAR', fontsize = 12)
        plt.show()

    return

if __name__ == '__main__':
     main()
