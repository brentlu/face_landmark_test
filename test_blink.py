#!/usr/bin/python3

from scipy.spatial import distance as dist
import csv
import cv2
import dlib
import matplotlib.pyplot as plt
import os
import process_video as pv

# a test program to
#   1. draw left eye
#   2. draw a green rectangle on the face if a blink is detected


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

state = 'open'
close_index = 0

def test_blink(frame_index, ear_value):
    threshold = 0.3
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

def main():
    input_video_path = './20200429_2B.mp4'
    output_video_path = './test.mp4'

    csv_index = 0
    frame_index = 0
 
    times = []
    ears = []

    csv_path = pv.get_csv_data_file(input_video_path)

    if csv_path == None:
        print('Fail to get cvs data for %s\n' % (input_video_path))
        return

    if os.path.isfile(csv_path) == False:
        # should not get here
        return

    # open the video file
    cap = cv2.VideoCapture(input_video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    hog_detector = dlib.get_frontal_face_detector()
    rotation = pv.auto_detect_rotation(input_video_path, hog_detector)

    if rotation == cv2.ROTATE_90_CLOCKWISE or rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
        temp = width
        width = height
        height = temp

    # open the csv file
    csvfile = open(csv_path, 'r', newline = '')
    csv_reader = csv.DictReader(csvfile)

    # read the first row
    try:
        row = next(csv_reader)
        csv_index = int(row['index'])
    except StopIteration:
        csv_index = frame_count + 1

    # always use mp4
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, (width, height))

    while True:
        ret, frame = cap.read()
        if ret == False:
            # no frame to process
            break;

        frame_index += 1

        if rotation != -1:
            frame = cv2.rotate(frame, rotation)

        if frame_index == csv_index:
            # this frame has csv data
            landmarks = []

            for n in range(0, 68):
                x = int(row['mark_%d_x' % (n)])
                y = int(row['mark_%d_y' % (n)])

                landmarks.append((x, y))

            time_stamp = float(row['time_stamp'])

            ear_left = calculate_ear_value(landmarks, 'left')
            ear_right = calculate_ear_value(landmarks, 'right')

            blink = test_blink(frame_index, ear_left)

            print('  frame: %3d, time_stamp: %4.3f, ear: (%4.3f,%4.3f), blink: %s' % (int(row['index']), time_stamp, ear_left, ear_right, str(blink)))

            times.append(time_stamp)
            ears.append(ear_left)

            # draw left eye
            for n in range(0, 68):
                if n >= 42 or n < 48:
                    cv2.circle(frame, landmarks[n], 6, (255, 0, 0), -1)

            # draw rect if blink found
            if blink != False:
                (x1, y1) = (int(row['target_left']), int(row['target_top']))
                (x2, y2) = (int(row['target_right']), int(row['target_bottom']))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # read the next row
            try:
                row = next(csv_reader)
                csv_index = int(row['index'])
            except StopIteration:
                csv_index = frame_count + 1

        video_writer.write(frame)

    if len(times) != 0:
        plt.plot(times, ears, 'r--')
        plt.title('EAR-time', fontsize = 20)
        plt.xlabel('time', fontsize = 12)
        plt.ylabel('EAR', fontsize = 12)
        plt.show()

    video_writer.release()
    csvfile.close()
    cap.release()

    return

if __name__ == '__main__':
     main()
