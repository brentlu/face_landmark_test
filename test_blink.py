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

class FacialVideo:
    def __init__(self, video_path):
        # open the video file
        self.cap = cv2.VideoCapture(video_path)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.hog_detector = dlib.get_frontal_face_detector()
        self.rotation = pv.auto_detect_rotation(video_path, self.hog_detector)

        if self.rotation == cv2.ROTATE_90_CLOCKWISE or self.rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
            temp = self.width
            self.width = self.height
            self.height = temp

        self.frame_index = 0

        csv_path = pv.get_csv_data_file(video_path)

        if csv_path != None:
            if os.path.isfile(csv_path) == False:
                # should not get here
                return

            # open the csv file
            self.csvfile = open(csv_path, 'r', newline = '')
            self.csv_reader = csv.DictReader(self.csvfile)

            # read the first row
            try:
                self.__csv_row = next(self.csv_reader)
                self.__csv_index = int(self.__csv_row['index'])
            except StopIteration:
                self.__csv_index = self.frame_count + 1

    def __del__(self):
        self.csvfile.close()
        self.cap.release()

    def read(self):
        ret, frame = self.cap.read()
        if ret == False:
            # no frame left in the video
            return ret, frame

        self.frame_index += 1

        if self.rotation != -1:
            frame = cv2.rotate(frame, self.rotation)

        if self.frame_index == self.__csv_index:
            # this frame has csv data
            self.time_stamp = float(self.__csv_row['time_stamp'])

            self.landmarks = []

            for n in range(0, 68):
                x = int(self.__csv_row['mark_%d_x' % (n)])
                y = int(self.__csv_row['mark_%d_y' % (n)])

                self.landmarks.append((x, y))

            self.rect = []

            self.rect.append((int(self.__csv_row['target_left']), int(self.__csv_row['target_top'])))
            self.rect.append((int(self.__csv_row['target_right']), int(self.__csv_row['target_bottom'])))

            # read the next row
            try:
                self.__csv_row = next(self.csv_reader)
                self.__csv_index = int(self.__csv_row['index'])
            except StopIteration:
                self.__csv_index = self.frame_count + 1

        return ret, frame

    def available(self):
        if len(self.landmarks) != 68:
            return False

        if len(self.rect) != 2:
            return False

        return True

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
    output_video_path = './test2.mp4'

    times = []
    ears = []

    fv = FacialVideo(input_video_path)

    # always use mp4
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                   fv.fps, (fv.width, fv.height))

    while True:
        ret, frame = fv.read()
        if ret == False:
            # no frame to process
            break;

        if fv.available() != False:
            ear_left = calculate_ear_value(fv.landmarks, 'left')
            ear_right = calculate_ear_value(fv.landmarks, 'right')

            blink = test_blink(fv.frame_index, ear_left)

            print('  frame: %3d, time_stamp: %4.3f, ear: (%4.3f,%4.3f), blink: %s' % (fv.frame_index, fv.time_stamp, ear_left, ear_right, str(blink)))

            times.append(fv.time_stamp)
            ears.append(ear_left)

            # draw left eye
            for n in range(0, 68):
                if n >= 42 or n < 48:
                    cv2.circle(frame, fv.landmarks[n], 6, (255, 0, 0), -1)

            # draw rect if blink found
            if blink != False:
                cv2.rectangle(frame, fv.rect[0], fv.rect[1], (0, 255, 0), 3)

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
