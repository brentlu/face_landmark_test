#!/usr/bin/python3

from scipy.spatial import distance as dist
import csv
import cv2
import dlib
import os

#import process_video as pv
from process_video import FacialEngine


class FacialVideo:
    def __init__(self, video_path):
        # open the video file
        self.cap = cv2.VideoCapture(video_path)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.__engine = FacialEngine(video_path)
        self.rotation = self.__engine.auto_detect_rotation()

        if self.rotation == cv2.ROTATE_90_CLOCKWISE or self.rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
            temp = self.width
            self.width = self.height
            self.height = temp

        self.__frame_index = 0

        self.csv_path = self.__engine.get_csv_data_file()

        if self.csv_path != None:
            if os.path.isfile(self.csv_path) == False:
                # should not get here
                return

            # open the csv file
            self.csvfile = open(self.csv_path, 'r', newline = '')
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

    def calculate_min_avg_ear(self, eye):
        total_ear = 0.0
        total_frame = 0

        ear = 0.0
        min_ear = 1

        # open the csv file
        csvfile = open(self.csv_path, 'r', newline = '')
        csv_reader = csv.DictReader(csvfile)

        for csv_row in csv_reader:
            landmarks = []

            for n in range(0, 68):
                x = int(csv_row['mark_%d_x' % (n)])
                y = int(csv_row['mark_%d_y' % (n)])

                landmarks.append((x, y))

            ear = calculate_ear_value(landmarks, eye)
            if ear < min_ear:
                min_ear = ear

            total_ear += ear
            total_frame += 1

        csvfile.close()

        return min_ear, total_ear / total_frame

    def read(self):
        ret, frame = self.cap.read()
        if ret == False:
            # no frame left in the video
            return ret, frame

        self.__frame_index += 1

        if self.rotation != -1:
            frame = cv2.rotate(frame, self.rotation)

        self.__time_stamp = 0.0
        self.__landmarks = []
        self.__rect = []

        # this frame has csv data
        if self.__frame_index == self.__csv_index:
            # update timestamp
            self.__time_stamp = float(self.__csv_row['time_stamp'])

            # update landmarks
            for n in range(0, 68):
                x = int(self.__csv_row['mark_%d_x' % (n)])
                y = int(self.__csv_row['mark_%d_y' % (n)])

                self.__landmarks.append((x, y))

            # update rect
            self.__rect.append((int(self.__csv_row['target_left']), int(self.__csv_row['target_top'])))
            self.__rect.append((int(self.__csv_row['target_right']), int(self.__csv_row['target_bottom'])))

            # read the next row
            try:
                self.__csv_row = next(self.csv_reader)
                self.__csv_index = int(self.__csv_row['index'])
            except StopIteration:
                self.__csv_index = self.frame_count + 1

        return ret, frame

    def get_frame_index(self):
        return self.__frame_index

    def get_landmarks(self):
        return self.__landmarks

    def get_rect(self):
        return self.__rect

    def get_time_stamp(self):
        return self.__time_stamp

    def available(self):
        if len(self.__landmarks) != 68:
            return False

        if len(self.__rect) != 2:
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
