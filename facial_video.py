#!/usr/bin/python3

import csv
import cv2
import dlib
import os
import process_video as pv


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

        self.__frame_index = 0

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
