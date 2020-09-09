#!/usr/bin/python3

from facial_engine import FacialEngine
from scipy.spatial import distance as dist
import csv
import cv2
import numpy
import os


# a wrapper class for opencv's VideoCapture class
class FacialVideo:
    def __init__(self, video_path):
        # open the video file
        self.__cap = cv2.VideoCapture(video_path)

        self.width = int(self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.__cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.__cap.get(cv2.CAP_PROP_FRAME_COUNT))

        engine = FacialEngine(video_path)
        self.__rotation = engine.get_video_rotation()

        if self.__rotation == cv2.ROTATE_90_CLOCKWISE or self.__rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
            temp = self.width
            self.width = self.height
            self.height = temp

        self.__frame_index = 0
        self.__csv_data = False

        self.__csv_path = engine.get_csv_data_file()

        if self.__csv_path != None:
            if os.path.isfile(self.__csv_path) == False:
                # should not get here
                return

            # open the csv file
            self.__csv_file = open(self.__csv_path, 'r', newline = '')
            self.__csv_reader = csv.DictReader(self.__csv_file)

            # read the first row
            try:
                self.__csv_row = next(self.__csv_reader)
                self.__csv_index = int(self.__csv_row['index'])
            except StopIteration:
                self.__csv_index = self.frame_count + 1

        # eye aspect ratio
        self.min_ear = [1.0, 1.0]
        self.max_ear = [0.0, 0.0]
        self.avg_ear = [0.0, 0.0]

        # eye width
        self.min_ew = [1000.0, 1000.0]
        self.max_ew = [0.0, 0.0]
        self.avg_ew = [0.0, 0.0]

    def __del__(self):
        self.__csv_file.close()
        self.__cap.release()

    def read(self):
        ret, frame = self.__cap.read()
        if ret == False:
            # no frame left in the video
            return ret, frame

        self.__frame_index += 1

        if self.__rotation != -1:
            frame = cv2.rotate(frame, self.__rotation)

        self.__time_stamp = 0.0
        self.__landmarks = []
        self.__rect = []

        # this frame has no csv data
        if self.__frame_index != self.__csv_index:
            self.__csv_data = False
            return ret, frame

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

        # data is ready for this frame
        self.__csv_data = True

        # read the next row
        try:
            self.__csv_row = next(self.__csv_reader)
            self.__csv_index = int(self.__csv_row['index'])
        except StopIteration:
            self.__csv_index = self.frame_count + 1

        return ret, frame

    def available(self):
        if self.__csv_data == False:
            return False

        return True

    def get_frame_index(self):
        return self.__frame_index

    def get_landmarks(self):
        return self.__landmarks

    def get_rect(self):
        return self.__rect

    def get_time_stamp(self):
        return self.__time_stamp

    def get_eye_aspect_ratio(self, eye, landmarks = None):
        if landmarks == None:
            landmarks = self.__landmarks

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
            return 0.0

        ear = (A + B) / (2.0 * C)

        return ear

    def get_eye_width(self, eye, landmarks = None):
        if landmarks == None:
            landmarks = self.__landmarks

        if len(landmarks) != 68:
            print('incomplete landmark')

        if eye == 'left':
            return dist.euclidean(landmarks[42], landmarks[45])
        elif eye == 'right':
            return dist.euclidean(landmarks[39], landmarks[36])
        else:
            print('get_eye_width: unknown eye %s' % (eye))

        return 0.0

    def update_static_data(self, start = 0, end = 0):
        total_frame = 0

        # open the csv file
        csvfile = open(self.__csv_path, 'r', newline = '')
        csv_reader = csv.DictReader(csvfile)

        for csv_row in csv_reader:
            landmarks = []

            index = int(csv_row['index'])

            if index < start:
                continue
            if end != 0:
                if index >= end:
                    break

            for n in range(0, 68):
                x = int(csv_row['mark_%d_x' % (n)])
                y = int(csv_row['mark_%d_y' % (n)])

                landmarks.append((x, y))

            # ear of left eye
            ear = self.get_eye_aspect_ratio('left', landmarks)
            if ear < self.min_ear[0]:
                self.min_ear[0] = ear
            if ear > self.max_ear[0]:
                self.max_ear[0] = ear

            self.avg_ear[0] += ear

            # ear of right eye
            ear = self.get_eye_aspect_ratio('right', landmarks)
            if ear < self.min_ear[1]:
                self.min_ear[1] = ear
            if ear > self.max_ear[1]:
                self.max_ear[1] = ear

            self.avg_ear[1] += ear

            # width of left eye
            ew = self.get_eye_width('left', landmarks)
            if ew < self.min_ew[0]:
                self.min_ew[0] = ew
            if ew > self.max_ew[0]:
                self.max_ew[0] = ew

            self.avg_ew[0] += ew

            # width of right eye
            ew = self.get_eye_width('right', landmarks)
            if ew < self.min_ew[1]:
                self.min_ew[1] = ew
            if ew > self.max_ew[1]:
                self.max_ew[1] = ew

            self.avg_ew[1] += ew

            total_frame += 1

        csvfile.close()

        if total_frame != 0:
            self.avg_ear[0] /= total_frame
            self.avg_ear[1] /= total_frame
            self.avg_ew[0] /= total_frame
            self.avg_ew[1] /= total_frame
            return True

        return False

    def find_face_rect(self, start = 0, end = 0):
        rect_left = self.width
        rect_right = 0
        rect_top = self.height
        rect_bottom = 0

        # open the csv file
        csvfile = open(self.__csv_path, 'r', newline = '')
        csv_reader = csv.DictReader(csvfile)

        for csv_row in csv_reader:
            index = int(csv_row['index'])

            if index < start:
                continue
            if end != 0:
                if index >= end:
                    break

            left = int(csv_row['target_left'])
            top = int(csv_row['target_top'])
            right = int(csv_row['target_right'])
            bottom = int(csv_row['target_bottom'])

            if left < rect_left:
                rect_left = left
            if top < rect_top:
                rect_top = top
            if right > rect_right:
                rect_right = right
            if bottom > rect_bottom:
                rect_bottom = bottom

        csvfile.close()

        rect = []
        rect.append((rect_left, rect_top))
        rect.append((rect_right, rect_bottom))

        return rect

    def find_frames_continuous(self, min = 30):

        frame_start = 1
        frame_count = 0

        frames = []

        # open the csv file
        csvfile = open(self.__csv_path, 'r', newline = '')
        csv_reader = csv.DictReader(csvfile)

        for csv_row in csv_reader:
            index = int(csv_row['index'])

            if frame_start + frame_count == index:
                frame_count += 1
            else:
                if frame_count >= min:
                    frames.append((frame_start, frame_start + frame_count))
                frame_start = index
                frame_count = 1

        if frame_start != 0 and frame_count >= min:
            frames.append((frame_start, frame_start + frame_count))

        csvfile.close()

        return frames

    def find_frames_eye_width_threshold(self, start, end, threshold, min):

        frame_start = start
        frame_count = 0

        frames = []

        # open the csv file
        csvfile = open(self.__csv_path, 'r', newline = '')
        csv_reader = csv.DictReader(csvfile)

        for csv_row in csv_reader:
            landmarks = []

            index = int(csv_row['index'])

            if index < start:
                continue
            elif index >= end:
                break

            for n in range(0, 68):
                x = int(csv_row['mark_%d_x' % (n)])
                y = int(csv_row['mark_%d_y' % (n)])

                landmarks.append((x, y))

            ew_left = self.get_eye_width('left', landmarks)
            ew_right = self.get_eye_width('right', landmarks)

            if (ew_left * 100.0) > (self.max_ew[0] * threshold) and (ew_right * 100.0) > (self.max_ew[1] * threshold):
                if frame_start + frame_count == index:
                    frame_count += 1
                #else:
                    # should not happen
            else:
                if frame_count >= min:
                    frames.append((frame_start, frame_start + frame_count))
                frame_start = index + 1
                frame_count = 0

        if frame_start != 0 and frame_count >= min:
            frames.append((frame_start, frame_start + frame_count))

        csvfile.close()

        return frames
