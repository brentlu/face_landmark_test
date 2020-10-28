#!/usr/bin/python3

from facial_engine import FacialEngine
from scipy.spatial import distance as dist
import csv
import cv2
import numpy as np
import os


# a wrapper class for opencv's VideoCapture class
class FacialVideo:
    LEFT_EYE = 0
    RIGHT_EYE = 1

    MIN = 0
    AVG = 1
    MAX = 2

    def __init__(self, video_path):
        self.__init = False

        # open the video file
        self.__cap = cv2.VideoCapture(video_path)

        self.width = int(self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.__cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.__cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.__engine = FacialEngine(video_path)
        if self.__engine.init() == False:
            print('fv: fail to init engine')
            self.__cap.release()
            return

        self.__rotation = self.__engine.get_video_rotation()

        if self.__rotation == cv2.ROTATE_90_CLOCKWISE or self.__rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
            temp = self.width
            self.width = self.height
            self.height = temp

        self.__frame_index = 0
        self.__csv_data = False

        self.__csv_path = self.__engine.get_csv_data_file()

        if self.__csv_path != None:
            if os.path.isfile(self.__csv_path) == False:
                # should not get here
                print('fv: meta-csv file not exist')
                self.__cap.release()
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

        self.__statistic_data = False

        # initial eye aspect ratio
        self.eye_aspect_ratio = np.zeros((2, 3), dtype = float)
        self.eye_aspect_ratio[self.LEFT_EYE][self.MIN] = 1.0
        self.eye_aspect_ratio[self.RIGHT_EYE][self.MIN] = 1.0

        # initial eye width
        self.eye_width = np.zeros((2, 3), dtype = float)
        self.eye_width[self.LEFT_EYE][self.MIN] = 1000.0
        self.eye_width[self.RIGHT_EYE][self.MIN] = 1000.0

        # initial eye-to-mouth length
        self.eye_to_mouth = np.zeros((2, 3), dtype = float)
        self.eye_to_mouth[self.LEFT_EYE][self.MIN] = 10000.0
        self.eye_to_mouth[self.RIGHT_EYE][self.MIN] = 10000.0

        self.__init = True
        return

    def __del__(self):
        if self.__init == False:
            return

        self.__csv_file.close()
        self.__cap.release()

    def init(self):
        return self.__init

    def read(self, no_image = False):
        frame = []

        ret = self.__cap.grab()
        if ret == False:
            # no frame left in the video
            return ret, frame

        if no_image == False:
            ret, frame = self.__cap.retrieve()
            if ret == False:
                # decode fail
                return ret, frame

        self.__frame_index += 1

        if self.__rotation != -1 and no_image == False:
            frame = cv2.rotate(frame, self.__rotation)

        self.__time_stamp = 0.0
        self.__landmarks = np.zeros((68, 2), dtype = int)
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

            self.__landmarks[n] = (x, y)

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
        # always available
        return self.__frame_index

    def get_landmarks(self):
        if self.__csv_data == False:
            print('fv: frame landmarks not available')

        return self.__landmarks

    def get_rect(self):
        if self.__csv_data == False:
            print('fv: frame rect not available')

        return self.__rect

    def get_time_stamp(self):
        if self.__csv_data == False:
            print('fv: frame time stamp not available')

        return self.__time_stamp

    def calculate_eye_aspect_ratio(self, landmarks = None):
        if landmarks is None:
            landmarks = self.__landmarks

        if len(landmarks) != 68:
            print('fv: invalid landmarks')
            return 0.0, 0.0

        # euclidean distances between the two sets of vertical eye landmarks
        A = dist.euclidean(landmarks[43], landmarks[47])
        B = dist.euclidean(landmarks[44], landmarks[46])
        # euclidean distance between the horizontal eye landmark
        C = dist.euclidean(landmarks[42], landmarks[45])

        left = (A + B) / (2.0 * C)

        # euclidean distances between the two sets of vertical eye landmarks
        A = dist.euclidean(landmarks[38], landmarks[40])
        B = dist.euclidean(landmarks[37], landmarks[41])
        # euclidean distance between the horizontal eye landmark
        C = dist.euclidean(landmarks[39], landmarks[36])

        right = (A + B) / (2.0 * C)

        return left, right

    def calculate_eye_width(self, landmarks = None):
        if landmarks is None:
            landmarks = self.__landmarks

        if len(landmarks) != 68:
            print('fv: invalid landmarks')
            return 0.0, 0.0

        left = dist.euclidean(landmarks[42], landmarks[45])
        right = dist.euclidean(landmarks[39], landmarks[36])

        return left, right

    def calculate_inner_eye_height(self, landmarks = None):
        if landmarks is None:
            landmarks = self.__landmarks

        if len(landmarks) != 68:
            print('fv: invalid landmarks')
            return 0.0, 0.0

        left = dist.euclidean(landmarks[43], landmarks[47])
        right = dist.euclidean(landmarks[38], landmarks[40])

        return left, right

    def calculate_eye_to_mouth_length(self, landmarks = None):
        if landmarks is None:
            landmarks = self.__landmarks

        if len(landmarks) != 68:
            print('fv: invalid landmarks')
            return 0.0, 0.0

        left = dist.euclidean(landmarks[36], landmarks[48])
        right = dist.euclidean(landmarks[45], landmarks[54])

        return left, right

    def update_statistic_data(self, start = 0, end = 0):
        total_frame = 0

        # initial eye aspect ratio
        self.eye_aspect_ratio = np.zeros((2, 3), dtype = float)
        self.eye_aspect_ratio[self.LEFT_EYE][self.MIN] = 1.0
        self.eye_aspect_ratio[self.RIGHT_EYE][self.MIN] = 1.0

        # initial eye width
        self.eye_width = np.zeros((2, 3), dtype = float)
        self.eye_width[self.LEFT_EYE][self.MIN] = 10000.0
        self.eye_width[self.RIGHT_EYE][self.MIN] = 10000.0

        # initial eye inner height
        self.inner_eye_height = np.zeros((2, 3), dtype = float)
        self.inner_eye_height[self.LEFT_EYE][self.MIN] = 10000.0
        self.inner_eye_height[self.RIGHT_EYE][self.MIN] = 10000.0

        # initial eye-to-mouth length
        self.eye_to_mouth = np.zeros((2, 3), dtype = float)
        self.eye_to_mouth[self.LEFT_EYE][self.MIN] = 10000.0
        self.eye_to_mouth[self.RIGHT_EYE][self.MIN] = 10000.0

        # open the csv file
        with open(self.__csv_path, 'r', newline = '') as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for csv_row in csv_reader:
                landmarks = np.zeros((68, 2), dtype = int)

                index = int(csv_row['index'])

                if index < start:
                    continue
                if end != 0:
                    if index > end:
                        break

                for n in range(0, 68):
                    x = int(csv_row['mark_%d_x' % (n)])
                    y = int(csv_row['mark_%d_y' % (n)])

                    landmarks[n] = (x, y)

                ear = self.calculate_eye_aspect_ratio(landmarks)
                ew = self.calculate_eye_width(landmarks)
                ieh = self.calculate_inner_eye_height(landmarks)
                em = self.calculate_eye_to_mouth_length(landmarks)

                eyes = (self.LEFT_EYE, self.RIGHT_EYE)

                for eye in eyes:
                    if ear[eye] < self.eye_aspect_ratio[eye][self.MIN]:
                        self.eye_aspect_ratio[eye][self.MIN] = ear[eye]
                    if ear[eye] > self.eye_aspect_ratio[eye][self.MAX]:
                        self.eye_aspect_ratio[eye][self.MAX] = ear[eye]
                    self.eye_aspect_ratio[eye][self.AVG] += ear[eye]

                    if ew[eye] < self.eye_width[eye][self.MIN]:
                        self.eye_width[eye][self.MIN] = ew[eye]
                    if ew[eye] > self.eye_width[eye][self.MAX]:
                        self.eye_width[eye][self.MAX] = ew[eye]
                    self.eye_width[eye][self.AVG] += ew[eye]

                    if ieh[eye] < self.inner_eye_height[eye][self.MIN]:
                        self.inner_eye_height[eye][self.MIN] = ieh[eye]
                    if ieh[eye] > self.inner_eye_height[eye][self.MAX]:
                        self.inner_eye_height[eye][self.MAX] = ieh[eye]
                    self.inner_eye_height[eye][self.AVG] += ieh[eye]

                    if em[eye] < self.eye_to_mouth[eye][self.MIN]:
                        self.eye_to_mouth[eye][self.MIN] = em[eye]
                    if em[eye] > self.eye_to_mouth[eye][self.MAX]:
                        self.eye_to_mouth[eye][self.MAX] = em[eye]
                    self.eye_to_mouth[eye][self.AVG] += em[eye]

                total_frame += 1

        if total_frame != 0:
            for eye in eyes:
                self.eye_aspect_ratio[eye][self.AVG] /= total_frame
                self.eye_width[eye][self.AVG] /= total_frame
                self.inner_eye_height[eye][self.AVG] /= total_frame
                self.eye_to_mouth[eye][self.AVG] /= total_frame

            self.__statistic_data = True
            return True

        self.__statistic_data = False
        return False

    def get_eye_aspect_ratio(self, type):
        if type != self.MIN and type != self.AVG and type != self.MAX:
            print('fv: invalid type %s' % (str(type)))
            return 0.0, 0.0

        if self.__statistic_data == False:
            print('fv: statistic data not available')
            return 0.0, 0.0

        return self.eye_aspect_ratio[self.LEFT_EYE][type], self.eye_width[self.RIGHT_EYE][type]

    def get_eye_width(self, type):
        if type != self.MIN and type != self.AVG and type != self.MAX:
            print('fv: invalid type %s' % (str(type)))
            return 0.0, 0.0

        if self.__statistic_data == False:
            print('fv: statistic data not available')
            return 0.0, 0.0

        return self.eye_width[self.LEFT_EYE][type], self.eye_width[self.RIGHT_EYE][type]

    def get_inner_eye_height(self, type):
        if type != self.MIN and type != self.AVG and type != self.MAX:
            print('fv: invalid type %s' % (str(type)))
            return 0.0, 0.0

        if self.__statistic_data == False:
            print('fv: statistic data not available')
            return 0.0, 0.0

        return self.inner_eye_height[self.LEFT_EYE][type], self.inner_eye_height[self.RIGHT_EYE][type]

    def get_eye_to_mouth_length(self, type):
        if type != self.MIN and type != self.AVG and type != self.MAX:
            print('fv: invalid type %s' % (str(type)))
            return 0.0, 0.0

        if self.__statistic_data == False:
            print('fv: statistic data not available')
            return 0.0, 0.0

        return self.eye_to_mouth[self.LEFT_EYE][type], self.eye_to_mouth[self.RIGHT_EYE][type]

    def find_face_rect(self, start = 0, end = 0):
        rect_left = self.width
        rect_right = 0
        rect_top = self.height
        rect_bottom = 0

        # open the csv file
        with open(self.__csv_path, 'r', newline = '') as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for csv_row in csv_reader:
                index = int(csv_row['index'])

                if index < start:
                    continue
                if end != 0:
                    if index > end:
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

        rect = []
        rect.append((rect_left, rect_top))
        rect.append((rect_right, rect_bottom))

        return rect

    def find_continuous_frames(self, min_duration = 30):

        frame_start = 1
        frame_count = 0

        frames = []

        # open the csv file
        with open(self.__csv_path, 'r', newline = '') as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for csv_row in csv_reader:
                index = int(csv_row['index'])

                if frame_start + frame_count == index:
                    frame_count += 1
                else:
                    if frame_count >= min_duration:
                        frames.append((frame_start, frame_start + frame_count - 1))
                    frame_start = index
                    frame_count = 1

            if frame_count >= min_duration:
                frames.append((frame_start, frame_start + frame_count - 1))

        return frames

    def find_front_face_frames(self, start, end, threshold, min_duration):
        frame_start = start
        frame_count = 0

        frames = []

        # open the csv file
        with open(self.__csv_path, 'r', newline = '') as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for csv_row in csv_reader:
                index = int(csv_row['index'])

                if index < start:
                    continue
                elif index > end:
                    break

                landmarks = np.zeros((68, 2), dtype = int)
                eye_percent = [0.0, 0.0]

                for n in range(0, 68):
                    x = int(csv_row['mark_%d_x' % (n)])
                    y = int(csv_row['mark_%d_y' % (n)])

                    landmarks[n] = (x, y)

                eye_width = self.calculate_eye_width(landmarks)

                eyes = (self.LEFT_EYE, self.RIGHT_EYE)

                for eye in eyes:
                    eye_percent[eye] = (eye_width[eye] * 100.0) / self.eye_width[eye][self.MAX]

                eye_diff = abs(eye_percent[self.LEFT_EYE] - eye_percent[self.RIGHT_EYE])

                if eye_diff <= threshold:
                    if frame_start + frame_count == index:
                        frame_count += 1
                    #else:
                        # should not happen
                else:
                    if frame_count >= min_duration:
                        frames.append((frame_start, frame_start + frame_count - 1))
                    frame_start = index + 1
                    frame_count = 0

            if frame_count >= min_duration:
                frames.append((frame_start, frame_start + frame_count - 1))

        return frames
