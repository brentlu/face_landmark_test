#!/usr/bin/python3

import csv
import cv2
import dlib
import hashlib
import magic
import os
import time


class Logger:
    def __init__(self):
        print('Start logger:')

        # get current time (UTC time)
        now = time.gmtime()

        timestamp = time.strftime('%Y-%m%d-%H%M', now)

        log_path_base = os.path.join(get_data_path('log'), timestamp)
        log_path_base = os.path.abspath(log_path_base)

        # find a free slot
        for n in range(1, 100):
            log_path = '%s-%s.log' % (log_path_base, str(n))
            if os.path.exists(log_path) == False:
                break

        self.__file = open(log_path, 'w')

        print('  log path = %s' % (log_path))

        return

    def __del__(self):
        self.__file.close()

    def print(self, string, end = '\n'):
        # print to screen directly
        print(string, end = end)

        if end == '\r':
            end = '\n'

        # get current time (local time)
        now = time.localtime()
        timestamp = time.strftime('%Y-%m-%dT%H:%M:%S%z', now)

        self.__file.write('%s %s%s' % (timestamp, string, end))

        return

class FacialEngine:
    def __init__(self, video_path):
        # translate to abs path
        self.input_video_path = os.path.abspath(video_path)

        # TODO: check if file exists
        mime = magic.Magic(mime=True)

        file_mine = mime.from_file(self.input_video_path)
        if file_mine.find('video') == -1:
            print('  not a video file')

        # first 64KB should be sufficient
        self.input_video_hash = self.calculate_md5_digest(64 * 1024)

        # check data directory
        if self.check_data_directory() == False:
            print('  fail to check data directory')
            # TODO: raise an exception for this
            return

        # start the logger
        self.logger = Logger()

        # init for all videos
        self.hog_detector = dlib.get_frontal_face_detector()
        self.cnn_detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        # default options
        self.output_video = False
        self.compress_video = False
        self.csv_policy = 'overwrite' # other policy could be 'abort' or 'update'
        self.max_frames = -1
        self.use_cnn = False

        self.input_csv = False

        return

    def __del__(self):
        # remove old csv file
        if self.input_csv != False:
            os.remove(self.input_csv_path)

        return

    # calculate md5 hash for input video file
    def calculate_md5_digest(self, size):
        block_size = 4096
        remain_size = size

        hash = hashlib.md5()
        with open(self.input_video_path, 'rb') as f:
            while remain_size > block_size:
                block = f.read(block_size)
                hash.update(block)
                remain_size -= block_size

            if remain_size != 0:
                block = f.read(remain_size)
                hash.update(block)

        return hash.hexdigest()

    def check_data_directory(self):
        # create basic layout of data directory
        # run before logger starts
        print('Check data directory:')

        data_path = get_data_path('data')
        data_path = os.path.abspath(data_path)

        csv_path = get_data_path('csv')
        csv_path = os.path.abspath(csv_path)

        log_path = get_data_path('log')
        log_path = os.path.abspath(log_path)

        video_path = get_data_path('video')
        video_path = os.path.abspath(video_path)

        pathes = (data_path, csv_path, log_path, video_path)

        for path in pathes:
            if os.path.isdir(path) == False:
                try:
                    print('  create %s directory' % (path))
                    os.mkdir(path)
                except OSError:
                    print('  fail to create %s directory' % (path))
                    return False

        print('  all good')

        return True

    def configure(self, csv_policy = 'overwrite', output_video = False, compress_video = False, max_frames = -1, use_cnn = False):
        self.logger.print('Configure engine:')
        self.logger.print('  input video path  = %s' % (self.input_video_path))

        self.output_csv_path = self.get_output_csv_path()
        self.logger.print('  output csv path   = %s' % (self.output_csv_path))

        if os.path.isfile(self.output_csv_path) != False:
            if csv_policy == 'update':
                # partial update csv file
                self.logger.print('  csv data will be updated')

                directory, _ = os.path.split(self.output_csv_path)
                self.input_csv_path = os.path.join(directory, 'tmp.csv')

                # delete the tmp csv if already exist
                if os.path.exists(self.input_csv_path):
                    os.remove(self.input_csv_path)

                os.rename('%s' % self.output_csv_path, '%s' % self.input_csv_path)

                self.input_csv = True
            elif csv_policy == 'abort':
                self.logger.print('  csv data exists, abort')
                return False
            elif csv_policy == 'overwrite':
                self.logger.print('  csv data will be overwritten')
            else:
                self.logger.print('  unknown csv policy %s, abort' % (policy))
                return False

        if output_video != False:
            if self.input_csv != False:
                    self.logger.print('  video data will not be generated')
            else:
                self.output_video = True

                self.output_video_path = self.get_output_video_path()
                self.logger.print('  output video path = %s' % (self.output_video_path))

                if os.path.isfile(self.output_video_path) != False:
                    self.logger.print('  video data will be overwritten')

                if compress_video != False:
                    self.compress_video = True
                    self.logger.print('  video data will be compressed')

        if max_frames != -1:
            self.max_frames = max_frames
            self.logger.print('  only process %d frames' % (self.max_frames))

        if use_cnn != False:
            self.use_cnn = True
            self.logger.print('  use cnn detector if hog fails')

        return True

    def get_output_csv_path(self):
        # remove directory part
        _, file_name = os.path.split(self.input_video_path)

        # remove ext part
        file_name, _ = os.path.splitext(file_name)

        # generate the path for output csv file
        output_csv_name = '%s-%s.csv' % (file_name, str(self.input_video_hash))
        output_csv_path = os.path.join(get_data_path('csv'), output_csv_name)
        output_csv_path = os.path.abspath(output_csv_path)

        return output_csv_path

    def get_output_video_path(self):
        # remove directory part
        _, file_name = os.path.split(self.input_video_path)

        # remove ext part
        file_name, _ = os.path.splitext(file_name)

        # generate the path for output video file
        output_video_name = '%s-%s.mp4' % (file_name, str(self.input_video_hash))
        output_video_path = os.path.join(get_data_path('video'), output_video_name)
        output_video_path = os.path.abspath(output_video_path)

        return output_csv_path

    def process_video(self):
        # init for video
        csv_index = 0
        frame_index = 0
        frame_fail_count = 0
        csv_fields = ['index', 'detector', 'total_face_num', 'center_face_num', 'target_left', 'target_top', 'target_right', 'target_bottom', 'time_stamp', \
                      'mark_0_x', 'mark_0_y', 'mark_1_x', 'mark_1_y', 'mark_2_x', 'mark_2_y', 'mark_3_x', 'mark_3_y', 'mark_4_x', 'mark_4_y', 'mark_5_x', 'mark_5_y', 'mark_6_x', 'mark_6_y', 'mark_7_x', 'mark_7_y', 'mark_8_x', 'mark_8_y', 'mark_9_x', 'mark_9_y', \
                      'mark_10_x', 'mark_10_y', 'mark_11_x', 'mark_11_y', 'mark_12_x', 'mark_12_y', 'mark_13_x', 'mark_13_y', 'mark_14_x', 'mark_14_y', 'mark_15_x', 'mark_15_y', 'mark_16_x', 'mark_16_y', 'mark_17_x', 'mark_17_y', 'mark_18_x', 'mark_18_y', 'mark_19_x', 'mark_19_y', \
                      'mark_20_x', 'mark_20_y', 'mark_21_x', 'mark_21_y', 'mark_22_x', 'mark_22_y', 'mark_23_x', 'mark_23_y', 'mark_24_x', 'mark_24_y', 'mark_25_x', 'mark_25_y', 'mark_26_x', 'mark_26_y', 'mark_27_x', 'mark_27_y', 'mark_28_x', 'mark_28_y', 'mark_29_x', 'mark_29_y', \
                      'mark_30_x', 'mark_30_y', 'mark_31_x', 'mark_31_y', 'mark_32_x', 'mark_32_y', 'mark_33_x', 'mark_33_y', 'mark_34_x', 'mark_34_y', 'mark_35_x', 'mark_35_y', 'mark_36_x', 'mark_36_y', 'mark_37_x', 'mark_37_y', 'mark_38_x', 'mark_38_y', 'mark_39_x', 'mark_39_y', \
                      'mark_40_x', 'mark_40_y', 'mark_41_x', 'mark_41_y', 'mark_42_x', 'mark_42_y', 'mark_43_x', 'mark_43_y', 'mark_44_x', 'mark_44_y', 'mark_45_x', 'mark_45_y', 'mark_46_x', 'mark_46_y', 'mark_47_x', 'mark_47_y', 'mark_48_x', 'mark_48_y', 'mark_49_x', 'mark_49_y', \
                      'mark_50_x', 'mark_50_y', 'mark_51_x', 'mark_51_y', 'mark_52_x', 'mark_52_y', 'mark_53_x', 'mark_53_y', 'mark_54_x', 'mark_54_y', 'mark_55_x', 'mark_55_y', 'mark_56_x', 'mark_56_y', 'mark_57_x', 'mark_57_y', 'mark_58_x', 'mark_58_y', 'mark_59_x', 'mark_59_y', \
                      'mark_60_x', 'mark_60_y', 'mark_61_x', 'mark_61_y', 'mark_62_x', 'mark_62_y', 'mark_63_x', 'mark_63_y', 'mark_64_x', 'mark_64_y', 'mark_65_x', 'mark_65_y', 'mark_66_x', 'mark_66_y', 'mark_67_x', 'mark_67_y']

        self.logger.print('Process video:')

        cap = cv2.VideoCapture(self.input_video_path)

        if cap.isOpened() == False:
            self.logger.print('  fail to open %s' % (self.input_video_path))
            self.logger.print('Failed to process video file %s\n' % (self.input_video_path))
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.print('  video property:')
        self.logger.print('    width       = %d' % (width))
        self.logger.print('    height      = %d' % (height))
        self.logger.print('    fps         = %d' % (fps))
        self.logger.print('    fourcc      = %s' % (self.decode_fourcc(fourcc)))
        self.logger.print('    frame_count = %d' % (frame_count))

        rotation = self.auto_detect_rotation()

        if rotation == cv2.ROTATE_90_CLOCKWISE or rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
            temp = width
            width = height
            height = temp

        if self.output_video != False:
            # always use mp4
            video_writer = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                           fps, (width, height))

        if self.input_csv != False:
            csv_file_read = open(self.input_csv_path, 'r', newline = '')
            csv_reader = csv.DictReader(csv_file_read)

            # read the first row
            try:
                row = next(csv_reader)
                csv_index = int(row['index'])
            except StopIteration:
                csv_index = frame_count + 1

        csv_file_write = open(self.output_csv_path, 'w', newline='')
        csv_writer = csv.DictWriter(csv_file_write, fieldnames = csv_fields)
        csv_writer.writeheader()

        while True:
            ret, frame = cap.read()
            if ret == False:
                # no frame to process
                break;

            # prepare an empty dict
            frame_result = {}

            frame_index += 1

            self.logger.print('  frame: (%3d/%d), ' % (frame_index, frame_count), end = '')

            if self.input_csv != False:
                if frame_index == csv_index:
                    # copy dict entry
                    self.logger.print('copy from csv file', end = '\r')
                    csv_writer.writerow(row)

                    # read the next row
                    try:
                        row = next(csv_reader)
                        csv_index = int(row['index'])
                    except StopIteration:
                        csv_index = frame_count + 1

                    continue

            if rotation != -1:
                frame = cv2.rotate(frame, rotation)

            ret = self.process_frame(frame, frame_result)
            if ret == False:
                frame_fail_count += 1
            else:
                time_stamp = frame_index / fps
                frame_result['time_stamp'] = time_stamp
                frame_result['index'] = frame_index

                csv_writer.writerow(frame_result)

            if self.output_video != False:
                video_writer.write(frame)

            # don't want to process entire video
            if self.max_frames > 0:
                if frame_index >= self.max_frames:
                    break;

        self.logger.print('Statistic:')
        self.logger.print('  total %d frames' % (frame_count))
        self.logger.print('  %d frames processed' % (frame_index))
        self.logger.print('  %d frames (%3.2f%%) failed' % (frame_fail_count, frame_fail_count * 100.0 / frame_index))

        # clean-up
        if self.output_video != False:
            video_writer.release()
        if self.input_csv != False:
            csv_file_read.close()
        csv_file_write.close()
        cap.release()

        self.logger.print('Success to process video file %s\n' % (self.input_video_path))
        return True

    def decode_fourcc(self, v):
        v = int(v)
        return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

    def auto_detect_rotation(self):
        degrees = [-1, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
        text = ['none', '90 degree clockwise', '180 degree', '90 degree counter clockwise']
        counts = [0, 0, 0, 0]

        cap = cv2.VideoCapture(self.input_video_path)

        self.logger.print('  orientation detection:')

        while True:
            ret, frame = cap.read()
            if ret == False:
                # no frame to process
                break;

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for i in range(len(degrees)):
                if degrees[i] != -1:
                    rotate = cv2.rotate(gray, degrees[i])
                    faces = self.hog_detector(rotate)
                else:
                    faces = self.hog_detector(gray)

                faces_num = len(faces)
                if faces_num != 0:
                    counts[i] += faces_num

                if counts[i] >= 5:
                    self.logger.print('    need to rotate %s' % (text[i]))
                    cap.release()
                    return degrees[i]

        # should not get here
        cap.release()
        return -1

    def process_frame(self, frame, frame_result):
        # init for frame
        state = 'init'
        frame_use_cnn = self.use_cnn

        while True:
            if state == 'init':
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # first try hog detector
                state = 'hog_detect'
            elif state == 'hog_detect':
                # get rectangle of front faces
                faces = self.hog_detector(gray)
                faces_num = len(faces)

                if faces_num == 0:
                    self.logger.print('process_frame: no face found by hog detector')

                    # try very slow cnn detector to find more faces
                    if frame_use_cnn != False:
                        state = 'cnn_detect'
                        continue

                    return False

                # we've got some faces
                frame_result['detector'] = 'h'
                frame_result['total_face_num'] = faces_num

                state = 'process_faces'
            elif state == 'cnn_detect':
                # get rectangle of front faces
                faces = self.cnn_detector(gray)
                faces_num = len(faces)

                if faces_num == 0:
                    self.logger.print('process_frame: no face found by cnn detector')
                    return False

                # we've got some faces
                # TODO: fix the rect
                frame_result['detector'] = 'c'
                frame_result['total_face_num'] = faces_num

                state = 'process_faces'
            elif state == 'process_faces':
                # find the target face
                target, center_num = self.find_target_face(frame, faces)

                frame_result['center_face_num'] = center_num

                if target == None:
                    # all faces found are not in the center position
                    self.logger.print('process_frame: fail to find target face')

                    # try very slow cnn detector to find more faces
                    if frame_use_cnn != False:
                        frame_use_cnn = False

                        state = 'cnn_detect'
                        continue

                    # draw red rectangles before leaving
                    if self.output_video != False:
                        self.draw_face_rectangles(frame, faces, target)

                    return False

                frame_result['target_left'] = target.left()
                frame_result['target_top'] = target.top()
                frame_result['target_right'] = target.right()
                frame_result['target_bottom'] = target.bottom()

                if self.output_video != False:
                    self.draw_face_rectangles(frame, faces, target)

                # get landmarks of the target face
                landmarks = self.predictor(gray, target)

                if self.output_video != False:
                    self.draw_landmarks(frame, landmarks, 'all', 'circle')

                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y

                    frame_result['mark_%d_x' % (n)] = x
                    frame_result['mark_%d_y' % (n)] = y

                self.logger.print('process_frame: success', end = '\r')
                return True
            else:
                self.logger.print('process_frame: unknown state %s' % (state))
                return False

        # should not get here
        return False

    def find_target_face(self, frame, faces):
        height, width, layers = frame.shape
        threshold_left = width * 0.4
        threshold_right = width * 0.6
        faces_center = []

        for face in faces:
            x1 = face.left()
            x2 = face.right()

            if x1 < threshold_right and x2 > threshold_left:
                faces_center.append(face)

        faces_center_num = len(faces_center)

        if faces_center_num > 1:
            # more than one face in the center, select the biggest one
            biggest = self.find_biggest_face(faces_center)
            if biggest == None:
                # should never happen
                self.logger.print('find_target_face: fail to find biggest face')

            return biggest, faces_center_num
        elif faces_center_num == 1:
            # unique center face found, draw a green rect on the face
            return faces_center[0], faces_center_num

        # all faces are not in the center
        return None, 0

    # returns biggest face among faces
    def find_biggest_face(self, faces):
        # init
        area_max = 0

        faces_num = len(faces)
        for face in faces:
            if faces_num == 1: # fast-pass
                return face
            else:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()

                area = (x2 - x1) * (y2 - y1)
                if area > area_max:
                    area_max = area
                    face_max = face

        if faces_num > 1:
            return face_max

        return None

    def draw_face_rectangles(self, frame, faces, target):
        for face in faces:
            self.draw_rect(frame, face, 'red')

        if target != None:
            self.draw_rect(frame, target, 'green')

        return

    def draw_rect(self, frame, rect, color):
        # draw rectangle
        x1 = rect.left()
        y1 = rect.top()
        x2 = rect.right()
        y2 = rect.bottom()

        if color == 'green':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        elif color == 'red':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        else:
            self.logger.print('draw_rect: unknown color %s' % (color))

        return

    def draw_landmarks(self, frame, landmarks, part, marker):
        for n in range(0, 68):
            if part == 'left-eye':
                if n < 42 or n >= 48:
                    continue
            elif part != 'all':
                self.logger.print('draw_landmarks: unknown part %s' % (part))

            x = landmarks.part(n).x
            y = landmarks.part(n).y

            if marker == 'circle':
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
            elif marker == 'text':
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
                cv2.putText(frame, str(n), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0), 1, 8, False);
            else:
                self.logger.print('draw_landmarks: unknown marker %s' % (marker))

        return

    def compress_output_video(self):
        # sanity check
        if self.output_video == False:
            return False

        directory, _ = os.path.split(self.output_video_path)
        tmp_path = os.path.join(directory, 'tmp.mp4')

        # delete the tmp video if already exist
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        os.rename('%s' % video_path, '%s' % tmp_path)

        # prepare ffmpeg command
        cmd = ['ffmpeg', '-i', tmp_path, '-c:v', 'libx264', '-preset', 'veryslow', '-crf', '28', '-c:a', 'copy', video_path]

        self.logger.print('Compress output video:')

        p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE, universal_newlines = True)
        outs, errs = p.communicate()

        if p.returncode != 0:
            self.logger.print('  compression fail')
            self.logger.print('  ffmpeg output:')
            self.logger.print(errs)

            if os.path.exists(video_path):
                os.remove(video_path)

            os.rename('%s' % tmp_path, '%s' % video_path)
            return False

        self.logger.print('  compression success')
        self.logger.print('  ffmpeg output:')
        self.logger.print(outs)

        # delete tmp video before leaving
        os.remove(tmp_path)

        return True

    def get_csv_data_file(self):

        csv_path = self.get_output_csv_path()

        if os.path.isfile(csv_path) != False:
            # already in data folder
            return csv_path

        # process the video
        ret = self.configure(self, csv_policy = 'abort')
        if ret == False:
            self.logger.print('get_csv_data_file: fail to configure engine')
            return None

        ret = self.process_video()
        if ret == False:
            self.logger.print('get_csv_data_file: fail process video')
            return None

        return csv_path

def get_data_path(directory):
    if directory == 'data':
        return './data'
    elif directory == 'csv':
        return './data/csv'
    elif directory == 'log':
        return './data/log'
    elif directory == 'video':
        return './data/video'
    else:
        print('get_data_path: unknown directory %s' % (directory))

    # should not get here
    return ''
