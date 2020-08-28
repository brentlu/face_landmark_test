#!/usr/bin/python3

from facial_engine import FacialEngine

#import csv
#import cv2
#import dlib
#import hashlib
import magic
#import numpy as np
import os
import re
#import subprocess
#import time


def main():
    #input_video_path = '/media'
    #input_video_path = '/media/Temp_AIpose20200811'
    #input_video_path = '20200429_2B.mp4'
    input_video_path = '.'

    if os.path.isfile(input_video_path) != False:

        engine = FacialEngine(input_video_path)

        # update the csv if found
        ret = engine.configure(csv_policy = 'update')
        if ret == False:
            print('main: fail to configure engine')
            return

        ret = engine.process_video()
        if ret == False:
            print('main: fail to process video')

    elif os.path.isdir(input_video_path):

        # looking for any video file which name ends with a 'A' or 'B' character
        prog = re.compile(r'.*[AB]\..+')

        mime = magic.Magic(mime=True)
        for root, dirs, files in os.walk(input_video_path):
            for file in files:
                file_path = os.path.join(root, file)

                file_mine = mime.from_file(file_path)
                if file_mine.find('video') == -1:
                    continue

                if prog.match(file) == None:
                    continue

                engine = FacialEngine(file_path)

                #ret = engine.configure(csv_policy = 'abort')
                ret = engine.configure(csv_policy = 'update')
                if ret == False:
                    print('main: fail to configure engine')

                ret = engine.process_video()
                if ret == False:
                    print('main: fail to process video')

    else:
        print('Fail to process input path %s' % (input_video_path))

    return



if __name__ == '__main__':
    main()
