#!/usr/bin/python3

from facial_engine import FacialEngine
import argparse
import csv
import magic
import os
import re


def process_one_video(video_path, csv_policy):

    _, filename = os.path.split(video_path)
    print('process_one_video: %s' % (filename))

    engine = FacialEngine(video_path)
    if engine.init() == False:
        print('  fail to init engine')
        return False

    ret = engine.configure(csv_policy = csv_policy)
    if ret == False:
        print('  fail to configure engine')
        return False

    ret = engine.process_video()
    if ret == False:
        print('  fail to process video')
        return False

    print('  success')
    return True

def process_training_csv(csv_path):

    _, filename = os.path.split(csv_path)
    print('process_training_csv: %s' % (filename))

    with open(csv_path, 'r', newline = '') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # update the csv if found
            ret = process_one_video(row['file_name'], 'update')
            if ret == False:
                return False

    print('  success')
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help = 'path to a video file, a directory, or a training recipe file')

    args = parser.parse_args()

    input_video_path = args.path

    print('Input path = %s' % (input_video_path))

    _, ext = os.path.splitext(input_video_path)

    if ext == '.csv':
        # could be a training recipe
        process_training_csv(input_video_path)

    elif os.path.isfile(input_video_path) != False:
        mime = magic.Magic(mime=True)

        file_mine = mime.from_file(input_video_path)
        if file_mine.find('video') == -1:
            print('  not a video file')
            return False

        # update the csv if found
        ret = process_one_video(input_video_path, 'update')

    elif os.path.isdir(input_video_path):
        # looking for any video file which:
        # 1. file name ends with a 'A' or 'B' or 'D' character
        # 2. in the SJCAM subdirectory
        prog = re.compile(r'.*/SJCAM/.*[ABD]\..+')

        mime = magic.Magic(mime=True)
        for root, dirs, files in os.walk(input_video_path):
            for file in files:
                file_path = os.path.join(root, file)

                file_mine = mime.from_file(file_path)
                if file_mine.find('video') == -1:
                    continue

                if prog.match(file_path) == None:
                    continue

                ret = process_one_video(file_path, 'abort')

    else:
        print('Unrecognized path')

    return

if __name__ == '__main__':
    main()
