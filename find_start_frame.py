#!/usr/bin/python3

from facial_video import FacialVideo
import argparse
import csv
import magic
import os


def process_one_video(input_video_path, duration = 30.0):
    found = 0

    print('Configuration:')
    print('  input video path:  %s' % (input_video_path))
    print('  duration:  %s' % (duration))

    fv = FacialVideo(input_video_path)

    ret = fv.update_static_data(1, fv.frame_count)

    if ret != False:
        print('  eye aspect ratio(left):  min %.3f, avg %.3f, max %.3f' % (fv.min_ear[0], fv.avg_ear[0], fv.max_ear[0]))
        print('  eye aspect ratio(right): min %.3f, avg %.3f, max %.3f' % (fv.min_ear[1], fv.avg_ear[1], fv.max_ear[1]))
        print('  eye width(left):         min %.3f, avg %.3f, max %.3f' % (fv.min_ew[0], fv.avg_ew[0], fv.max_ew[0]))
        print('  eye width(right):        min %.3f, avg %.3f, max %.3f' % (fv.min_ew[1], fv.avg_ew[1], fv.max_ew[1]))

    print('Continuous frame:')

    segments = fv.find_frames_continuous(duration * fv.fps)

    if len(segments) == 0:
        print('  none')
        return

    for segment in segments:
        print('  segment: start: %d, end: %d' % (segment[0], segment[1]))

    print('Best fit:')
    for n in range(90, 30, -1):
        for segment in segments:
            frames = fv.find_frames_eye_width_threshold(segment[0], segment[1], n, duration * fv.fps)

            # try next segment
            if len(frames) == 0:
                continue

            for frame in frames:
                print('  threshold: %d, start: %d, end: %d' % (n, frame[0], frame[1]))
                found = 1

        if found != 0:
            break

    if found == 0:
        print('  none')

    return


def process_training_csv(csv_path):

    _, filename = os.path.split(csv_path)
    print('process_training_csv: %s' % (filename))

    with open(csv_path, 'r', newline = '') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            video_path = row['file_name']
            duration = float(row['duration'])

            ret = process_one_video(video_path, duration)
            if ret == False:
                return False

    print('  success')
    return True

def main():

    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help = 'path to a video file or a training recipe file')

    parser.add_argument('-d', '--duration', help = 'duration (sec)')

    args = parser.parse_args()

    input_video_path = args.path

    print('User input:')
    print('  input path: %s' % (input_video_path))

    if args.duration != None:
        duration = float(args.duration)
        print('  duration:   %.3f' % (duration))

    _, ext = os.path.splitext(input_video_path)

    if ext == '.csv':
        if args.duration != None:
            print('  ignore duration')

        # could be a training recipe
        ret = process_training_csv(input_video_path)

    elif os.path.isfile(input_video_path) != False:
        mime = magic.Magic(mime=True)

        file_mine = mime.from_file(input_video_path)
        if file_mine.find('video') == -1:
            print('  not a video file')
            return False

        ret = process_one_video(input_video_path, duration)

    else:
        print('Unrecognized path')

    return

if __name__ == '__main__':
    main()
