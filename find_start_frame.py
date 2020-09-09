#!/usr/bin/python3

from facial_video import FacialVideo
from tempfile import NamedTemporaryFile
import argparse
import csv
import magic
import os
import shutil


def process_one_video(input_video_path, duration = 30.0):
    start_frame = 0
    found = 0

    print('Configuration:')
    print('  input video path: %s' % (input_video_path))
    print('  duration: %s' % (duration))

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
        return 0, fv.fps

    for segment in segments:
        print('  segment: start: %d (%.3f), end: %d (%.3f)' % (segment[0], segment[0] / fv.fps, segment[1], segment[1] / fv.fps))

    print('Best fit:')
    for n in range(90, 30, -1):
        for segment in segments:
            frames = fv.find_frames_eye_width_threshold(segment[0], segment[1], n, duration * fv.fps)

            # try next segment
            if len(frames) == 0:
                continue

            for frame in frames:
                print('  threshold: %d, start: %d (%.3f), end: %d (%.3f)' % (n, frame[0], frame[0] / fv.fps, frame[1], frame[1] / fv.fps))
                found = 1
                if start_frame == 0:
                    start_frame = frame[0]

            if found != 0:
                segments.remove(segment)
                found = 0

    if start_frame == 0:
        print('  none')

    return start_frame, fv.fps

def process_training_csv(csv_path, update):
    csv_fields = ['file_name', 'start_time', 'duration', 'pd_stage']

    _, filename = os.path.split(csv_path)
    print('process_training_csv: %s' % (filename))

    # update the record in csv file if found
    tempfile = NamedTemporaryFile(mode = 'w', delete = False)

    with open(csv_path, 'r') as csv_file, tempfile:
        csv_reader = csv.DictReader(csv_file, fieldnames = csv_fields)
        #csv_reader = csv.DictReader(csv_file)
        csv_writer = csv.DictWriter(tempfile, fieldnames = csv_fields)
        for row in csv_reader:
            video_path = row['file_name']
            if video_path == 'file_name':
                # copy the field names
                csv_writer.writerow(row)
                continue

            duration = float(row['duration'])

            start_frame, fps = process_one_video(video_path, duration)
            if start_frame != 0 and update != False:
                row['start_time'] = '%.3f' % ((start_frame + 1) / fps)

            # copy the rows
            csv_writer.writerow(row)

    shutil.move(tempfile.name, csv_path)

    print('  success')
    return True

def main():
    update = False

    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help = 'path to a video file or a training recipe file')

    parser.add_argument('-d', '--duration', help = 'duration (sec)')
    parser.add_argument("-u", "--update", action = "count", default = 0, help = 'update the recipe')

    args = parser.parse_args()

    input_video_path = args.path

    if args.update != 0:
        update = True

    print('User input:')
    print('  input path: %s' % (input_video_path))
    print('  update csv: %s' % (str(update)))

    if args.duration != None:
        duration = float(args.duration)
        print('  duration:   %.3f' % (duration))
    else:
        duration = 30.0

    _, ext = os.path.splitext(input_video_path)

    if ext == '.csv':
        if args.duration != None:
            print('  ignore duration')

        # could be a training recipe
        ret = process_training_csv(input_video_path, update)

    elif os.path.isfile(input_video_path) != False:
        mime = magic.Magic(mime=True)

        file_mine = mime.from_file(input_video_path)
        if file_mine.find('video') == -1:
            print('  not a video file')
            return False

        start_frame, fps = process_one_video(input_video_path, duration)

    else:
        print('Unrecognized path')

    return

if __name__ == '__main__':
    main()
