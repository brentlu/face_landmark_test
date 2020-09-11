#!/usr/bin/python3

from facial_video import FacialVideo
from tempfile import NamedTemporaryFile
import argparse
import csv
import magic
import os
import shutil


def process_one_video(input_video_path, min_duration = 30.0, out_duration = 0.0):
    start_frame = 0
    end_frame = 0
    found = 0

    print('Configuration:')
    print('  input video path: %s' % (input_video_path))
    print('  min_duration: %s' % (min_duration))
    print('  out_duration: %s' % (out_duration))

    fv = FacialVideo(input_video_path)

    ret = fv.update_static_data(1, fv.frame_count)
    if ret != False:
        print('  eye aspect ratio(left):  min %.3f, avg %.3f, max %.3f' % (fv.eye_aspect_ratio[fv.LEFT_EYE][fv.MIN], fv.eye_aspect_ratio[fv.LEFT_EYE][fv.AVG], fv.eye_aspect_ratio[fv.LEFT_EYE][fv.MAX]))
        print('  eye aspect ratio(right): min %.3f, avg %.3f, max %.3f' % (fv.eye_aspect_ratio[fv.RIGHT_EYE][fv.MIN], fv.eye_aspect_ratio[fv.RIGHT_EYE][fv.AVG], fv.eye_aspect_ratio[fv.RIGHT_EYE][fv.MAX]))
        print('  eye width(left):         min %.3f, avg %.3f, max %.3f' % (fv.eye_width[fv.LEFT_EYE][fv.MIN], fv.eye_width[fv.LEFT_EYE][fv.AVG], fv.eye_width[fv.LEFT_EYE][fv.MAX]))
        print('  eye width(right):        min %.3f, avg %.3f, max %.3f' % (fv.eye_width[fv.RIGHT_EYE][fv.MIN], fv.eye_width[fv.RIGHT_EYE][fv.AVG], fv.eye_width[fv.RIGHT_EYE][fv.MAX]))

    print('Continuous frame:')

    segments = fv.find_frames_continuous(min_duration * fv.fps)

    if len(segments) == 0:
        print('  none')
        return 0, 0

    for segment in segments:
        print('  segment: start: %d (%.3f), end: %d (%.3f)' % (segment[0], segment[0] / fv.fps, segment[1], segment[1] / fv.fps))

    print('Best fit for each segment:')
    for n in range(90, 30, -1):
        for segment in segments:
            frames = fv.find_frames_eye_width_threshold(segment[0], segment[1], n, min_duration * fv.fps)

            # try next segment
            if len(frames) == 0:
                continue

            for frame in frames:
                print('  eye width: %d, start: %d (%.3f), end: %d (%.3f)' % (n, frame[0], frame[0] / fv.fps, frame[1], frame[1] / fv.fps))
                found = 1
                if start_frame == 0:
                    start_frame = frame[0]
                if end_frame == 0:
                    if out_duration == 0.0:
                        end_frame = frame[1]
                    else:
                        end_frame = int(frame[0] + (out_duration * fv.fps) - 1)

            if found != 0:
                segments.remove(segment)
                found = 0

    if start_frame == 0:
        print('  none')

    return start_frame, end_frame

def process_training_csv(csv_path, update_csv, use_all):
    csv_fields = ['file_name', 'start_frame', 'end_frame', 'min_duration', 'pd_stage']

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

            min_duration = float(row['min_duration'])

            out_duration = min_duration
            if use_all != False:
                out_duration = 0.0

            start_frame, end_frame = process_one_video(video_path, min_duration, out_duration)
            if update_csv != False:
                if start_frame != 0:
                    row['start_frame'] = str(start_frame)
                if end_frame != 0:
                    row['end_frame'] = str(end_frame)

            # copy the rows
            csv_writer.writerow(row)

    shutil.move(tempfile.name, csv_path)

    print('  success')
    return True

def main():
    min_duration = 30.0
    update_csv = False
    use_all = False

    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help = 'path to a video file or a training recipe file')

    parser.add_argument('-d', '--min_duration', help = 'minimum duration (sec)')
    parser.add_argument("-u", "--update_csv", action = "count", default = 0, help = 'update the recipe')
    parser.add_argument("-a", "--use_all", action = "count", default = 0, help = 'use all available frames')

    args = parser.parse_args()

    input_video_path = args.path

    if args.min_duration != None:
        min_duration = float(args.min_duration)

    if args.update_csv != 0:
        update_csv = True

    if args.use_all != 0:
        use_all = True

    print('User input:')
    print('  input video path: %s' % (input_video_path))
    print('  minimum duration: %s' % (str(min_duration)))
    print('  update csv: %s' % (str(update_csv)))
    print('  use all: %s' % (str(use_all)))

    _, ext = os.path.splitext(input_video_path)

    if ext == '.csv':
        if args.min_duration != None:
            print('  ignore min_duration')

        # could be a training recipe
        ret = process_training_csv(input_video_path, update_csv, use_all)

    elif os.path.isfile(input_video_path) != False:
        mime = magic.Magic(mime=True)

        file_mine = mime.from_file(input_video_path)
        if file_mine.find('video') == -1:
            print('  not a video file')
            return False

        out_duration = min_duration
        if use_all != False:
            out_duration = 0.0

        start_frame, end_frame = process_one_video(input_video_path, min_duration, out_duration)

    else:
        print('Unrecognized path')

    return

if __name__ == '__main__':
    main()
