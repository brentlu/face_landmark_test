#!/usr/bin/python3

from facial_recipe import FacialRecipe
from facial_video import FacialVideo
import argparse
import magic
import os


def process_one_video(input_video_path, min_duration = 30.0, out_duration = 0.0):
    start_frame = 0
    end_frame = 0
    width_diff = 0

    best_found = False
    ret = False

    print('Process video:')
    print('  input video path: %s' % (input_video_path))
    print('  min_duration: %s' % (min_duration))
    print('  out_duration: %s' % (out_duration))

    fv = FacialVideo(input_video_path)

    if fv.init() == False:
        print('  fail to init engine')
        return ret, (start_frame, end_frame, width_diff)

    print('Statistic data:')

    ret = fv.update_statistic_data(1, fv.frame_count)
    if ret == False:
        print('  fail')
        return ret, (start_frame, end_frame, width_diff)

    ear_min = fv.get_eye_aspect_ratio(fv.MIN)
    ear_avg = fv.get_eye_aspect_ratio(fv.AVG)
    ear_max = fv.get_eye_aspect_ratio(fv.MAX)

    ew_min = fv.get_eye_width(fv.MIN)
    ew_avg = fv.get_eye_width(fv.AVG)
    ew_max = fv.get_eye_width(fv.MAX)

    print('  eye aspect ratio(left):  min %.3f, avg %.3f, max %.3f' % (ear_min[fv.LEFT_EYE], ear_avg[fv.LEFT_EYE], ear_max[fv.LEFT_EYE]))
    print('  eye aspect ratio(right): min %.3f, avg %.3f, max %.3f' % (ear_min[fv.RIGHT_EYE], ear_avg[fv.RIGHT_EYE], ear_max[fv.RIGHT_EYE]))
    print('  eye width(left):         min %.3f, avg %.3f, max %.3f' % (ew_min[fv.LEFT_EYE], ew_avg[fv.LEFT_EYE], ew_max[fv.LEFT_EYE]))
    print('  eye width(right):        min %.3f, avg %.3f, max %.3f' % (ew_min[fv.RIGHT_EYE], ew_avg[fv.RIGHT_EYE], ew_max[fv.RIGHT_EYE]))

    print('Continuous frames:')

    segments = fv.find_continuous_frames(min_duration * fv.fps)

    if len(segments) == 0:
        print('  none')
        return ret, (start_frame, end_frame, width_diff)

    for segment in segments:
        print('  segment: start: %d (%.3f), end: %d (%.3f)' % (segment[0], segment[0] / fv.fps, segment[1], segment[1] / fv.fps))

    print('Best fit for each segment:')
    for n in range(1, 50):
        for segment in segments:
            frames = fv.find_front_face_frames(segment[0], segment[1], n, min_duration * fv.fps)

            # try next segment
            if len(frames) == 0:
                continue

            for frame in frames:
                print('  eye width diff(%%): %d, start: %d (%.3f), end: %d (%.3f)' % (n, frame[0], frame[0] / fv.fps, frame[1], frame[1] / fv.fps))
                if best_found == False:
                    best_found = True

                    start_frame = frame[0]

                    if out_duration == 0.0:
                        end_frame = frame[1]
                    else:
                        end_frame = int(frame[0] + (out_duration * fv.fps) - 1)

                    width_diff = n

            segments.remove(segment)

    if best_found == False:
        print('  none')
    else:
        ret = True

    return ret, (start_frame, end_frame, width_diff)

def process_training_csv(csv_path, min_duration, overwrite, use_all):

    _, filename = os.path.split(csv_path)
    print('Process training csv: %s' % (filename))
    print('  overwrite: %s' % (str(overwrite)))
    print('  use_all: %s' % (str(use_all)))

    fr = FacialRecipe(csv_path)

    if fr.init() == False:
        print('  fail to init recipe')
        return False

    while fr.read_next() != False:
        if fr.get_start_frame() != 0 and overwrite == False:
            continue

        file_path = fr.get_file_path()

        if min_duration == 0.0:
            min_duration = fr.get_duration()
        else:
            fr.set_duration(min_duration)

        out_duration = min_duration
        if use_all != False:
            out_duration = 0.0

        ret, (start_frame, end_frame, width_diff) = process_one_video(file_path, min_duration, out_duration)
        if ret == False:
            start_frame = 0
            end_frame = 0
            width_diff = 0

        fr.set_start_frame(start_frame)
        fr.set_end_frame(end_frame)
        fr.set_width_diff(width_diff)

        # reset the data field
        fr.reset_data_fields()

    print('  success')
    return True

def main():
    min_duration = 0.0
    overwrite = False
    use_all = False

    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help = 'path to a video file or a training recipe file')

    parser.add_argument('-d', '--min_duration', help = 'minimum duration (sec)')
    parser.add_argument("-o", "--overwrite", action = "count", default = 0, help = 'overwrite the recipe')
    parser.add_argument("-a", "--use_all", action = "count", default = 0, help = 'use all available frames')

    args = parser.parse_args()

    input_path = args.input_path

    if args.min_duration != None:
        min_duration = float(args.min_duration)

    if args.overwrite != 0:
        overwrite = True

    if args.use_all != 0:
        use_all = True

    print('User input:')
    print('  input path: %s' % (input_path))
    print('  minimum duration: %s' % (str(min_duration)))
    print('  overwrite: %s' % (str(overwrite)))
    print('  use all: %s' % (str(use_all)))

    _, ext = os.path.splitext(input_path)

    if ext == '.csv':
        if args.min_duration != None:
            print('  ignore min_duration')

        # could be a training recipe
        ret = process_training_csv(input_path, min_duration, overwrite, use_all)

    elif os.path.isfile(input_path) != False:
        mime = magic.Magic(mime=True)

        file_mine = mime.from_file(input_path)
        if file_mine.find('video') == -1:
            print('  not a video file')
            return False

        if args.min_duration == None:
            min_duration = 30.0

        out_duration = min_duration
        if use_all != False:
            out_duration = 0.0

        ret, _ = process_one_video(input_path, min_duration, out_duration)

    else:
        print('Unrecognized path')

    return

if __name__ == '__main__':
    main()
