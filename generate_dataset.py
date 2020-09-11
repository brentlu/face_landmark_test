#!/usr/bin/python3

from facial_video import FacialVideo
import argparse
import csv
import os


def process_one_video(file_name, start_frame, end_frame, pd_stage, csv_writer):
    csv_row = []
    ret = True

    print('Process video:')
    print('  file name: %s' % (file_name))
    print('  start frame: %s' % (str(start_frame)))
    print('  end frame: %s' % (str(end_frame)))
    print('  pd stage: %s' % (str(pd_stage)))

    fv = FacialVideo(file_name)

    while True:
        ret, _ = fv.read()
        if ret == False:
            # no frame to process
            break;

        frame_index = fv.get_frame_index()
        if frame_index < start_frame:
            continue
        elif frame_index > end_frame:
            break

        if fv.available() != False:
            ear = fv.calculate_eye_aspect_ratio()
            ear_string = '%.3f' % (ear[fv.LEFT_EYE])

            print('  frame: %3d, ear: %.3f %.3f' % (frame_index, ear[fv.LEFT_EYE], ear[fv.RIGHT_EYE]), end = '\r')


            csv_row.append(ear_string)
        else:
            # should not happen
            print('  invalid frame index %d' %(frame_index))
            ret = False

    if ret != False:
        csv_row.append(str(pd_stage))
        csv_writer.writerow(csv_row)

    return ret

def main():
    dataset_path = 'test.csv'

    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help = 'path to a training recipe file')

    args = parser.parse_args()

    input_path = args.input_path

    print('User input:')
    print('  input path: %s' % (input_path))

    _, ext = os.path.splitext(input_path)

    if ext != '.csv':
        print('Unrecognized path')
        return False

    csv_file_write = open(dataset_path, 'w', newline='')
    csv_writer = csv.writer(csv_file_write)

    _, filename = os.path.split(input_path)
    print('Process training csv: %s' % (filename))

    with open(input_path, 'r', newline = '') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            file_name = row['file_name']
            start_frame = int(row['start_frame'])
            end_frame = int(row['end_frame'])
            pd_stage = int(row['pd_stage'])

            ret = process_one_video(file_name, start_frame, end_frame, pd_stage, csv_writer)
            if ret == False:
                csv_file_write.close()
                print('  fail')
                return

    csv_file_write.close()

    print('  success')
    return

if __name__ == '__main__':
    main()
