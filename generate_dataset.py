#!/usr/bin/python3

from facial_video import FacialVideo
import argparse
import csv
import os
import time


def find_m2e_csv_row(csv_path, date, pid):

    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:

            if row['m2e'] != 'yes':
                continue
            if row['date'] != date:
                continue
            if row['pid'] != pid:
                continue
            start_frame = int(row['start_frame'])
            if start_frame == 0:
                continue

            return True, row

    return False, None

def process_training_csv_for_svm(input_path, dataset_path):

    _, filename = os.path.split(input_path)
    print('Process training csv: %s' % (filename))
    print('  dataset path: %s' % (dataset_path))

    with open(dataset_path, 'w', newline = '') as csv_file_write:
        csv_writer = csv.writer(csv_file_write)

        with open(input_path, 'r', newline = '') as csv_file_read:
            csv_reader = csv.DictReader(csv_file_read)
            for row in csv_reader:

                if row['blink'] != 'yes':
                    continue

                start_frame = int(row['start_frame'])
                if start_frame == 0:
                    continue

                if row['m2e'] != 'yes':
                    ret, m2e_row = find_m2e_csv_row(input_path, row['date'], row['pid'])
                    if ret == False:
                        continue

                    dataset_row = [row['data_blink'], m2e_row['data_m2e'], row['pd_stage']]
                else:
                    dataset_row = [row['data_blink'], row['data_m2e'], row['pd_stage']]

                csv_writer.writerow(dataset_row)

    return True

def process_one_video_for_rnn(file_name, start_frame, end_frame, pd_stage, csv_writer):
    frame_index = 0
    csv_row = []
    ret = False

    print('Process video:')
    print('  file name: %s' % (file_name))
    print('  start frame: %s' % (str(start_frame)))
    print('  end frame: %s' % (str(end_frame)))
    print('  pd stage: %s' % (str(pd_stage)))

    fv = FacialVideo(file_name)

    if fv.init() == False:
        print('  fail to init engine')
        return ret

    print('Process frame:')
    while True:
        frame_index += 1

        if frame_index < start_frame:
            # don't decode this frame to speed up
            ret, _ = fv.read(True)
            continue
        elif frame_index > end_frame:
            ret = True
            break
        else:
            # don't decode this frame to speed up
            ret, _ = fv.read(True)

        if ret == False:
            # no frame to process
            break;

        if frame_index != fv.get_frame_index():
            print('  expect frame %d but got %d' %(frame_index, fv.get_frame_index()))
            break

        if fv.available() != False:
            ear = fv.calculate_eye_aspect_ratio()
            ear_string = '%.3f' % (ear[fv.LEFT_EYE])

            print('  frame: %3d, ear: %.3f %.3f' % (frame_index, ear[fv.LEFT_EYE], ear[fv.RIGHT_EYE]), end = '\r')

            csv_row.append(ear_string)
        else:
            # should not happen
            print('  invalid frame %d' %(frame_index))
            break

    if ret != False:
        csv_row.append(str(pd_stage))
        csv_writer.writerow(csv_row)

    return ret

def process_training_csv_for_rnn(input_path, dataset_path):

    _, filename = os.path.split(input_path)
    print('Process training csv: %s' % (filename))
    print('  dataset path: %s' % (dataset_path))

    with open(dataset_path, 'w', newline = '') as csv_file_write:
        csv_writer = csv.writer(csv_file_write)

        with open(input_path, 'r', newline = '') as csv_file_read:
            csv_reader = csv.DictReader(csv_file_read)
            for row in csv_reader:

                if row['blink'] != 'yes':
                    continue

                start_frame = int(row['start_frame'])
                if start_frame == 0:
                    continue

                video_path = '/media/Temp_AIpose%s/SJCAM/%s_%s%s.mp4' % (row['date'], row['date'], row['pid'], row['type'])
                end_frame = int(row['end_frame'])
                pd_stage = int(row['pd_stage'])

                ret = process_one_video_for_rnn(video_path, start_frame, end_frame, pd_stage, csv_writer)
                if ret == False:
                    print('  fail')
                    return False

    return True

def main():

    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_type', help = 'type of output dataset')
    parser.add_argument('input_path', help = 'path to a recipe file')

    args = parser.parse_args()

    dataset_type = args.dataset_type
    input_path = args.input_path

    print('User input:')
    print('  dataset type: %s' % (dataset_type))
    print('  input path: %s' % (input_path))

    _, ext = os.path.splitext(input_path)

    if ext != '.csv':
        print('Unrecognized path')
        return

    # get current time (local time)
    now = time.localtime()
    timestamp = time.strftime('%Y-%m%d-%H%M', now)

    if dataset_type == 'rnn':
        file_name = 'dataset-rnn-%s.csv' % (timestamp)
        dataset_path = os.path.join('.', file_name)
        dataset_path = os.path.abspath(dataset_path)

        ret = process_training_csv_for_rnn(input_path, dataset_path)
    elif dataset_type == 'svm':
        file_name = 'dataset-svm-%s.csv' % (timestamp)
        dataset_path = os.path.join('.', file_name)
        dataset_path = os.path.abspath(dataset_path)

        ret = process_training_csv_for_svm(input_path, dataset_path)
    else:
        print('Unrecognized type')

    print('  success')
    return

if __name__ == '__main__':
    main()
