#!/usr/bin/python3

from facial_video import FacialVideo
import argparse
import csv
import os

def process_one_video(file_name, start_frame, end_frame, pd_stage, csv_writer):
    csv_row = []
    ret = True

    _, name = os.path.split(file_name)
    print('process_one_video: %s' % (name))

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
            ear = fv.get_eye_aspect_ratio()
            ear_string = '%.3f' % (ear[fv.LEFT_EYE])

            csv_row.append(ear_string)
        else:
            # should not happen
            print('  no landmarks, index %d' %(frame_index))
            ret = False

    csv_row.append(str(pd_stage))

    csv_writer.writerow(csv_row)

    return ret

def main():
    dataset_path = 'test.csv'
    parser = argparse.ArgumentParser()
    parser.add_argument('recipe', help = 'path to a training recipe file')

    args = parser.parse_args()

    input_csv_path = args.recipe

    print('Input recipe = %s' % (input_csv_path))

    _, ext = os.path.splitext(input_csv_path)

    if ext != '.csv':
        print('  not a csv file')
        return False

    csv_file_write = open(dataset_path, 'w', newline='')
    csv_writer = csv.writer(csv_file_write)

    with open(input_csv_path, 'r', newline = '') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            file_name = row['file_name']
            if file_name == 'file_name':
                continue

            start_frame = int(row['start_frame'])
            end_frame = int(row['end_frame'])
            pd_stage = int(row['pd_stage'])

            # update the csv if found
            ret = process_one_video(file_name, start_frame, end_frame, pd_stage, csv_writer)
            if ret == False:
                print('  fail to process video')
                break

    csv_file_write.close()

    return

if __name__ == '__main__':
    main()
