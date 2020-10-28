#!/usr/bin/python3
from tempfile import NamedTemporaryFile
import csv
import os
import shutil


class FacialRecipe:
    csv_fields = ['blink', 'm2e', 'date', 'pid', 'type', 'start_frame', 'end_frame', 'duration', 'width_diff', 'data_blink', 'data_eh', 'data_m2e', 'pd_stage']

    def __init__(self, recipe_path):
        self.__init = False

        if os.path.isfile(recipe_path) == False:
            # should not get here
            print('fr: recipe file not exist')
            return

        self.recipe_path = recipe_path

        self.temp_file = NamedTemporaryFile(mode = 'w', delete = False)
        self.recipe_file = open(self.recipe_path, 'r', newline = '')

        self.csv_reader = csv.DictReader(self.recipe_file)
        self.csv_writer = csv.DictWriter(self.temp_file, fieldnames = self.csv_fields)
        self.csv_writer.writeheader()

        self.__init = True
        self.__csv_data = False
        return

    def __del__(self):
        if self.__init == False:
            return

        if self.__csv_data != False:
            # update the row
            self.csv_writer.writerow(self.csv_row)

        # flush all rows
        while True:
            # read the next row
            try:
                self.csv_row = next(self.csv_reader)
            except StopIteration:
                break

            self.csv_writer.writerow(self.csv_row)

        self.temp_file.close()
        self.recipe_file.close()

        shutil.move(self.temp_file.name, self.recipe_path)

    def init(self):
        return self.__init

    def read_next(self):
        ret = True

        if self.__csv_data != False:
            # update the row
            self.csv_writer.writerow(self.csv_row)

        # read the next row
        try:
            self.csv_row = next(self.csv_reader)
        except StopIteration:
            ret = False

        self.__csv_data = ret

        return ret

    def get_file_path(self):
        if self.__csv_data == False:
            return ''

        file_path = '/media/Temp_AIpose%s/SJCAM/%s_%s%s.mp4' % (self.csv_row['date'], self.csv_row['date'], self.csv_row['pid'], self.csv_row['type'])
        return file_path

    def reset_data_fields(self):
        if self.__csv_data == False:
            return

        self.csv_row['data_blink'] = '0'
        self.csv_row['data_eh'] = '0'
        self.csv_row['data_m2e'] = '0'


    # standard get/set functions
    def get_start_frame(self):
        if self.__csv_data == False:
            return 0

        return int(self.csv_row['start_frame'])

    def set_start_frame(self, start_frame):
        if self.__csv_data == False:
            return

        self.csv_row['start_frame'] = str(start_frame)

    def set_end_frame(self, end_frame):
        if self.__csv_data == False:
            return

        self.csv_row['end_frame'] = str(end_frame)

    def get_duration(self):
        if self.__csv_data == False:
            return 0.0

        return float(self.csv_row['duration'])

    def set_duration(self, duration):
        if self.__csv_data == False:
            return

        self.csv_row['duration'] = str(duration)

    def set_width_diff(self, width_diff):
        if self.__csv_data == False:
            return

        self.csv_row['width_diff'] = str(width_diff)

    def set_data_blink(self, data_blink):
        if self.__csv_data == False:
            return

        self.csv_row['data_blink'] = str(data_blink)

    def set_data_eh(self, data_eh):
        if self.__csv_data == False:
            return

        self.csv_row['data_eh'] = str(data_eh)

    def set_data_m2e(self, data_m2e):
        if self.__csv_data == False:
            return

        self.csv_row['data_m2e'] = str(data_m2e)

