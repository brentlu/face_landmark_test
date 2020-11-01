#!/usr/bin/python3
from tempfile import NamedTemporaryFile
import csv
import os
import shutil


class FacialRecipe:
    csv_fields = ['blink', 'm2e', 'date', 'pid', 'type', 'start_frame', 'end_frame', 'duration', 'width_diff', 'data_blink', 'data_eh', 'data_ma', 'data_m2e', 'pd_stage']

    def __init__(self, recipe_path, no_update = False):
        self.__init = False
        self.__no_update = no_update

        if os.path.isfile(recipe_path) == False:
            # should not get here
            print('fr: recipe file not exist')
            return

        self.recipe_path = recipe_path

        self.recipe_file = open(self.recipe_path, 'r', newline = '')
        self.csv_reader = csv.DictReader(self.recipe_file)

        if self.__no_update == False:
            self.temp_file = NamedTemporaryFile(mode = 'w', delete = False)
            self.csv_writer = csv.DictWriter(self.temp_file, fieldnames = self.csv_fields)
            self.csv_writer.writeheader()

        self.__init = True
        self.__csv_data = False
        return

    def __del__(self):
        if self.__init == False:
            return

        if self.__no_update == False:
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
        else:
            self.recipe_file.close()

    def init(self):
        return self.__init

    def read_next(self):
        ret = True

        if self.__no_update == False:
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
            print('fr: csv data not available')
            return ''

        file_path = '/media/Temp_AIpose%s/SJCAM/%s_%s%s.mp4' % (self.csv_row['date'], self.csv_row['date'], self.csv_row['pid'], self.csv_row['type'])
        return file_path

    def reset_data_fields(self):
        if self.__csv_data == False:
            print('fr: csv data not available')
            return

        self.csv_row['data_blink'] = '0'
        self.csv_row['data_eh'] = '0'
        self.csv_row['data_m2e'] = '0'

    def find_data_ma(self):
        if self.__csv_data == False:
            print('fr: csv data not available')
            return False, 0.0

        with open(self.recipe_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:

                if row['m2e'] != 'yes':
                    continue
                if row['date'] != self.csv_row['date']:
                    continue
                if row['pid'] != self.csv_row['pid']:
                    continue
                start_frame = int(row['start_frame'])
                if start_frame == 0:
                    continue

                if row['data_ma'] == '':
                    return 0.0

                return True, float(row['data_ma'])

        print("fr: fail to find data_ma");
        return False, 0.0

    def find_data_m2e(self):
        if self.__csv_data == False:
            print('fr: csv data not available')
            return False, 0.0

        with open(self.recipe_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:

                if row['m2e'] != 'yes':
                    continue
                if row['date'] != self.csv_row['date']:
                    continue
                if row['pid'] != self.csv_row['pid']:
                    continue
                start_frame = int(row['start_frame'])
                if start_frame == 0:
                    continue

                if row['data_m2e'] == '':
                    return 0.0

                return True, float(row['data_m2e'])

        print("fr: fail to find data_m2e");
        return False, 0.0

    # standard get/set functions
    def get_blink(self):
        if self.__csv_data == False:
            print('fr: csv data not available')
            return 'no'

        if 'blink' not in self.csv_row:
            return 'no'

        if self.csv_row['blink'] == '':
            return 'no'

        return self.csv_row['blink']

    def get_m2e(self):
        if self.__csv_data == False:
            print('fr: csv data not available')
            return 'no'

        if 'm2e' not in self.csv_row:
            return 'no'

        if self.csv_row['m2e'] == '':
            return 'no'

        return self.csv_row['m2e']

    def get_start_frame(self):
        if self.__csv_data == False:
            print('fr: csv data not available')
            return 0

        if 'start_frame' not in self.csv_row:
            return 0

        if self.csv_row['start_frame'] == '':
            return 0

        return int(self.csv_row['start_frame'])

    def set_start_frame(self, start_frame):
        if self.__csv_data == False:
            print('fr: csv data not available')
            return

        self.csv_row['start_frame'] = str(start_frame)

    def get_end_frame(self):
        if self.__csv_data == False:
            print('fr: csv data not available')
            return 0

        if 'end_frame' not in self.csv_row:
            return 0

        if self.csv_row['end_frame'] == '':
            return 0

        return int(self.csv_row['end_frame'])

    def set_end_frame(self, end_frame):
        if self.__csv_data == False:
            print('fr: csv data not available')
            return

        self.csv_row['end_frame'] = str(end_frame)

    def get_duration(self):
        if self.__csv_data == False:
            print('fr: csv data not available')
            return 0.0

        if 'data_blink' not in self.csv_row:
            return 0.0

        if self.csv_row['duration'] == '':
            return 0.0

        return float(self.csv_row['duration'])

    def set_duration(self, duration):
        if self.__csv_data == False:
            print('fr: csv data not available')
            return

        self.csv_row['duration'] = '%.3f' % (duration)

    def set_width_diff(self, width_diff):
        if self.__csv_data == False:
            print('fr: csv data not available')
            return

        self.csv_row['width_diff'] = str(width_diff)

    def get_data_blink(self):
        if self.__csv_data == False:
            print('fr: csv data not available')
            return 0

        if 'data_blink' not in self.csv_row:
            return 0

        if self.csv_row['data_blink'] == '':
            return 0

        return int(self.csv_row['data_blink'])

    def set_data_blink(self, data_blink):
        if self.__csv_data == False:
            print('fr: csv data not available')
            return

        self.csv_row['data_blink'] = str(data_blink)

    def get_data_eh(self):
        if self.__csv_data == False:
            print('fr: csv data not available')
            return 0.0

        if 'data_eh' not in self.csv_row:
            return 0.0

        if self.csv_row['data_eh'] == '':
            return 0.0

        return float(self.csv_row['data_eh'])

    def set_data_eh(self, data_eh):
        if self.__csv_data == False:
            print('fr: csv data not available')
            return

        self.csv_row['data_eh'] = '%.3f' % (data_eh)

    def get_data_ma(self):
        if self.__csv_data == False:
            print('fr: csv data not available')
            return 0.0

        if 'data_ma' not in self.csv_row:
            return 0.0

        if self.csv_row['data_ma'] == '':
            return 0.0

        return float(self.csv_row['data_ma'])

    def set_data_ma(self, data_ma):
        if self.__csv_data == False:
            print('fr: csv data not available')
            return

        self.csv_row['data_ma'] = '%.3f' % (data_ma)

    def get_data_m2e(self):
        if self.__csv_data == False:
            print('fr: csv data not available')
            return 0.0

        if 'data_m2e' not in self.csv_row:
            return 0.0

        if self.csv_row['data_m2e'] == '':
            return 0.0

        return float(self.csv_row['data_m2e'])

    def set_data_m2e(self, data_m2e):
        if self.__csv_data == False:
            print('fr: csv data not available')
            return

        self.csv_row['data_m2e'] = '%.3f' % (data_m2e)

    def get_pd_stage(self):
        if self.__csv_data == False:
            print('fr: csv data not available')
            return 0

        if 'pd_stage' not in self.csv_row:
            return 0

        if self.csv_row['pd_stage'] == '':
            return 0

        return int(self.csv_row['pd_stage'])
