#!/usr/bin/python

from do_request import do_request
from datetime import datetime
import time
import os

current_base_path = os.path.dirname(os.path.abspath(__file__))

# Configuration
output_directory = current_base_path + '/downloaded/'
custom_df = "%Y-%m-%d_%H-%M-%S"
current_date_hour = datetime.now()
date_time_str = current_date_hour.strftime(custom_df)

all_data = {date_time_str + '_observation_data':
                f'/observacion/convencional/todas'}

max_retries = 3
retry = 0
success = False
sleep_time = 30
while not success and retry < max_retries:
    try:
        for file_name, endpoint in all_data.items():
            print(f'Downloading "{file_name}"...')
            do_request(endpoint, f'{output_directory}{file_name}.json')
            time.sleep(0.5)
        success = True
    except Exception as ex:
        print(f'An exception has occurred! in the try number {retry}: ', ex)
        time.sleep(sleep_time)
    finally:
        retry += 1

print(f'Done in {retry} tries.')
