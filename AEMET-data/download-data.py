#!/drives/d/Installed-programs/python-3.10.6/python

from do_request import do_request
import datetime
import time
import os

# Configuration
output_directory = 'downloaded_historical_climate_val/'


start_date = datetime.date(2022, 9, 18)
end_date = datetime.datetime.now().date()

custom_df = "%Y-%m-%dT%H:%M:%SUTC"

all_data = {'climate_values':
            f'/valores/climatologicos/diarios/datos/fechaini/#START_DATE#/fechafin/#END_DATE#/todasestaciones'}

while start_date <= end_date:
    start_date_str = start_date.strftime(custom_df)
    end_date_str = (start_date + datetime.timedelta(days=1)).strftime(custom_df)
    curr_all_data = all_data.copy()
    curr_all_data['climate_values'] = curr_all_data['climate_values'].replace('#START_DATE#', start_date_str)
    curr_all_data['climate_values'] = curr_all_data['climate_values'].replace('#END_DATE#', end_date_str)

    for file_name, endpoint in curr_all_data.items():
        output_file_name = f'{output_directory}{file_name}_{start_date_str.replace(":","-")}.json'
        print(f'Downloading "{file_name}"...')
        #check if file already exists
        if not os.path.exists(output_file_name):
            do_request(endpoint, output_file_name)
            time.sleep(0.5)

    start_date += datetime.timedelta(days=1)
