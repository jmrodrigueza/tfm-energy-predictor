#!/bin/python

import requests
import datetime
import time
import random
import os
from datetime import timedelta

# Configuration
# Time configuration
start_time = datetime.datetime(2022, 9, 18, 0, 0)
end_time = datetime.datetime(2024, 9, 27, 4, 0)
time_interval = 15
min_delay = 3
max_delay = 6
# Spain coordinates ranges
max_lat = 44.0
min_lat = 35.5
max_lon = 5.0
min_lon = -9.5
# URL configuration
output_dir = './output'
url = f"https://view.eumetsat.int/geoserver/ows?service=WMS&request=GetMap&version=1.3.0&" \
      f"layers=msg_fes:clm&styles=&format=image/png&crs=EPSG:4326&bbox={min_lat},{min_lon},{max_lat},{max_lon}&" \
      f"width=800&height=600&time=#time_str#"

# Headers que simulan una solicitud desde un navegador web
headers = {
    'Accept': 'image/png,image/*;q=0.8,*/*;q=0.5',
}


# Function to download images
def download_images():
    # Asegúrate de que el directorio de salida existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    current_time = start_time
    while current_time <= end_time:
        current_url = url

        time_str = current_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        safe_time_str = time_str.replace(':', '-')
        current_url = current_url.replace('#time_str#', time_str)
        print("Downloading image...")
        print(current_url)

        response = requests.get(current_url, headers=headers)
        if response.status_code == 200:
            with open(f'{output_dir}/image_{safe_time_str}.png', 'wb') as file:
                file.write(response.content)
            print(f'Image downloaded for time: {time_str}')
        else:
            print(f'Failed to download image for time: {time_str}')

        current_time += timedelta(minutes=time_interval)

        # random wait time
        delay = random.uniform(min_delay, max_delay)
        print(f'Waiting {delay:.2f} seconds before the next download.')
        time.sleep(delay)


if __name__ == '__main__':
    download_images()
