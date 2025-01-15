import os
import shutil
from datetime import datetime, timedelta
from os.path import abspath, dirname


file_path = abspath(dirname(__file__))
output_path = file_path + '/output_reciente/'
end_date = datetime.strptime('2024-11-17T10-30-00.000Z', "%Y-%m-%dT%H-%M-%S.%fZ")


def format_filename(timestamp):
    # Formatear la fecha con precisión hasta microsegundos (6 dígitos)
    formatted_timestamp = timestamp.strftime("%Y-%m-%dT%H-%M-%S.%f")[:-3] + "Z"
    return f"image_{formatted_timestamp}.png"


def adjust_time_in_filename(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.png')]
    missing_files = 0

    for file in files:
        timestamp_str = file.split('_')[1].rstrip('.png')
        original_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S.%fZ")

        current_timestamp = original_timestamp
        next_timestamp = current_timestamp + timedelta(minutes=15)
        next_filename = format_filename(next_timestamp)

        while not os.path.exists(os.path.join(directory, next_filename)) and next_timestamp <= end_date:
            shutil.copy(os.path.join(directory, format_filename(current_timestamp)), os.path.join(directory, next_filename))
            print(f"Copied {format_filename(current_timestamp)} to {next_filename}")

            current_timestamp = next_timestamp
            next_timestamp = current_timestamp + timedelta(minutes=15)
            next_filename = format_filename(next_timestamp)
            missing_files += 1
    print(f'Total files: {len(files)}; Missing files copied: {missing_files}')


adjust_time_in_filename(output_path)
