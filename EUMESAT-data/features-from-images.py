#!/bin/python
import os
import re
import cv2
import pandas as pd
import numpy as np
from common.dataframe_manager import save_df, load_df

# pandas configuration
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# HSV values for each community
ccaas = {
    "Andalucía": (31, 44, 99),
    "Aragón": (20, 53, 84),
    "Asturias": (341, 66, 79),
    "Cantabria": (12, 59, 82),
    "Castilla La Mancha": (24, 50, 99),
    "Castilla y León": (17, 58, 99),
    "Cataluña": (280, 78, 44),
    "Extremadura": (37, 38, 100),
    "Galicia": (331, 68, 72),
    "Islas Baleares": (350, 63, 86),
    "La Rioja": (14, 84, 65),
    "Madrid": (261, 82, 38),
    "Murcia": (357, 59, 91),
    "Navarra": (8, 58, 95),
    "País Vasco": (288, 75, 47),
    "Valencia": (312, 71, 56)
}
# Tolerance for HSV values
tolerancia_hsv = (3, 3, 3)
# White pixel threshold
white_threshold = 240

# Select folder to extract hist or obs path
features_to_extract = 'hist'
hist_path = './output'
obs_path = './output_reciente'
images_path = hist_path if features_to_extract == 'hist' else obs_path


# Show features clouds in an output image
def export_mask_with_clouds(maks_image, input_gray_img):
    mask_image_with_clouds = maks_image.copy()
    _, white_mask = cv2.threshold(input_gray_img, white_threshold, 255, cv2.THRESH_BINARY)
    if len(mask_image_with_clouds.shape) == 2 or mask_image_with_clouds.shape[2] == 1:
        mask_image_with_clouds = cv2.cvtColor(mask_image_with_clouds, cv2.COLOR_GRAY2BGR)
    if white_mask.shape != mask_image_with_clouds.shape[:2]:
        white_mask = cv2.resize(white_mask,
                                (mask_image_with_clouds.shape[1], mask_image_with_clouds.shape[0]))

    mask_image_with_clouds[white_mask == 255] = [255, 255, 255]

    # Save the image with overlay clouds
    cv2.imwrite(f"{images_path}/mask_with_overlay_clouds.png", mask_image_with_clouds)


# List PNG files in the input directory
def retrieve_files(directory):
    files = os.listdir(directory)
    png_files = [file for file in files if file.endswith('.png')]
    pattern = r'^image_(.*).png$'
    files = {}

    for png_file in png_files:
        match = re.search(pattern, png_file)
        if match:
            date_time = match.group(1)
            files[png_file] = date_time
    return files


# Process png file
def process_image(mask_image, mask_image_hsv, file_name):
    try:
        input_image = cv2.imread(f"{images_path}/{file_name}")
        input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        # Count dictionary for each CCAA
        counts = {ccaa: {"total": 0, "white": 0} for ccaa in ccaas}

        for ccaa, (h, s, v) in ccaas.items():
            h = int(h / 2)
            s = int(s * 2.55)
            v = int(v * 2.55)

            lower_bound = np.array(
                [max(0, h - tolerancia_hsv[0]), max(0, s - tolerancia_hsv[1]), max(0, v - tolerancia_hsv[2])],
                dtype=np.uint8)
            upper_bound = np.array(
                [min(179, h + tolerancia_hsv[0]), min(255, s + tolerancia_hsv[1]), min(255, v + tolerancia_hsv[2])],
                dtype=np.uint8)
            mask_ccaa = cv2.inRange(mask_image_hsv, lower_bound, upper_bound)

            counts[ccaa]["total"] = cv2.countNonZero(mask_ccaa)

            white_ccaa = cv2.bitwise_and(input_image_gray, input_image_gray, mask=mask_ccaa)
            white_mask = np.where(white_ccaa >= white_threshold, 1, 0).astype(np.uint8)
            counts[ccaa]["white"] = cv2.countNonZero(white_mask)

        for ccaa in counts:
            if counts[ccaa]["total"] > 0:
                counts[ccaa]["white_ratio"] = counts[ccaa]["white"] / counts[ccaa]["total"]
            else:
                counts[ccaa]["white_ratio"] = 0

        # Show results
        df = pd.DataFrame(counts).T
        df = df.sort_values(by="white_ratio", ascending=False)

    except Exception as e:
        print(f'Error processing file {file_name}: {e}')
        raise e

    return df


# Main function
def main():
    df_cloud_ratio_ccaa = pd.DataFrame()
    mask_image = cv2.imread("./input/mask.png")
    # Convert the mask image to HSV for color analysis
    mask_image_hsv = cv2.cvtColor(mask_image, cv2.COLOR_BGR2HSV)

    files = retrieve_files(images_path)

    final_df = pd.DataFrame()
    total_files = len(files)
    for idx, file in enumerate(files.keys()):
        df = process_image(mask_image, mask_image_hsv, file)
        new_row = {'file': f'{file}', 'date_time': f'{files[file]}'}

        ratio_dict = df['white_ratio'].to_dict()
        final_dict = new_row.copy()
        final_dict.update(ratio_dict)

        final_df = pd.concat([final_df, pd.DataFrame([final_dict])], ignore_index=True)
        if idx % 1000 == 0:
            print(f'{idx}/{total_files} files processed...')

    save_df(final_df, f'./extracted-features/cloudiness_by_region_{features_to_extract}.json')

    print(load_df(f'./extracted-features/cloudiness_by_region_{features_to_extract}.json'))


if __name__ == '__main__':
    main()
