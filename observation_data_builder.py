#!/usr/bin/python
import os
import re

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from common import dataframe_manager as dm
from common import region_manager as rm
from common.dataframe_manager import aemet_observation_data_path, aemet_historical_data_path


# COLUMNS UNLIMITED IN PANDAS
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# check if it is a valid hour
def is_valid_hour(input_hour):
    # Function to check if a time string is valid (in HH:MM format, between 00:00 and 23:59)
    pattern = r'^(?:[01]\d|2[0-3]):[0-5]\d$'
    return bool(re.match(pattern, input_hour))


# Wrap hour
def wrap_hour(input_hour):
    hour = pd.NA if str(input_hour) in ['nan', 'Varias', ''] \
        else (f'{str(input_hour).zfill(2)}:00' if len(str(input_hour)) <= 2 else str(input_hour))
    # hour 24:00 is equivalent to 00:00
    hour = hour.replace('24:00', '00:00') if hour is not pd.NA else pd.NA

    return hour if hour is not pd.NA and is_valid_hour(hour) else pd.NA


# Wrap columns dataframe to suitable format
def wrap_historical_columns(input_df: pd.DataFrame):
    df_wrapped = input_df.copy()
    df_wrapped['fecha'] = pd.to_datetime(df_wrapped['fecha'], format='%Y-%m-%d')

    # Temperatures
    df_wrapped['horatmin'] = df_wrapped['horatmin'].str.replace('24:00', '00:00')
    df_wrapped['horatmin'] = df_wrapped['horatmin'].str.replace('Varias', '00:00')
    df_wrapped['horatmin'] = pd.to_datetime(df_wrapped['horatmin'], format='%H:%M')
    df_wrapped['horatmax'] = df_wrapped['horatmax'].str.replace('Varias', '00:00').apply(wrap_hour)
    df_wrapped['horatmax'] = pd.to_datetime(df_wrapped['horatmax'], format='%H:%M')
    df_wrapped['tmed'] = df_wrapped['tmed'].str.replace(',', '.').astype(float)
    df_wrapped['tmin'] = df_wrapped['tmin'].str.replace(',', '.').astype(float)
    df_wrapped['tmax'] = df_wrapped['tmax'].str.replace(',', '.').astype(float)

    # Precipitation
    df_wrapped['prec'] = df_wrapped['prec'].str.replace(',', '.')
    # Ip code is 'Inapreciable' less than (0.1 mm).
    # Reference: https://www.aemet.es/es/eltiempo/prediccion/espana/ayuda
    df_wrapped['prec'] = df_wrapped['prec'].str.replace('Ip', '0').replace('Acum', 0).astype(float)

    # Wind
    df_wrapped['velmedia'] = df_wrapped['velmedia'].str.replace(',', '.').astype(float)
    df_wrapped['racha'] = df_wrapped['racha'].str.replace(',', '.').astype(float)
    df_wrapped['horaracha'] = df_wrapped['horaracha'].str.replace('Varias', '00:00').apply(wrap_hour)
    df_wrapped['horaracha'] = pd.to_datetime(df_wrapped['horaracha'], format='%H:%M')

    # Atmospheric pressure
    df_wrapped['presMax'] = df_wrapped['presMax'].str.replace(',', '.').astype(float)
    df_wrapped['presMin'] = df_wrapped['presMin'].str.replace(',', '.').astype(float)
    df_wrapped['horaPresMax'] = df_wrapped['horaPresMax'].str.replace('Varias', '0')
    df_wrapped['horaPresMax'] = df_wrapped['horaPresMax'].apply(wrap_hour)
    df_wrapped['horaPresMax'] = df_wrapped['horaPresMax'].str.replace('24:00', '00:00')
    df_wrapped['horaPresMax'] = pd.to_datetime(df_wrapped['horaPresMax'], format='%H:%M')
    df_wrapped['horaPresMin'] = df_wrapped['horaPresMin'].apply(wrap_hour)
    df_wrapped['horaPresMin'] = pd.to_datetime(df_wrapped['horaPresMin'], format='%H:%M')

    # Relative humidity
    df_wrapped['horaHrMax'] = df_wrapped['horaHrMax'].apply(wrap_hour)
    df_wrapped['horaHrMax'] = pd.to_datetime(df_wrapped['horaHrMax'], format='%H:%M')
    df_wrapped['horaHrMin'] = df_wrapped['horaHrMin'].apply(wrap_hour)
    df_wrapped['horaHrMin'] = pd.to_datetime(df_wrapped['horaHrMin'], format='%H:%M')

    # Insolation
    df_wrapped['sol'] = df_wrapped['sol'].str.replace(',', '.').astype(float)

    return df_wrapped


def load_and_generate_csv_hist_data():
    files = sorted(os.listdir(aemet_historical_data_path))
    json_files = [file for file in files if
                  file.startswith('climate_values_') and
                  file.endswith('.json') and
                  not re.search('metadata', file)]
    df = pd.DataFrame()
    for file in json_files:
        print(file)
        df_loaded = dm.load_df(aemet_historical_data_path + '/' + file,
                               orient='columns', encoding='windows-1252')
        df = pd.concat([df, wrap_historical_columns(df_loaded)], ignore_index=True)

    df = df.drop_duplicates(subset=['fecha', 'indicativo'], keep='first')
    df.reset_index(drop=True, inplace=True)

    print(df)
    print(df.info())
    print(df.describe())
    # Save the DataFrame to a JSON file
    dm.save_df_to_csv(df, aemet_historical_data_path + '/' + 'full_climate_values_2022-09-18_to_2024-11-04.csv')


# Load and generate csv historical data
def wrap_observation_columns(input_df: pd.DataFrame):
    df_wrapped = input_df.copy()
    df_wrapped['fint'] = pd.to_datetime(df_wrapped['fint'], format='%Y-%m-%dT%H:%M:%S%z')

    df_wrapped = df_wrapped[
        ['fint', 'idema', 'lon', 'lat', 'alt', 'ubi', 'ta', 'tamin', 'tamax', 'prec', 'hr', 'vv', 'pres']]
    return df_wrapped


# Load and generate csv observation data
def load_and_generate_csv_obs_data(filename='full_climate_values_2022-09-18_to_2024-11-04.csv'):
    files = sorted(os.listdir(aemet_observation_data_path))
    json_files = [file for file in files if
                  re.search('.*_observation_data.json', file)]
    df = pd.DataFrame()
    for file in json_files:
        print(file)
        df_loaded = dm.load_df(aemet_observation_data_path + '/' + file,
                               orient='columns', encoding='windows-1252')
        df = pd.concat([df, wrap_observation_columns(df_loaded)], ignore_index=True)

    df = df.drop_duplicates(subset=['fint', 'idema'], keep='first')
    df.reset_index(drop=True, inplace=True)

    print(df.head(n=200))
    print(df.info())
    print(df.describe())
    dm.save_df_to_csv(df, aemet_observation_data_path + f'/{filename}')


# Impute missing values to default (-1000)
def impute_with_fixed_value(df, target_column, impute_value=-1000):
    df_return = df.copy()
    categorical_target_types = ['int64', 'bool']
    if impute_value is None:
        if df[target_column].dtype in categorical_target_types:
            val_to_impute = df_return.groupby(['Año', 'Mes', 'Día', 'Hora'])[target_column].median()
        else:
            val_to_impute = df_return.groupby(['Año', 'Mes', 'Día', 'Hora'])[target_column].mean()
        df_return[target_column] = df_return.apply(
            lambda row: val_to_impute[row['Año'], row['Mes'], row['Día'], row['Hora']]
            if pd.isna(row[target_column]) or row[target_column] is None else row[target_column], axis=1)
    else:
        df_return[target_column] = df[target_column].apply(lambda val: impute_value if pd.isna(val) else val)
    return df_return


# Impute missing values with mean values or -1000
def impute_values(df: pd.DataFrame, column: str, fixed_value: int = None):
    print('Imputing values for column: ', column)
    df_return = df.sort_values(by=['Id_estación', 'fint'])

    if column == 'Presión':
        df_return = impute_pressure_values(df)
    else:
        mean_values = df_return.groupby(['Año', 'Mes', 'Día', 'Hora', 'Provincia'])[column].mean()
        # Impute mean values for missing values
        df_return[column] = df_return.apply(
            lambda row: mean_values[row['Año'], row['Mes'], row['Día'], row['Hora'], row['Provincia']]
            if pd.isna(row[column]) or row[column] is None else row[column], axis=1)

    # Rest of the missing values are imputed with -1000
    df_return = impute_with_fixed_value(df_return, column, impute_value=fixed_value)

    return df_return


# Add Province column to df from Id_estación
def add_province_column(df: pd.DataFrame):
    df_return = df.copy()
    df_return['Provincia'] = df_return['Id_estación'].apply(lambda id_station: rm.retrieve_province(id_station))
    return df_return


# Impute pressure values with mean values by instant, Province and altitude for nan values
# Pressure decrease with altitude by 1hPa every 8 meters. Adjust pressure values by altitude
# and restore pressure values by altitude
def impute_pressure_values(df: pd.DataFrame):
    df_return = df.copy()
    df_return['Imputed'] = df_return['Presión'].isna()
    # Apply pressure decrease by altitude only for Imputed values, as if all stations were at sea level.
    df_return['Presión'] = df_return.apply(lambda row: row['Presión'] - (row['Altitud'] / 8), axis=1)
    # Impute mean values for missing values only for Imputed values
    mean_values = df_return.groupby(['Año', 'Mes', 'Día', 'Hora', 'Provincia'])['Presión'].mean()
    df_return['Presión'] = df_return.apply(
        lambda row: mean_values[row['Año'], row['Mes'], row['Día'], row['Hora'], row['Provincia']] if row[
            'Imputed'] else row['Presión'], axis=1)
    # Restore pressure values by altitude
    df_return['Presión'] = df_return.apply(lambda row: row['Presión'] + (row['Altitud'] / 8), axis=1)
    df_return = df_return.drop(columns=['Imputed'])

    return df_return


def load_csv_obs_data(filename='full_observation_values_2024-09-27_to_2024-11-07.csv'):
    df = dm.load_df_from_csv(aemet_observation_data_path + f'/{filename}')

    # Count stations with so much missing values
    count_res = df.groupby('idema').size().reset_index(name='count')
    count_ordered = count_res.sort_values(by='count', ascending=False)
    high_miss_values = 600
    res_less_than_high = count_res[count_res['count'] < high_miss_values]
    print(count_ordered)
    print(f'Total: {count_res.shape[0]} ; < {high_miss_values}: {res_less_than_high.shape[0]}')
    to_drop = res_less_than_high['idema'].tolist()
    # Drop stations with so much missin values to train
    df = df[~df['idema'].isin(to_drop)]
    count_res = df.groupby('idema').size().reset_index(name='count')
    print(f'Total: {count_res.shape[0]}')


    df['año'] = df['fint'].dt.year
    df['mes'] = df['fint'].dt.month
    df['día'] = df['fint'].dt.day
    df['hora'] = df['fint'].dt.hour
    df['minuto'] = df['fint'].dt.minute
    fecha_hora = ['año', 'mes', 'día', 'hora', 'minuto']
    df = df[fecha_hora + [col for col in df.columns if col not in fecha_hora]]

    df.columns = ['Año', 'Mes', 'Día', 'Hora', 'Minuto', 'fint', 'Id_estación', 'Lon', 'Lat', 'Altitud', 'Población',
                  'Temperatura', 'Temperatura_min', 'Temperatura_max', 'Precipitación', 'Humedad_relativa', 'Viento',
                  'Presión']
    print(df.info())
    cols_impute = ['Temperatura', 'Precipitación', 'Humedad_relativa', 'Viento', 'Presión']

    df = add_province_column(df)
    # Drop all rows with 'Unknown' province because is not in Historical data
    df = df[df['Provincia'] != 'Unknown']
    df.reset_index(drop=True, inplace=True)

    for column in cols_impute:
        df = impute_values(df, column)
    dm.eda_for_columns(df, cols_impute)

    # Update temp max/min to daily temperature
    df['Temperatura_min'] = df.groupby(['Año', 'Mes', 'Día', 'Id_estación'])['Temperatura_min'].transform('min')
    df['Temperatura_max'] = df.groupby(['Año', 'Mes', 'Día', 'Id_estación'])['Temperatura_max'].transform('max')
    df['Humedad_min'] = df.groupby(['Año', 'Mes', 'Día', 'Id_estación'])['Humedad_relativa'].transform('min')
    df['Humedad_max'] = df.groupby(['Año', 'Mes', 'Día', 'Id_estación'])['Humedad_relativa'].transform('max')
    df['Presión_min'] = df.groupby(['Año', 'Mes', 'Día', 'Id_estación'])['Presión'].transform('min')
    df['Presión_max'] = df.groupby(['Año', 'Mes', 'Día', 'Id_estación'])['Presión'].transform('max')
    df['Viento_medio'] = df.groupby(['Año', 'Mes', 'Día', 'Id_estación'])['Viento'].transform('mean')

    # Temperatura_min and Temperatura_max still have missing values because in a day all values are missing
    df = impute_values(df, 'Temperatura_min')
    df = impute_values(df, 'Temperatura_max')
    # Calculates the mean of values grouped by date and station between max and min values
    # for differences max and min for temperature, humidity and pressure
    df['Temperatura_media'] = df.groupby(['Año', 'Mes', 'Día', 'Id_estación'])['Temperatura'].transform('mean')
    df['Humedad_relativa_media'] = df.groupby(['Año', 'Mes', 'Día', 'Id_estación'])['Humedad_relativa'].transform(
        'mean')
    df['Presión_media'] = df.groupby(['Año', 'Mes', 'Día', 'Id_estación'])['Presión'].transform('mean')
    df = df.drop(columns=['fint'])
    df = df.drop(columns=['Provincia'])
    df.reset_index(drop=True, inplace=True)

    return df


def load_csv_hist_data():
    df = dm.load_df_from_csv(aemet_historical_data_path+'/full_climate_values_2022-09-18_to_2024-11-04.csv')
    df['año'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month
    df['día'] = df['fecha'].dt.day
    df['hora'] = df['fecha'].dt.hour
    df['minuto'] = df['fecha'].dt.minute
    fecha_hora = ['año', 'mes', 'día', 'hora', 'minuto']
    df = df[fecha_hora + [col for col in df.columns if col not in fecha_hora]]
    df.columns = ['Año', 'Mes', 'Día', 'Hora', 'Minuto', 'fint', 'Id_estación', 'Población', 'Provincia', 'Altitud',
                  'Temperatura', 'Precipitación', 'Temperatura_min', 'Hora_temperatura_min', 'Temperatura_max',
                  'Hora_temperatura_max', 'Dirección_viento', 'Viento', 'Racha_viento', 'Hora_racha_viento',
                  'Presión_max', 'Hora_presión_max', 'Presión_min', 'Hora_presión_min', 'Humedad_relativa',
                  'Humedad_relativa_max', 'Hora_humedad_relativa_max', 'Humedad_relativa_min', 'Hora_humedad_relativa',
                  'Horas_sol']
    # Add CCAA column
    df['CCAA'] = df['Provincia'].apply(lambda provincia: rm.retrieve_ccaa(provincia))
    print(df.info())

    for column in ['Temperatura', 'Precipitación', 'Temperatura_min', 'Temperatura_max', 'Viento', 'Racha_viento',
                   'Presión_max', 'Presión_min', 'Humedad_relativa', 'Humedad_relativa_max', 'Humedad_relativa_min',
                   'Hora_racha_viento', 'Hora_temperatura_max', 'Hora_temperatura_min', 'Hora_humedad_relativa',
                   'Hora_humedad_relativa_max', 'Hora_presión_max', 'Hora_presión_min']:
        df = impute_values(df, column)

    df = df.drop(columns=['fint', 'Horas_sol'])
    return df


if __name__ == '__main__':
    # Step 1 -> Load and generate CSV data from json
    # load_and_generate_csv_hist_data()
    # load_and_generate_csv_obs_data()

    # Step 2 -> impute values from CSV generated by previous step
    df_obs = load_csv_obs_data()
    print(df_obs.head())
    print(df_obs.info())
    print(df_obs.describe())

    # df_hist = load_csv_hist_data()
    # print(df_hist.head())
    # print(df_hist.info())
    # print(df_hist.describe())
    # print(df_hist.columns)

    # dm.save_df_to_csv(df_hist,
    #                   aemet_historical_data_path + '/full_climate_values_2022-09-18_to_2024-11-04_prepared.csv')
    dm.save_df_to_csv(df_obs,
                      aemet_observation_data_path + '/full_observation_values_2024-09-27_to_2024-11-07_prepared.csv')

    # Step 3: Load again the prepared csv
    # df_obs = dm.load_df_from_csv(
    #     aemet_observation_data_path + '/full_observation_values_2024-09-27_to_2024-11-07_prepared.csv')
    # print(df_obs.head())
    # print(df_obs.info())
    # print(df_obs.describe())
    #
    # df_hist = dm.load_df_from_csv(
    #     aemet_historical_data_path + '/full_climate_values_2022-09-18_to_2024-11-04_prepared.csv')
    #
    # print(df_hist.head())
    # print(df_hist.info())
    # print(df_hist.describe())
    # print(df_hist.columns)
