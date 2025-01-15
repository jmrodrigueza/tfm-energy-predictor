import os
import re

import numpy as np
import pandas as pd

import common.dataframe_manager as dm
import common.models
import common.models as mod
import common.tfm_util as util
from common.dataframe_manager import aemet_historical_data_path, aemet_observation_data_path, obs_features_file_path, \
    hist_features_file_path
from common.map_printer import print_map

pd.set_option('display.max_columns', None)

# COLUMNS UNLIMITED IN PANDAS
pd.set_option('display.width', 1000)
pd.set_option('display.max_info_columns', 500)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
np.random.seed(1)


# Load cloudiness data
# param type_cloudiness: str -> hist or obs
def load_cloudiness_data(type_cloudiness: str):
    file_path = hist_features_file_path if type_cloudiness == 'hist' else obs_features_file_path
    df = dm.load_df(file_path)
    df = dm.convert_date_and_hours_col_to_datetime(df, date_format='%Y-%m-%dT%H-%M-%S.%fZ')
    return df


# To fix This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate
# compiler flags.
def add_cloudiness(df: pd.DataFrame, type_cloudiness: str) -> pd.DataFrame:
    df_full = df.copy()
    df_cloudiness = load_cloudiness_data(type_cloudiness)
    # Add YYYY-MM-DD HH:MM:SS columns to the cloudiness dataframe
    df_cloudiness['Año'] = df_cloudiness['date_time'].dt.year
    df_cloudiness['Mes'] = df_cloudiness['date_time'].dt.month
    df_cloudiness['Día'] = df_cloudiness['date_time'].dt.day
    df_cloudiness['Hora'] = df_cloudiness['date_time'].dt.hour
    df_cloudiness['Minuto'] = df_cloudiness['date_time'].dt.minute
    df_cloudiness = df_cloudiness.drop(columns=['date_time'], axis=1)
    df_cloudiness = df_cloudiness.drop(columns=['file'], axis=1)

    # Append Cloudiness data
    return pd.merge(df_full, df_cloudiness, how='left', on=['Año', 'Mes', 'Día', 'Hora', 'Minuto'])


# Sample 10% by default of the data
def sample_stations(group, fraction=0.1):
    unique_ids = group['Station_Group'].unique()
    sampled_ids = np.random.choice(unique_ids, size=int(len(unique_ids) * fraction), replace=False)
    return group[group['Station_Group'].isin(sampled_ids)]


def load_dataset_train_obs(ccaa_to_add: pd.DataFrame, fraction=1.0) -> pd.DataFrame:
    # Load the csv file into a DataFrame
    df = dm.load_df_from_csv(
        aemet_observation_data_path + '/full_observation_values_2024-09-27_to_2024-11-07_prepared.csv')

    # Append CCAA data
    df = pd.merge(df, ccaa_to_add, how='left', on='Id_estación')
    df['CCAA'] = df['CCAA'].fillna('unknown')

    # Take a sample of 10% of the data
    unique_stations = df.groupby(['CCAA', 'Id_estación']).ngroup()
    df['Station_Group'] = unique_stations
    df = df.groupby('CCAA').apply(sample_stations, fraction=fraction).reset_index(drop=True)
    df.drop('Station_Group', axis=1, inplace=True)
    # Print stations map
    print_map(df[['Lat', 'Lon']].drop_duplicates())

    # Append cloudiness data
    df = add_cloudiness(df, 'obs')

    print('Observation data' + '-' * 50)
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.columns)
    print('-' * 60)

    return df


def prepare_dataset_historical() -> pd.DataFrame:
    df = dm.load_df_from_csv(
        aemet_historical_data_path + '/full_climate_values_2022-09-18_to_2024-11-04_prepared.csv')

    df['fecha'] = pd.to_datetime(
        df[['Año', 'Mes', 'Día']].rename(columns={'Año': 'year', 'Mes': 'month', 'Día': 'day'}))
    limit_date = pd.Timestamp('2024-09-27')
    df = df[df['fecha'] <= limit_date]
    # drop fecha column
    df = df.drop('fecha', axis=1)

    df_hist_expanded = df.loc[df.index.repeat(24)].reset_index(drop=True)
    df_hist_expanded['Hora'] = list(range(24)) * len(df)

    # Append cloudiness data
    df = add_cloudiness(df_hist_expanded, 'hist')
    # Add mean values for the Temp, Humidity, Pressure and Wind columns grouped by station and date
    df['temp_media'] = (df['Temperatura_max'] + df['Temperatura_min']) / 2
    df['hum_rel_media'] = (df['Humedad_relativa_max'] + df['Humedad_relativa_min']) / 2
    df['pres_media'] = (df['Presión_max'] + df['Presión_min']) / 2

    df['Temperatura_media'] = df.groupby(['Id_estación', 'Año', 'Mes', 'Día'])['temp_media'].transform('mean')
    df['Humedad_relativa_media'] = df.groupby(['Id_estación', 'Año', 'Mes', 'Día'])['hum_rel_media'].transform('mean')
    df['Presión_media'] = df.groupby(['Id_estación', 'Año', 'Mes', 'Día'])['pres_media'].transform('mean')
    df = df.drop(['temp_media', 'hum_rel_media', 'pres_media'], axis=1)

    predictor_columns = ['Año', 'Mes', 'Día', 'Hora', 'Minuto', 'Id_estación', 'Altitud', 'Precipitación',
                         'Temperatura_min', 'Temperatura_max', 'Humedad_relativa_min', 'Humedad_relativa_max',
                         'Presión_min', 'Presión_max', 'Temperatura_media', 'Humedad_relativa_media', 'Presión_media',
                         'Viento', 'Galicia', 'Asturias', 'Madrid', 'Valencia', 'Aragón',
                         'Cataluña', 'Cantabria', 'Islas Baleares', 'Castilla y León', 'Castilla La Mancha', 'La Rioja',
                         'Andalucía', 'Navarra', 'Murcia', 'País Vasco', 'Extremadura', 'CCAA']
    df = df[predictor_columns]
    print('Historical data' + '-' * 50)
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.columns)
    print('-' * 60)
    dm.save_df_to_parquet(df,
                          aemet_historical_data_path + '/' +
                          'full_climate_values_2022-09-18_to_2024-09-17_24hCloud_prepared.parquet')

    return df


def load_dataset_historical() -> pd.DataFrame:
    df = dm.load_df_from_csv(
        aemet_historical_data_path + '/full_climate_values_2022-09-18_to_2024-11-04_prepared.csv')
    # Append cloudiness data
    df = add_cloudiness(df, 'hist')

    print('Historical data' + '-' * 50)
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.columns)
    print('-' * 60)

    return df


def train_imputation_meteo_models(df_hist: pd.DataFrame):
    # Retrieve CCAA to be set into observation data
    station_ccaa = df_hist[['Id_estación', 'CCAA']]
    station_ccaa = station_ccaa.drop_duplicates()
    # Load 10% of the stations observation data
    df_obs = load_dataset_train_obs(station_ccaa, fraction=0.1)

    # Work with reduced dataframes to develop the models and 1 epoch.
    # This is to reduce the time needed to train the models
    epoch = 30

    predictor_columns = ['Mes', 'Dia', 'Hora', 'Altitud', 'Temperatura_media',
                         'Humedad_relativa_media', 'Presion_media', 'Viento_medio', 'Galicia', 'Asturias', 'Madrid',
                         'Valencia', 'Aragon', 'Cataluna', 'Cantabria', 'Islas_Baleares', 'Castilla_y_Leon',
                         'Castilla_La_Mancha', 'La_Rioja', 'Andalucia', 'Navarra', 'Murcia', 'Pais_Vasco',
                         'Extremadura']
    target_columns = ['Temperatura', 'Precipitación', 'Humedad_relativa', 'Viento', 'Presión']
    target_columns = util.sanitize_array_names(target_columns)
    df_obs.describe()

    df_obs = util.sanitize_column_names(df_obs)
    mod.train_and_models(df_obs, predictor_columns, target_columns, epoch=epoch)


def impute_meteorological_historical(df_hist: pd.DataFrame):
    # Load all models from directory
    models = {}
    # for model_file_name in os.listdir(common.models.trained_met_imp_models_base_path):
    for model_file_name in ['LSTM_5390Y.keras']:
        # Retrieve the name from model_file with a regex
        match = re.match(r"LSTM_([a-zA-Z0-9]+).keras", model_file_name)
        if match:
            station_id = match.group(1)
            model, scalers = mod.load_model(station_id, model_file_name,
                                            model_base_path=common.models.trained_met_imp_models_base_path,
                                            scaler_base_path=common.models.met_imp_scalers_base_path)
            models[station_id] = {'model': model, 'scalers': scalers}

    print(f'Model loaded for station_ids: {models.keys()}')

    # sanitize column names
    df_hist_sanitized = util.sanitize_column_names(df_hist)
    predictor_columns = ['Id_estacion', 'Ano', 'Mes', 'Dia', 'Hora', 'Minuto', 'Altitud', 'Temperatura_min',
                         'Temperatura_max', 'Humedad_relativa_min', 'Humedad_relativa_max', 'Presion_min',
                         'Presion_max', 'Temperatura_media', 'Humedad_relativa_media', 'Presion_media', 'Viento',
                         'Galicia', 'Asturias', 'Madrid', 'Valencia', 'Aragon', 'Cataluna', 'Cantabria',
                         'Islas_Baleares', 'Castilla_y_Leon', 'Castilla_La_Mancha', 'La_Rioja', 'Andalucia', 'Navarra',
                         'Murcia', 'Pais_Vasco', 'Extremadura', 'Precipitacion']
    target_columns = ['Temperatura', 'Precipitacion', 'Humedad_relativa', 'Viento', 'Presión']
    # Filter out the stations that do not have a model and the predictor columns
    df_hist_sanitized = df_hist_sanitized[df_hist_sanitized['Id_estacion'].isin(models.keys())]
    df_hist_sanitized = df_hist_sanitized[predictor_columns]

    predictor_columns = [x for x in predictor_columns if
                         x not in ['Id_estacion', 'Temperatura_min', 'Temperatura_max', 'Humedad_relativa_min',
                                   'Humedad_relativa_max', 'Presion_min', 'Presion_max', 'Precipitacion', 'Ano',
                                   'Minuto']]

    df_hist_imputed = pd.DataFrame()
    for station in models.keys():
        model = models[station]['model']
        scalers = models[station]['scalers']
        df_out = mod.do_predictions(station, model, scalers, df_hist_sanitized, predictor_columns)
        df_hist_imputed = pd.concat([df_hist_imputed, df_out], ignore_index=True)

    # Add Solar_altitude to the historical data
    station_ccaa = df_hist[['Id_estacion', 'CCAA']]
    station_ccaa = station_ccaa.drop_duplicates()
    station_ccaa.columns = ['Id_estación', 'CCAA']
    df_obs = load_dataset_train_obs(station_ccaa)
    df_obs = util.sanitize_column_names(df_obs)
    df_obs_merge = df_obs[['Id_estacion', 'Lat', 'Lon']].drop_duplicates().reset_index(drop=True)
    df_hist_imputed = pd.merge(df_hist_imputed, df_obs_merge, how='left', on=['Id_estacion'])
    df_hist_imputed = dm.append_solar_elevation_angle(df_hist_imputed)

    dm.save_df_to_parquet(df_hist_imputed,
                          aemet_historical_data_path + '/' +
                          'full_climate_values_2022-09-18_to_2024-09-17_24hCloud_imputed.parquet')
    print(df_hist_imputed.head(n=100))
    print(df_hist_imputed.info())
    print(df_hist_imputed.describe())


# Main function
if __name__ == '__main__':
    # Step 1: Prepare historical dataset adding cloudiness data
    # df_historical = load_dataset_historical()
    # Step 2: generate parquet file
    # df_historical = prepare_dataset_historical()
    # Step 3: Train imputation models
    df_historical = dm.load_df_from_parquet(
        aemet_historical_data_path + '/full_climate_values_2022-09-18_to_2024-09-17_24hCloud_prepared.parquet')
    # train_imputation_meteo_models(df_historical)
    impute_meteorological_historical(df_historical)
    # Step 4: Impute meteorological imputation data
    # df_historical = dm.load_df_from_parquet(
    #     aemet_historical_data_path + '/full_climate_values_2022-09-18_to_2024-09-17_24hCloud_prepared.parquet')
    # impute_meteorological_historical(df_historical)
