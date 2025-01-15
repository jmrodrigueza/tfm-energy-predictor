import os
import re
from os.path import abspath, dirname

import numpy as np
import pandas as pd
import common.chart_manager as cm

import common.dataframe_manager as dm

file_path = abspath(dirname(__file__))
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def date_parser(date_time):
    return pd.to_datetime(date_time, format='%Y-%m-%d %H:%M', errors='coerce')


parse_dates = ['Hora']
date_regex = '([0-9]{4}-[0-9]{2}-[0-9]{2})'
emissions_filename_regex = f'Custom-Report-{date_regex}-Emisiones \\(t CO2 eq _ MWh\\).csv'
generation_filename_regex = f'Custom-Report-{date_regex}-Estructura de generación \\(MW\\).csv'
demand_filename_regex = f'Custom-Report-{date_regex}-Seguimiento de la demanda de energía eléctrica \\(MW\\).csv'
emission_cols = ['Hora', 'Eolica', 'Nuclear', 'Carbon', 'Ciclo_combinado', 'Hidraulica', 'Intercambios_int',
                 'Solar_fotovoltaica', 'Solar_termica', 'Termica_renovable', 'Motores_diesel', 'Turbina_de_gas',
                 'Turbina_de_vapor', 'Generacion_auxiliar', 'Cogeneracion_y_residuos', 'delete']
generation_cols = ['Hora', 'Eolica', 'Nuclear', 'Carbon', 'Ciclo_combinado', 'Hidraulica', 'Intercambios_int',
                   'Solar_fotovoltaica', 'Solar_termica', 'Termica_renovable', 'Motores_diesel', 'Turbina_de_gas',
                   'Turbina_de_vapor', 'Generacion_auxiliar', 'Cogeneracion_y_residuos', 'delete']
demand_cols = ['Hora', 'Real', 'Prevista', 'Programada', 'delete']
aemet_cols = ['Altitud', 'Viento', 'Temperatura', 'Precipitacion', 'Humedad_relativa', 'Viento_pred', 'Presion']
energy_meteo_files_path = f'{file_path}/energy_meteo_files'


# Load all csv files from REE-data folder
def load_ree_data():
    df_emissions = pd.DataFrame()
    df_generation = pd.DataFrame()
    df_demand = pd.DataFrame()

    for filename in os.listdir(dm.ree_historical_data_path):
        # Load filename depending on the regex
        if filename.endswith('.csv'):
            if re.match(emissions_filename_regex, filename):
                df_emissions = pd.concat(
                    [df_emissions,
                     dm.load_df_from_csv(dm.ree_historical_data_path + '/' + filename, skiprows=3, enc='cp1252',
                                         parse_dates=False, names=emission_cols)], ignore_index=True)
            elif re.match(generation_filename_regex, filename):
                df_generation = pd.concat(
                    [df_generation,
                     dm.load_df_from_csv(dm.ree_historical_data_path + '/' + filename, skiprows=3, enc='cp1252',
                                         parse_dates=False, names=generation_cols)], ignore_index=True)
            elif re.match(demand_filename_regex, filename):
                df_demand = pd.concat(
                    [df_demand,
                     dm.load_df_from_csv(dm.ree_historical_data_path + '/' + filename, skiprows=3, enc='cp1252',
                                         parse_dates=False, names=demand_cols)], ignore_index=True)
    # drop unneeded delete columns
    df_emissions.drop(columns=['delete'], inplace=True)
    df_generation.drop(columns=['delete'], inplace=True)
    df_demand.drop(columns=['delete'], inplace=True)
    # Rename Hora column to datetime
    df_emissions = dm.convert_datetime_cols_to_day_hour_cols(df_emissions, 'Hora')
    df_emissions.columns = ['Ano', 'Mes', 'Dia', 'Hora', 'Minuto'] + [col + '_emi' for col in df_emissions.columns[5:]]
    df_generation = dm.convert_datetime_cols_to_day_hour_cols(df_generation, 'Hora')
    df_generation.columns = ['Ano', 'Mes', 'Dia', 'Hora', 'Minuto'] + [col + '_gen' for col in
                                                                       df_generation.columns[5:]]
    df_demand = dm.convert_datetime_cols_to_day_hour_cols(df_demand, 'Hora')

    # print head of each dataframe
    print(df_emissions.head())
    print(df_generation.head())
    print(df_demand.head())
    print('--------------------------------')

    # print info of each dataframe
    print(df_emissions.info())
    print(df_generation.info())
    print(df_demand.info())
    print('--------------------------------')

    # print describe of each dataframe
    print(df_emissions.describe())
    print(df_generation.describe())
    print(df_demand.describe())
    print('--------------------------------')

    return df_emissions, df_generation, df_demand


def load_cloudiness_data():
    df = dm.load_df(dm.hist_features_file_path)
    df = dm.convert_date_and_hours_col_to_datetime(df, date_format='%Y-%m-%dT%H-%M-%S.%fZ')
    df = dm.convert_date_time_to_day_hour(df, 'date_time')
    print(df.head())
    print(df.describe())
    return df


def load_aemet_data():
    df_aemet_hist = dm.load_df_from_parquet(
        dm.aemet_historical_data_path + '/full_climate_values_2022-09-18_to_2024-09-17_24hCloud_imputed.parquet')
    df_aemet_hist = dm.filter_df_by_date(df_aemet_hist, '2022-09-18', '2024-09-17')

    # Impute missing values for Temperatura, Precipitacion, Humedad_relativa, Presion
    df_aemet_hist.loc[df_aemet_hist['Temperatura'].isna(), 'Temperatura'] = df_aemet_hist['Temperatura_media']
    df_aemet_hist.loc[df_aemet_hist['Precipitacion'].isna(), 'Precipitacion'] = df_aemet_hist['Temperatura_media']
    df_aemet_hist.loc[df_aemet_hist['Humedad_relativa'].isna(),
    'Humedad_relativa'] = df_aemet_hist['Humedad_relativa_media']
    df_aemet_hist.loc[df_aemet_hist['Presion'].isna(), 'Presion'] = df_aemet_hist['Presion_media']
    df_aemet_hist = dm.impute_mean_value_to_df(df_aemet_hist, window=6, times=16)
    df_aemet_hist = dm.impute_mean_value_to_df(df_aemet_hist, window=12, times=16)
    df_aemet_hist = dm.impute_mean_value_to_df(df_aemet_hist, window=24, times=2)

    print(df_aemet_hist.head())
    print(df_aemet_hist.describe())
    print(df_aemet_hist.info())

    return df_aemet_hist


# Merge all dataframes by Ano, Mes, Dia, Hora, Minuto in df aemet_hist
def merge_dataframes(df_emission, df_generation, df_demand, df_cloud, df_aemet_hist):
    df_aemet_pivoted = df_aemet_hist.pivot_table(index=['Ano', 'Mes', 'Dia', 'Hora', 'Minuto'], columns='Id_estacion',
                                                 values=aemet_cols, aggfunc='first')
    df_aemet_pivoted.columns = [f'{col[1]}_{col[0]}' for col in df_aemet_pivoted.columns.values]
    df_aemet_pivoted.reset_index(inplace=True)

    df_out = pd.merge(df_aemet_pivoted, df_emission, on=['Ano', 'Mes', 'Dia', 'Hora', 'Minuto'], how='left')
    df_out = pd.merge(df_out, df_generation, on=['Ano', 'Mes', 'Dia', 'Hora', 'Minuto'], how='left')
    df_out = pd.merge(df_out, df_demand, on=['Ano', 'Mes', 'Dia', 'Hora', 'Minuto'], how='left')
    df_out = pd.merge(df_out, df_cloud, on=['Ano', 'Mes', 'Dia', 'Hora', 'Minuto'], how='left')
    df_out.reset_index(drop=True, inplace=True)
    df_out = dm.append_datetime_from_date_and_time_cols(df_out)
    df_out.set_index('datetime', inplace=True)
    df_out.fillna(method='ffill', inplace=True)
    df_out.fillna(method='bfill', inplace=True)
    df_out.interpolate(method='linear', inplace=True)

    # Add mean Solar_altitude
    # Group by Ano, Mes, Dia, Hora, Minuto to the mean Solar_altitude in df_aemet_hist
    df_out['Solar_altitude'] = df_out.apply(lambda x: df_aemet_hist[
        (df_aemet_hist['Ano'] == x['Ano']) &
        (df_aemet_hist['Mes'] == x['Mes']) &
        (df_aemet_hist['Dia'] == x['Dia']) &
        (df_aemet_hist['Hora'] == x['Hora']) &
        (df_aemet_hist['Minuto'] == x['Minuto'])]['Solar_altitude'].mean(), axis=1)

    # print head, describe and info of the resulting dataframe
    print(df_out.head())
    print(df_out.describe())
    print(df_out.info())
    # cm.plot_series(df_out, np.array([['0201D_Presion']]), '0201D_Presion', 'tiempo', '0201D_Presion')
    return df_out


# Main
if __name__ == '__main__':
    # Load data from csv
    df_emi, df_gen, df_dem = load_ree_data()
    df_cloudiness = load_cloudiness_data()
    df_aemet = load_aemet_data()

    col_to_plot = ['Carbon_emi', 'Ciclo_combinado_emi', 'Motores_diesel_emi', 'Turbina_de_gas_emi',
                   'Turbina_de_vapor_emi', 'Cogeneracion_y_residuos_emi']
    col_names = ['Carbono', 'Ciclo combinado', 'Motores diésel', 'Turbina de gas', 'Turbina de vapor',
                 'Cogeneración y residuos']
    dm.eda_for_columns(df_emi, col_to_plot, col_names, 'Emisiones en tCO2 eq/MWh')
    col_to_plot = ['Eolica_gen', 'Hidraulica_gen', 'Solar_fotovoltaica_gen', 'Solar_termica_gen']
    col_names = ['Eólica', 'Hidráulica', 'Solar fotovoltaica', 'Solar térmica']
    dm.eda_for_columns(df_gen, col_to_plot, col_names, 'Energía producida en MW')
    col_to_plot = ['Real', 'Prevista', 'Programada']
    dm.eda_for_columns(df_dem, col_to_plot, col_to_plot, 'Energía demandada en MW')

    # Impute outliers
    df_emi = dm.impute_outliers(df_emi, ['Carbon_emi', 'Ciclo_combinado_emi', 'Motores_diesel_emi',
                                         'Turbina_de_gas_emi',
                                         'Turbina_de_vapor_emi', 'Cogeneracion_y_residuos_emi'])
    df_gen = dm.impute_outliers(df_gen, ['Eolica_gen', 'Hidraulica_gen', 'Solar_fotovoltaica_gen', 'Solar_termica_gen'])
    df_dem = dm.impute_outliers(df_dem, ['Real'])

    # Merge all dataframes
    df = merge_dataframes(df_emi, df_gen, df_dem, df_cloudiness, df_aemet)
    # Save resulting df to parquet
    dm.save_df_to_parquet(df, energy_meteo_files_path + '/energy_meteo_data.parquet')
