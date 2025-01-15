from io import StringIO

import numpy as np
import pandas as pd

# COLUMNS UNLIMITED IN PANDAS
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_info_columns', 2000)


# save DF to JSON file
def save_df(df, output_file_path, orient='index', indent=2, encoding='utf-8'):
    df_json = df.to_json(orient=orient, indent=indent)
    with open(output_file_path, 'w', encoding=encoding) as f:
        f.write(df_json)


# load DF from JSON file
def load_df(input_file_path, orient='index', encoding='utf-8') -> pd.DataFrame:
    with open(input_file_path, 'r', encoding=encoding) as f:
        json_str = f.read()
    return pd.read_json(StringIO(json_str), orient=orient)


# Save dataframe to csv file
def save_df_to_csv(df, file_name, index=False, sep=',', enc='utf-8'):
    try:
        df.to_csv(file_name, index=index, sep=sep, encoding=enc, float_format="%.6f")
    except Exception as e:
        print(f"Error saving to file {file_name}: {e}")


# save dataframe to parquet file
def save_df_to_parquet(df, file_name, index=False):
    try:
        df.to_parquet(file_name, index=index)
    except Exception as e:
        print(f"Error saving to file {file_name}: {e}")


# Load dataframe from parquet file
def load_df_from_parquet(file_name):
    df = pd.read_parquet(file_name)
    return df


# Función para convertir solo columnas que cumplen con el formato de hora
def convert_date_and_hours_col_to_datetime(df, date_format=None):
    for columna in df.columns:
        is_date_column = columna.startswith('hora') or columna.startswith('datetime') or columna.startswith('fecha') \
                         or columna.startswith('fint') or columna.startswith('date_time')
        if is_date_column and df[columna].dtype == 'object':
            df[columna] = pd.to_datetime(df[columna]) if date_format is None else pd.to_datetime(df[columna],
                                                                                                 format=date_format)

    return df


# convert date hour columns to day and hour columns
def convert_date_time_to_day_hour(df, date_time_col):
    df_out = df.copy()
    df_out['Ano'] = df_out[date_time_col].dt.year
    df_out['Mes'] = df_out[date_time_col].dt.month
    df_out['Dia'] = df_out[date_time_col].dt.day
    df_out['Hora'] = df_out[date_time_col].dt.hour
    df_out['Minuto'] = df_out[date_time_col].dt.minute
    df_out.drop(columns=[date_time_col], inplace=True)

    # Reorder columns
    df_out = df_out[['Ano', 'Mes', 'Dia', 'Hora', 'Minuto'] + [col for col in df_out.columns if
                                                               col not in ['Ano', 'Mes', 'Dia', 'Hora', 'Minuto']]]

    return df_out


def convert_datetime_cols_to_day_hour_cols(df, date_time_col):
    df.rename(columns={date_time_col: 'datetime'}, inplace=True)
    df = convert_date_and_hours_col_to_datetime(df)
    df = df.drop_duplicates(subset='datetime', keep='last')
    df = convert_date_time_to_day_hour(df, 'datetime')
    return df


def append_datetime_from_date_and_time_cols(df, drop_day_hour_cols=False):
    """
    Convert columns Ano, Mes, Dia, Hora, Minuto to datetime column
    :param df: The input DataFrame
    :param drop_day_hour_cols: Drop day and hour columns
    :return: The DataFrame with datetime column
    """
    df['datetime'] = pd.to_datetime({'year': df['Ano'], 'month': df['Mes'], 'day': df['Dia'],
                                     'hour': df['Hora'], 'minute': df['Minuto']})
    if drop_day_hour_cols:
        df.drop(columns=['Ano', 'Mes', 'Dia', 'Hora', 'Minuto'], inplace=True)
    return df


def append_weekend_column(df):
    """
    Append weekend column to DataFrame
    :param df: The input DataFrame
    :return: The DataFrame with weekend column
    """
    datetime_added = False
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime({'year': df['Ano'], 'month': df['Mes'], 'day': df['Dia'],
                                         'hour': df['Hora'], 'minute': df['Minuto']})
        datetime_added = True
    df['is_weekend'] = np.where(df['datetime'].dt.dayofweek >= 5, 1, 0)
    if datetime_added:
        df.drop(columns=['datetime'], inplace=True)
    return df


def filter_df_by_date(df, start_date, end_date):
    """
    Filter data in range 2022-09-18 to 2024-09-17
    :param df: The input DataFrame
    :param start_date: The start date
    :param end_date: The end date
    :return: The filtered DataFrame
    """
    start_date_hist = pd.to_datetime(start_date)
    end_date_hist = pd.to_datetime(end_date)
    df_copy = df.copy()
    df_copy['Date'] = pd.to_datetime({'year': df_copy['Ano'], 'month': df_copy['Mes'], 'day': df_copy['Dia']})
    df_copy = df_copy[(df_copy['Date'] >= start_date_hist) & (df_copy['Date'] <= end_date_hist)]
    df_copy.drop(columns=['Date'], inplace=True)

    return df_copy
