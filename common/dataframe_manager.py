from io import StringIO
from os.path import dirname, abspath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import common.chart_manager as cm
from common.tfm_util import calculate_altitude_angle

file_path = abspath(dirname(dirname(__file__)))
aemet_observation_data_path = file_path + '/AEMET-data/downloaded'
aemet_historical_data_path = file_path + '/AEMET-data/downloaded_historical_climate_val'
ree_historical_data_path = file_path + '/REE-data/datos'
obs_features_file_path = file_path + '/EUMESAT-data/extracted-features/cloudiness_by_region_obs.json'
hist_features_file_path = file_path + '/EUMESAT-data/extracted-features/cloudiness_by_region_hist.json'

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


def append_datetime_from_date_and_time_cols(df, drop_day_hour_cols=False, set_as_index=False):
    """
    Convert columns Ano, Mes, Dia, Hora, Minuto to datetime column
    :param df: The input DataFrame
    :param drop_day_hour_cols: Drop day and hour columns
    :param set_as_index: Set datetime column as index
    :return: The DataFrame with datetime column
    """
    df['datetime'] = pd.to_datetime({'year': df['Ano'], 'month': df['Mes'], 'day': df['Dia'],
                                     'hour': df['Hora'], 'minute': df['Minuto']})
    if drop_day_hour_cols:
        df.drop(columns=['Ano', 'Mes', 'Dia', 'Hora', 'Minuto'], inplace=True)
    if set_as_index:
        df.set_index('datetime', inplace=True)
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


def append_solar_elevation_angle(df):
    """
    Append solar elevation angle to DataFrame
    :param df: The input DataFrame
    :return: The DataFrame with solar elevation angle
    """
    df_out = append_datetime_from_date_and_time_cols(df)
    df_out['Solar_altitude'] = df_out.apply(lambda x: calculate_altitude_angle(x['Lat'], x['Lon'], x['datetime']),
                                            axis=1)
    df_out.drop(columns=['datetime'], inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    return df_out


def impute_mean_value_to_df(df, window=48, times=1):
    """
    Impute mean value to df grouping next 48 ordered rows to mean value. This is done 'times' times.
    :param df: The input DataFrame
    :param window: The window size
    :param times: The number of times to apply the imputation
    :return: The DataFrame with imputed values
    """
    for i in range(times):
        for column in df.select_dtypes(include=[np.number]).columns:
            rolling_mean = df[column].rolling(window=window, center=True, min_periods=1).mean()
            df[column] = df[column].fillna(rolling_mean)
    return df


# Load dataframe from csv file
def load_df_from_csv(file_name, index_col=None, sep=',', enc='utf-8', skiprows=0, parse_dates=None, names=None):
    if parse_dates is not None and names is not None:
        df = pd.read_csv(file_name, index_col=index_col, sep=sep, encoding=enc, skiprows=skiprows,
                         parse_dates=parse_dates, names=names)
    else:
        df = pd.read_csv(file_name, index_col=index_col, sep=sep, encoding=enc, skiprows=skiprows)

    return convert_date_and_hours_col_to_datetime(df)


def eda_for_columns(df: pd.DataFrame, columns: list, cols_to_show: list, suptitle) -> pd.DataFrame:
    """
    Perform EDA for the specified columns in the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame to analyze.
        columns (list): List of columns to analyze.
    """
    results = []

    for col in columns:
        print(f'EDA for feature: {col} --------------------------')

        # Check if column exists in DataFrame
        if col not in df.columns:
            print(f'Column {col} does not exist in DataFrame.')
            continue

        # Calculate basic statistics
        missing_values = df[col].isna().sum()
        mean_value = df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A'
        mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'

        # Detect outliers using 1.5*IQR rule
        if pd.api.types.is_numeric_dtype(df[col]):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = df[(df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))][col]
            num_outliers = outliers.count()
        else:
            num_outliers = 'N/A'

        if 'Id_estación' in df.columns:
            missing_stations = df.groupby('Id_estación')[col].apply(lambda x: x.isna().all())
            stations_with_all_missing = missing_stations[missing_stations].index.tolist()
        else:
            stations_with_all_missing = "La columna Id_estación no se encuentra en el DataFrame."

        # Append the results to the list
        results.append({
            'Column': col,
            'Missing Values': missing_values,
            'Mean': mean_value,
            'Most Frequent Value': mode_value,
            'Number of Outliers': num_outliers,
            "Estaciones con todos los valores perdidos": stations_with_all_missing
        })

        # Print the results
        print(f'Missing Values: {missing_values}')
        print(f'Mean: {mean_value}')
        print(f'Most Frequent Value: {mode_value}')
        print(f'Number of Outliers: {num_outliers}')
        print(f"Estaciones con todos los valores perdidos: {stations_with_all_missing}")

    # Plot boxplot for all given columns
    cm.plot_boxplots_in_line(df, columns, cols_to_show, suptitle)

    return pd.DataFrame(results)


def impute_outliers(df, columns_impute: list):
    """
    Impute outliers using linear interpolation.
    :param df: The input DataFrame
    :param columns_impute: The columns to impute
    :return: The DataFrame with imputed outliers
    """
    for col in columns_impute:
        # Detect outliers using 1.5*IQR rule
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        # Impute outliers using linear interpolation
        df[col + '_imputed'] = df[col].copy()
        df.loc[outliers_mask, col + '_imputed'] = np.nan
        df[col + '_imputed'] = df[col + '_imputed'].interpolate(method='linear')
        df[col] = df[col + '_imputed'].fillna(method='bfill').fillna(method='ffill')
        df.drop(columns=[col + '_imputed'], inplace=True)
    return df


def generate_correlations(df: pd.DataFrame, columns: list, cols_names=None, generate_plot=True,
                          annot=False) -> pd.DataFrame:
    """
    Generate correlation matrix for the specified columns in the DataFrame.
    Parameters:
        :param df: DataFrame to analyze.
        :param columns: List of columns to analyze.
        :param cols_names: List of column names to use in the correlation matrix.
        :param generate_plot: Whether to generate a heatmap plot of the correlation matrix.
        :param annot: Whether to display the correlation values in the heatmap plot.
    """
    corr_matrix = df[columns].corr(method='pearson')
    new_labels = cols_names if cols_names is not None else columns
    corr_matrix.index = new_labels
    corr_matrix.columns = new_labels
    if generate_plot:
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(corr_matrix, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200),
                         square=True, annot=annot, annot_kws={"size": 8}, cbar_kws={"shrink": 0.8, "aspect": 50},
                         fmt=".2f")
        # Rotate x-axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        plt.tight_layout()
        plt.show()
    return corr_matrix
