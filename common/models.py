import gc

import joblib
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model
from keras.src.initializers import HeNormal
from keras.src.layers import Conv1D, LSTM, Dense, Lambda, Concatenate
from keras.src.optimizers import Adam
from keras.src.saving import register_keras_serializable
from sklearn.metrics import r2_score
from tensorflow.python.data import Dataset
import common.chart_manager as cm
from os.path import dirname, abspath
from common.robust_scaler_serializable import RobustScalerSerializable
from common.early_stop import EarlyStopperCallback

file_path = abspath(dirname(dirname(__file__)))
trained_models_base_path = file_path + '/trained_models'
trained_met_imp_models_base_path = trained_models_base_path + '/meteo_imputation_models'
met_imp_scalers_base_path = trained_models_base_path + '/meteo_imputation_scalers'

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=200)
np.set_printoptions(precision=4, floatmode='fixed')


def create_windowed_dataset(dataset, window_size, shift):
    dataset = dataset.window(window_size, shift=shift, drop_remainder=True)
    return dataset.flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(window_size), y.batch(window_size))))


def windowed_dataset(df, label_columns, target_columns, shuffle=True, batch_size=32, window_size=10):
    df_x = df[label_columns]
    df_y = df[target_columns]
    num_samples = len(df_x) // window_size

    print('df_x.shape: ', df_x.shape, ' df_y.shape: ', df_y.shape)
    ds = Dataset.from_tensor_slices((df_x, df_y))
    ds = create_windowed_dataset(ds, window_size, 1)
    if shuffle:
        ds = ds.shuffle(buffer_size=100)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def create_windowed_dataset_one_var(dataset, window_size, shift):
    dataset = dataset.window(window_size, shift=shift, drop_remainder=True)
    dataset = dataset.flat_map(lambda x: x.batch(window_size))

    return dataset


def windowed_dataset_one_var(df, label_columns, shuffle=True, batch_size=32, window_size=10):
    df_x = df[label_columns]
    print('df_x.shape: ', df_x.shape)

    ds = Dataset.from_tensor_slices(df_x)
    ds = create_windowed_dataset_one_var(ds, window_size, 1)
    if shuffle:
        ds = ds.shuffle(buffer_size=100)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


@register_keras_serializable(package="tfm.common", name="limit_output")
def limit_output(inputs, min_value, max_value, train_scaler, target_column_index):
    # de-scale the input values to limit them
    medians = train_scaler.center_[target_column_index]
    iqrs = train_scaler.scale_[target_column_index]
    x_descale = inputs * iqrs + medians
    x_clipped = tf.clip_by_value(x_descale, clip_value_min=min_value, clip_value_max=max_value)
    x_rescale = (x_clipped - medians) / iqrs

    return x_rescale


def get_min_max_values(min_value: float, max_value: float, scaler, target_column_index: int):
    return {'min_value': min_value, 'max_value': max_value, 'train_scaler': scaler,
            'target_column_index': target_column_index}


def generate_lambda_layer(input_dense, train_scaler):
    return [Lambda(limit_output,
                   output_shape=(1,),
                   arguments=get_min_max_values(-50, 50, train_scaler, 0))(input_dense[0]),
            Lambda(limit_output,
                   output_shape=(1,),
                   arguments=get_min_max_values(0, 300, train_scaler, 1))(input_dense[1]),
            Lambda(limit_output,
                   output_shape=(1,),
                   arguments=get_min_max_values(0, 100, train_scaler, 2))(input_dense[2]),
            Lambda(limit_output,
                   output_shape=(1,),
                   arguments=get_min_max_values(0, 300, train_scaler, 3))(input_dense[3]),
            Lambda(limit_output,
                   output_shape=(1,),
                   arguments=get_min_max_values(600, 1100, train_scaler, 4))(input_dense[4])]


# Build the model
def build_model(predictor_columns, target_columns, train_scaler, window_size=64, batch_size=32):
    tf.keras.backend.clear_session()
    tf.random.set_seed(0)
    np.random.seed(0)

    in1 = Input(shape=[None, len(predictor_columns)])
    conv1 = Conv1D(filters=128, kernel_size=47, strides=1, padding="causal", activation='relu')(in1)
    conv2 = Conv1D(filters=128, kernel_size=23, strides=1, padding="causal", activation='relu')(in1)
    conv3 = Conv1D(filters=256, kernel_size=11, strides=1, padding="causal", activation='relu')(in1)
    conv4 = Conv1D(filters=256, kernel_size=5, strides=1, padding="causal", activation='relu')(in1)
    lstm1 = LSTM(256, return_sequences=True)(conv1)
    lstm2 = LSTM(256, return_sequences=True)(conv2)
    lstm3 = LSTM(256, return_sequences=True)(conv3)
    lstm4 = LSTM(256, return_sequences=True)(conv4)
    concat = Concatenate()([lstm1, lstm2, lstm3, lstm4])
    dense1 = Dense(512, activation='relu', kernel_initializer=HeNormal())(concat)
    dense2 = Dense(64, activation='relu', kernel_initializer=HeNormal())(dense1)
    dense3 = Dense(1, activation='linear', kernel_initializer=HeNormal())(dense2)
    dense4 = Dense(1, activation='linear', kernel_initializer=HeNormal())(dense2)
    dense5 = Dense(1, activation='linear', kernel_initializer=HeNormal())(dense2)
    dense6 = Dense(1, activation='linear', kernel_initializer=HeNormal())(dense2)
    dense7 = Dense(1, activation='linear', kernel_initializer=HeNormal())(dense2)
    concat = Concatenate()(generate_lambda_layer([dense3, dense4, dense5, dense6, dense7], train_scaler))
    built_model = Model(inputs=in1, outputs=concat)
    optimizer = Adam()
    built_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    built_model.summary()
    return built_model


def get_optimal_bsize_and_window(x_train, max_batch_size=256, min_window_size=1):
    num_samples = x_train.shape[0]
    optimal_batch_size = max_batch_size
    while optimal_batch_size > num_samples or num_samples % optimal_batch_size != 0:
        optimal_batch_size -= 1
    optimal_window_size = min_window_size
    while optimal_window_size < num_samples and num_samples % optimal_window_size != 0:
        optimal_window_size += 1

    optimal_window_size = min(optimal_window_size, num_samples)
    return optimal_batch_size, optimal_window_size


def scale_input_dataframe(input_scaler, df, columns_to_scale):
    """
    Scale the input dataframe using the input scaler.
    :param input_scaler: The input scaler to use.
    :param df: The dataframe to scale.
    :param columns_to_scale: The columns to scale.
    :return: The scaled dataframe.
    """
    normalized_array_in = input_scaler.fit_transform(df[columns_to_scale])
    return pd.DataFrame(normalized_array_in, columns=columns_to_scale)


# scale only the input predictor variables, not target columns
# def scale_dataframe(scalers, df, columns_to_scale, target_columns):
def scale_dataframe(df, columns_to_scale, target_columns):
    input_scaler, output_scaler = RobustScalerSerializable(), RobustScalerSerializable()

    normalized_array_in = input_scaler.fit_transform(df[columns_to_scale])
    normalized_array_out = output_scaler.fit_transform(df[target_columns])
    scaled_df_in = pd.DataFrame(normalized_array_in, columns=columns_to_scale)
    scaled_df_out = pd.DataFrame(normalized_array_out, columns=target_columns)
    scaled_df = pd.concat([scaled_df_in, scaled_df_out], axis=1)

    return scaled_df, input_scaler, output_scaler


def train_and_models(df: pd.DataFrame, predictor_columns: list, target_columns: list, epoch: int = 1):
    train_data_rate = 0.8
    # test_data_rate = 0.2
    # Last 2 days to predict
    window_size = 48
    batch_size = 48
    shuffle_buffer_size = 1000
    trained_models = {}
    df = df.drop(['Poblacion', 'CCAA'], axis=1)

    stations_to_process = df['Id_estacion'].unique()
    for i, station in enumerate(stations_to_process):
        gc.collect()
        print('Processing station: ', station, ' nº: ', i, ' of:', len(stations_to_process))
        df_station = df[df['Id_estacion'] == station]
        df_station = df_station.drop(['Id_estacion'], axis=1)
        train_data_size = int(df_station.shape[0] * train_data_rate)
        df_train = df_station[:train_data_size]
        df_test = df_station[train_data_size:]
        x_train, in_train_scaler, out_train_scaler = scale_dataframe(df_train, predictor_columns, target_columns)
        x_test, in_test_scaler, out_test_scaler = scale_dataframe(df_test, predictor_columns, target_columns)

        bsize, win_size = get_optimal_bsize_and_window(x_train, max_batch_size=batch_size, min_window_size=window_size)
        bsize = batch_size
        win_size = window_size

        train_set = windowed_dataset(x_train, batch_size=bsize, label_columns=predictor_columns,
                                     target_columns=target_columns, window_size=win_size)
        x_test_set = windowed_dataset(x_test, batch_size=bsize, label_columns=predictor_columns,
                                      target_columns=target_columns, window_size=win_size, shuffle=False)
        model = build_model(predictor_columns, target_columns, out_train_scaler, window_size=win_size, batch_size=bsize)

        history = model.fit(train_set, validation_data=x_test_set, epochs=epoch, batch_size=bsize,
                            callbacks=[EarlyStopperCallback(mae_threshold=0.25)])

        cm.plot_fit_history(history, True, f'fit_history_station_{station}',
                            title_detail=f'con el LSTM de la estación {station}', min_loss=0.0, max_loss=0.6)

        y_pred = model.predict(x_test_set, batch_size=bsize)
        y_pred = y_pred[:, -1, :]
        y_pred = out_test_scaler.inverse_transform(y_pred)
        x_test_scaled = x_test[target_columns].iloc[-y_pred.shape[0]:].reset_index(drop=True)
        adjusted_target = pd.DataFrame(out_test_scaler.inverse_transform(x_test_scaled), columns=target_columns)

        r2 = r2_score(adjusted_target, y_pred)
        print("R^2 score:", r2)

        df_comparison = df_station[['Ano', 'Mes', 'Dia', 'Hora', 'Minuto']].iloc[(train_data_size+win_size-1):] \
            .reset_index(drop=True)
        df_comparison['y_pred_temp'] = y_pred[:, 0]
        df_comparison['x_test_temp'] = adjusted_target['Temperatura']

        df_comparison['y_pred_prec'] = y_pred[:, 1]
        df_comparison['x_test_prec'] = adjusted_target['Precipitacion']

        df_comparison['y_pred_hum'] = y_pred[:, 2]
        df_comparison['x_test_hum'] = adjusted_target['Humedad_relativa']

        df_comparison['y_pred_wind'] = y_pred[:, 3]
        df_comparison['x_test_wind'] = adjusted_target['Viento']

        df_comparison['y_pred_pres'] = y_pred[:, 4]
        df_comparison['x_test_pres'] = adjusted_target['Presion']

        cm.plot_series(df_comparison, np.array([['y_pred_temp', 'x_test_temp'],
                                                ['y_pred_prec', 'x_test_prec']], dtype=object),
                       [f'Temperatura estación {station}', f'Precipitación estación {station}'], ['Fecha', 'Fecha'],
                       ['Temperatura', 'Precipitación'], save=True,
                       file_name=f'{station}_temp_prec_x_test_vs_y_pred_validation')

        cm.plot_series(df_comparison, np.array([['y_pred_hum', 'x_test_hum'],
                                                ['y_pred_wind', 'x_test_wind'],
                                                ['y_pred_pres', 'x_test_pres']], dtype=object),
                       [f'Humedad estación {station}', f'Viento estación {station}', f'Presión estación {station}'],
                       ['Fecha', 'Fecha', 'Fecha'], ['Humedad', 'Viento', 'Presión'], save=True,
                       file_name=f'{station}_hum_wind_pres_x_test_vs_y_pred_validation')

        model.save(f'{trained_met_imp_models_base_path}/LSTM_{station}.keras')
        joblib.dump(in_train_scaler, f'{met_imp_scalers_base_path}/LSTM_{station}_in.joblib')
        joblib.dump(out_train_scaler, f'{met_imp_scalers_base_path}/LSTM_{station}_out.joblib')
        trained_models[station] = {'model': model, 'scalers': (in_train_scaler, out_train_scaler)}
    print('End of training...')
    return trained_models


def do_predictions(station, model, scalers, df, predictor_columns):
    in_scaler, out_scaler = scalers[0], scalers[1]

    # Prepare input df
    df_station = df[df['Id_estacion'] == station]
    df_station = df_station.sort_values(by=['Ano', 'Mes', 'Dia', 'Hora', 'Minuto'])
    df_station = df_station.reset_index(drop=True)
    df_station_train = df_station.drop(['Id_estacion', 'Ano'], axis=1)
    df_station_train = df_station_train.reset_index(drop=True)
    df_station_scaled = scale_input_dataframe(in_scaler, df_station_train, predictor_columns)

    bsize, _ = get_optimal_bsize_and_window(df_station_scaled, max_batch_size=64, min_window_size=48)
    # windowed dataset
    df_prediction_windowed = windowed_dataset_one_var(df_station_scaled, batch_size=48,
                                                      label_columns=predictor_columns,
                                                      window_size=48, shuffle=False)
    # Do the prediction for the station
    prediction = model.predict(df_prediction_windowed)
    y_pred = prediction[:, -1, :]
    y_pred = out_scaler.inverse_transform(y_pred)
    y_pred = y_pred[~np.isnan(y_pred).any(axis=1)]
    print(f'Prediction for station {station}: {y_pred}')

    # Create missing values because of the window size
    missing_values = len(df_station_scaled) - len(y_pred)
    df_station['Temperatura'] = pd.Series([np.nan] * missing_values + y_pred[:, 0].tolist())
    df_station['Precipitacion_pred'] = pd.Series([np.nan] * missing_values + y_pred[:, 1].tolist())
    df_station['Humedad_relativa'] = pd.Series([np.nan] * missing_values + y_pred[:, 2].tolist())
    df_station['Viento_pred'] = pd.Series([np.nan] * missing_values + y_pred[:, 3].tolist())
    df_station['Presion'] = pd.Series([np.nan] * missing_values + y_pred[:, 4].tolist())
    # select only last year
    df_station = df_station.tail(n=8760)

    cm.plot_series(df_station, np.array([['Temperatura', 'Temperatura_media'],
                                         ['Precipitacion_pred', 'Precipitacion']], dtype=object),
                   [f'Temperatura estación {station}', f'Precipitación estación {station}'], ['Fecha', 'Fecha'],
                   ['Temperatura', 'Precipitación'], labels=['Predicción', 'Media real'], save=True,
                   file_name=f'{station}_temp_prec_x_test_vs_y_pred')
    cm.plot_series(df_station, np.array([['Humedad_relativa', 'Humedad_relativa_media'],
                                         ['Viento_pred', 'Viento']], dtype=object),
                   [f'Humedad estación {station}', f'Viento estación {station}'], ['Fecha', 'Fecha'],
                   ['Humedad', 'Viento'], labels=['Predicción', 'Media real'], save=True,
                   file_name=f'{station}_hum_wind_x_test_vs_y_pred')
    cm.plot_series(df_station, np.array([['Presion', 'Presion_media']], dtype=object), [f'Presión estación {station}'],
                   ['Fecha'], ['Presión'], labels=['Predicción', 'Media real'], save=True,
                   file_name=f'{station}_pres_x_test_vs_y_pred')

    print('-' * 50)
    print(df_station.head(n=48))
    print(df_station.info())
    print(df_station.describe())
    print('-' * 50)

    return df_station


def load_model(station_id, model_filename, model_base_path=trained_models_base_path,
               scaler_base_path=met_imp_scalers_base_path):
    model = keras.models.load_model(f'{model_base_path}/{model_filename}',
                                    custom_objects={"limit_output": limit_output,
                                                    "RobustScalerSerializable": RobustScalerSerializable})
    # Cargar el scaler desde el archivo
    scaler_in = joblib.load(f'{scaler_base_path}/LSTM_{station_id}_in.joblib')
    scaler_out = joblib.load(f'{scaler_base_path}/LSTM_{station_id}_out.joblib')
    return model, (scaler_in, scaler_out)
