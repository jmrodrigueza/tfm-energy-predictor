import abc

from os.path import abspath, dirname

import joblib
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model
from keras.src.initializers import HeNormal
from keras.src.layers import Conv1D, Concatenate, Dense, GlobalAveragePooling1D, Dropout, LayerNormalization, \
    MultiHeadAttention
from keras.src.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

import common.chart_manager as cm
import common.dataframe_manager as dm
from common.early_stop import EarlyStopperCallback
from common.models import scale_dataframe, limit_output
from common.robust_scaler_serializable import RobustScalerSerializable
from meteo_energy_databuilder import energy_meteo_files_path


def transformer_encoder(inputs, num_heads, ff_dim, dropout=0.0):
    kernel_size = 1
    # Attention and Normalization
    x = MultiHeadAttention(key_dim=inputs.shape[-1], num_heads=num_heads, dropout=dropout)(query=inputs,
                                                                                           key=inputs,
                                                                                           value=inputs)
    x = Dropout(dropout)(x)
    res = x + inputs
    attention_out = LayerNormalization(epsilon=1e-6)(res)

    # Feed Forward Part
    x = Conv1D(filters=ff_dim, kernel_size=kernel_size, padding="same", activation="relu")(attention_out)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1, activation="relu")(x)
    res = x + inputs
    ff_out = LayerNormalization(epsilon=1e-6)(res)
    return ff_out


class EnergyBasePredictor:
    """
    Energy meteo model class
    """

    def __init__(self):
        """
        Constructor
        """
        file_path = abspath(dirname(__file__))
        self.model_basepath = f'{file_path}/trained_models/energy_models'
        self.scalers_basepath = f'{file_path}/trained_models/energy_scalers'
        self.model_name = 'Transformer_base'
        self.epoch = 0
        self.train_data_rate = 0.8
        self.test_data_rate = 0.2
        self.win_size = 0
        self.bsize = 0
        self.x_test_set = None
        self.x_train_set = None
        self.test_scalers = None
        self.train_scalers = None
        self.built_model = None
        self.y_pred = None
        self.df = None
        self.df_train = None
        self.df_test = None
        self.x_train = None
        self.x_test = None
        self.df_comparative = None
        self.predictor_columns = ['Ano', 'Mes', 'Dia', 'Hora', 'is_weekend', 'Solar_altitude',
                                  '0016A_Humedad_relativa', '0201D_Humedad_relativa', '0244X_Humedad_relativa',
                                  '0367_Humedad_relativa', '1025X_Humedad_relativa', '1056K_Humedad_relativa',
                                  '1074C_Humedad_relativa', '1111X_Humedad_relativa', '1186P_Humedad_relativa',
                                  '1279X_Humedad_relativa', '1387E_Humedad_relativa', '1390X_Humedad_relativa',
                                  '1466A_Humedad_relativa', '1475X_Humedad_relativa', '1719_Humedad_relativa',
                                  '2044B_Humedad_relativa', '2048A_Humedad_relativa', '2331_Humedad_relativa',
                                  '2734D_Humedad_relativa', '2777K_Humedad_relativa', '2873X_Humedad_relativa',
                                  '2891A_Humedad_relativa', '2946X_Humedad_relativa', '3104Y_Humedad_relativa',
                                  '3140Y_Humedad_relativa', '3266A_Humedad_relativa', '3475X_Humedad_relativa',
                                  '3504X_Humedad_relativa', '3526X_Humedad_relativa', '3562X_Humedad_relativa',
                                  '4096Y_Humedad_relativa', '4340_Humedad_relativa', '5390Y_Humedad_relativa',
                                  '5402_Humedad_relativa', '5582A_Humedad_relativa', '5598X_Humedad_relativa',
                                  '5612X_Humedad_relativa', '5906X_Humedad_relativa', '5972X_Humedad_relativa',
                                  '6045X_Humedad_relativa', '6172X_Humedad_relativa', '6268Y_Humedad_relativa',
                                  '6307X_Humedad_relativa', '6312E_Humedad_relativa', '7066Y_Humedad_relativa',
                                  '7195X_Humedad_relativa', '7275C_Humedad_relativa', '8025_Humedad_relativa',
                                  '8036Y_Humedad_relativa', '8177A_Humedad_relativa', '8270X_Humedad_relativa',
                                  '8486X_Humedad_relativa', '8500A_Humedad_relativa', '9016X_Humedad_relativa',
                                  '9257X_Humedad_relativa', '9301X_Humedad_relativa', '9352A_Humedad_relativa',
                                  '9377Y_Humedad_relativa', '9434_Humedad_relativa', '9573X_Humedad_relativa',
                                  '9677_Humedad_relativa', '9814X_Humedad_relativa', '9843A_Humedad_relativa',
                                  '9946X_Humedad_relativa', 'B275E_Humedad_relativa', 'B569X_Humedad_relativa',
                                  'B760X_Humedad_relativa', 'B925_Humedad_relativa', 'C148F_Humedad_relativa',
                                  'C249I_Humedad_relativa', 'C619Y_Humedad_relativa', 'C639M_Humedad_relativa',
                                  'C649R_Humedad_relativa', 'C659H_Humedad_relativa', 'C659M_Humedad_relativa',
                                  '0016A_Precipitacion', '0201D_Precipitacion', '0244X_Precipitacion',
                                  '0367_Precipitacion', '1025X_Precipitacion', '1056K_Precipitacion',
                                  '1074C_Precipitacion', '1111X_Precipitacion', '1186P_Precipitacion',
                                  '1279X_Precipitacion', '1387E_Precipitacion', '1390X_Precipitacion',
                                  '1466A_Precipitacion', '1475X_Precipitacion', '1719_Precipitacion',
                                  '2044B_Precipitacion', '2048A_Precipitacion', '2331_Precipitacion',
                                  '2734D_Precipitacion', '2777K_Precipitacion', '2873X_Precipitacion',
                                  '2891A_Precipitacion', '2946X_Precipitacion', '3104Y_Precipitacion',
                                  '3140Y_Precipitacion', '3266A_Precipitacion', '3475X_Precipitacion',
                                  '3504X_Precipitacion', '3526X_Precipitacion', '3562X_Precipitacion',
                                  '4096Y_Precipitacion', '4340_Precipitacion', '5390Y_Precipitacion',
                                  '5402_Precipitacion', '5582A_Precipitacion', '5598X_Precipitacion',
                                  '5612X_Precipitacion', '5906X_Precipitacion', '5972X_Precipitacion',
                                  '6045X_Precipitacion', '6172X_Precipitacion', '6268Y_Precipitacion',
                                  '6307X_Precipitacion', '6312E_Precipitacion', '7066Y_Precipitacion',
                                  '7195X_Precipitacion', '7275C_Precipitacion', '8025_Precipitacion',
                                  '8036Y_Precipitacion', '8177A_Precipitacion', '8270X_Precipitacion',
                                  '8486X_Precipitacion', '8500A_Precipitacion', '9016X_Precipitacion',
                                  '9257X_Precipitacion', '9301X_Precipitacion', '9352A_Precipitacion',
                                  '9377Y_Precipitacion', '9434_Precipitacion', '9573X_Precipitacion',
                                  '9677_Precipitacion', '9814X_Precipitacion', '9843A_Precipitacion',
                                  '9946X_Precipitacion', 'B275E_Precipitacion', 'B569X_Precipitacion',
                                  'B760X_Precipitacion', 'B925_Precipitacion', 'C148F_Precipitacion',
                                  'C249I_Precipitacion', 'C619Y_Precipitacion', 'C639M_Precipitacion',
                                  'C649R_Precipitacion', 'C659H_Precipitacion', 'C659M_Precipitacion', '0016A_Presion',
                                  '0201D_Presion', '0244X_Presion', '0367_Presion', '1025X_Presion', '1056K_Presion',
                                  '1074C_Presion', '1111X_Presion', '1186P_Presion', '1279X_Presion', '1387E_Presion',
                                  '1390X_Presion', '1466A_Presion', '1475X_Presion', '1719_Presion', '2044B_Presion',
                                  '2048A_Presion', '2331_Presion', '2734D_Presion', '2777K_Presion', '2873X_Presion',
                                  '2891A_Presion', '2946X_Presion', '3104Y_Presion', '3140Y_Presion', '3266A_Presion',
                                  '3475X_Presion', '3504X_Presion', '3526X_Presion', '3562X_Presion', '4096Y_Presion',
                                  '4340_Presion', '5390Y_Presion', '5402_Presion', '5582A_Presion', '5598X_Presion',
                                  '5612X_Presion', '5906X_Presion', '5972X_Presion', '6045X_Presion', '6172X_Presion',
                                  '6268Y_Presion', '6307X_Presion', '6312E_Presion', '7066Y_Presion', '7195X_Presion',
                                  '7275C_Presion', '8025_Presion', '8036Y_Presion', '8177A_Presion', '8270X_Presion',
                                  '8486X_Presion', '8500A_Presion', '9016X_Presion', '9257X_Presion', '9301X_Presion',
                                  '9352A_Presion', '9377Y_Presion', '9434_Presion', '9573X_Presion', '9677_Presion',
                                  '9814X_Presion', '9843A_Presion', '9946X_Presion', 'B275E_Presion', 'B569X_Presion',
                                  'B760X_Presion', 'B925_Presion', 'C148F_Presion', 'C249I_Presion', 'C619Y_Presion',
                                  'C639M_Presion', 'C649R_Presion', 'C659H_Presion', 'C659M_Presion',
                                  '0016A_Temperatura', '0201D_Temperatura', '0244X_Temperatura', '0367_Temperatura',
                                  '1025X_Temperatura', '1056K_Temperatura', '1074C_Temperatura', '1111X_Temperatura',
                                  '1186P_Temperatura', '1279X_Temperatura', '1387E_Temperatura', '1390X_Temperatura',
                                  '1466A_Temperatura', '1475X_Temperatura', '1719_Temperatura', '2044B_Temperatura',
                                  '2048A_Temperatura', '2331_Temperatura', '2734D_Temperatura', '2777K_Temperatura',
                                  '2873X_Temperatura', '2891A_Temperatura', '2946X_Temperatura', '3104Y_Temperatura',
                                  '3140Y_Temperatura', '3266A_Temperatura', '3475X_Temperatura', '3504X_Temperatura',
                                  '3526X_Temperatura', '3562X_Temperatura', '4096Y_Temperatura', '4340_Temperatura',
                                  '5390Y_Temperatura', '5402_Temperatura', '5582A_Temperatura', '5598X_Temperatura',
                                  '5612X_Temperatura', '5906X_Temperatura', '5972X_Temperatura', '6045X_Temperatura',
                                  '6172X_Temperatura', '6268Y_Temperatura', '6307X_Temperatura', '6312E_Temperatura',
                                  '7066Y_Temperatura', '7195X_Temperatura', '7275C_Temperatura', '8025_Temperatura',
                                  '8036Y_Temperatura', '8177A_Temperatura', '8270X_Temperatura', '8486X_Temperatura',
                                  '8500A_Temperatura', '9016X_Temperatura', '9257X_Temperatura', '9301X_Temperatura',
                                  '9352A_Temperatura', '9377Y_Temperatura', '9434_Temperatura', '9573X_Temperatura',
                                  '9677_Temperatura', '9814X_Temperatura', '9843A_Temperatura', '9946X_Temperatura',
                                  'B275E_Temperatura', 'B569X_Temperatura', 'B760X_Temperatura', 'B925_Temperatura',
                                  'C148F_Temperatura', 'C249I_Temperatura', 'C619Y_Temperatura', 'C639M_Temperatura',
                                  'C649R_Temperatura', 'C659H_Temperatura', 'C659M_Temperatura', '0016A_Viento_pred',
                                  '0201D_Viento_pred', '0244X_Viento_pred', '0367_Viento_pred', '1025X_Viento_pred',
                                  '1056K_Viento_pred', '1074C_Viento_pred', '1111X_Viento_pred', '1186P_Viento_pred',
                                  '1279X_Viento_pred', '1387E_Viento_pred', '1390X_Viento_pred', '1466A_Viento_pred',
                                  '1475X_Viento_pred', '1719_Viento_pred', '2044B_Viento_pred', '2048A_Viento_pred',
                                  '2331_Viento_pred', '2734D_Viento_pred', '2777K_Viento_pred', '2873X_Viento_pred',
                                  '2891A_Viento_pred', '2946X_Viento_pred', '3104Y_Viento_pred', '3140Y_Viento_pred',
                                  '3266A_Viento_pred', '3475X_Viento_pred', '3504X_Viento_pred', '3526X_Viento_pred',
                                  '3562X_Viento_pred', '4096Y_Viento_pred', '4340_Viento_pred', '5390Y_Viento_pred',
                                  '5402_Viento_pred', '5582A_Viento_pred', '5598X_Viento_pred', '5612X_Viento_pred',
                                  '5906X_Viento_pred', '5972X_Viento_pred', '6045X_Viento_pred', '6172X_Viento_pred',
                                  '6268Y_Viento_pred', '6307X_Viento_pred', '6312E_Viento_pred', '7066Y_Viento_pred',
                                  '7195X_Viento_pred', '7275C_Viento_pred', '8025_Viento_pred', '8036Y_Viento_pred',
                                  '8177A_Viento_pred', '8270X_Viento_pred', '8486X_Viento_pred', '8500A_Viento_pred',
                                  '9016X_Viento_pred', '9257X_Viento_pred', '9301X_Viento_pred', '9352A_Viento_pred',
                                  '9377Y_Viento_pred', '9434_Viento_pred', '9573X_Viento_pred', '9677_Viento_pred',
                                  '9814X_Viento_pred', '9843A_Viento_pred', '9946X_Viento_pred', 'B275E_Viento_pred',
                                  'B569X_Viento_pred', 'B760X_Viento_pred', 'B925_Viento_pred', 'C148F_Viento_pred',
                                  'C249I_Viento_pred', 'C619Y_Viento_pred', 'C639M_Viento_pred', 'C649R_Viento_pred',
                                  'C659H_Viento_pred', 'C659M_Viento_pred', 'Asturias', 'Cantabria', 'Navarra',
                                  'País Vasco', 'Cataluña', 'Aragón', 'Galicia', 'Islas Baleares', 'La Rioja',
                                  'Valencia', 'Castilla y León', 'Castilla La Mancha', 'Extremadura', 'Andalucía',
                                  'Murcia', 'Madrid']
        self.target_columns = []
        self.cols_corr = None
        self.cols_corr_new = None
        self.transformers_num_heads = 10
        self.transformers_ff_dim = 512
        self.transformers_num_blocks = 1
        self.transformers_dropout = 0.25
        self.transformers_mlp_units = [512, 128, 32]
        self.fit_history = None

    def load_parquet_data(self):
        """
        Load parquet data
        """
        self.df = dm.load_df_from_parquet(energy_meteo_files_path + '/energy_meteo_data.parquet')
        self.df = dm.append_weekend_column(self.df)
        self.append_data_energy_days_before()
        df_train = dm.filter_df_by_date(self.df, '2022-09-18', '2023-09-17')
        self.train_data_rate = df_train.shape[0] / self.df.shape[0]
        self.test_data_rate = 1 - self.train_data_rate
        del df_train

    def append_data_energy_days_before(self):
        """
        Append data energy days before
        """
        for col in self.target_columns:
            self.df[f'{col}_1day_before'] = self.df[col].shift(1 * 24).fillna(self.df[col])
            self.df[f'{col}_2days_before'] = self.df[col].shift(2 * 24).fillna(self.df[col])
            self.df[f'{col}_3days_before'] = self.df[col].shift(3 * 24).fillna(self.df[col])

    def print_parquet_data(self):
        """
        Print parquet data
        """
        print(self.df.head())
        print(self.df.info())
        print(self.df.describe())
        print(self.df.columns)

    #
    @abc.abstractmethod
    def get_custom_correlation_names(self):
        """
        Retrieve of custom correlation new names
        :return:
        """
        pass

    def generate_correlations(self, all_cols=True, annot=False):
        """
        Generate correlations
        :param all_cols: If True, generate correlations for all columns. If False, generate correlations for a subset of
        :param annot: If True, show the values in the plot. If False, do not show the values in the plot
        columns
        """
        if all_cols:
            self.cols_corr = self.predictor_columns + self.target_columns
            self.cols_corr_new = ['Año' if col == 'Ano' else col for col in self.cols_corr]
            self.cols_corr_new = ['Día' if col == 'Dia' else col for col in self.cols_corr_new]
            self.cols_corr_new = [col.replace('ion', 'ión') if 'ion' in col else col for col in self.cols_corr_new]
            self.cols_corr_new = [col.replace('Eolica', 'Eólica') if 'Eolica' in col else col for col in self.cols_corr_new]
        else:
            self.get_custom_correlation_names()
        dm.generate_correlations(self.df, self.cols_corr, self.cols_corr_new, annot=annot)

    def build_model(self):
        mlp_dropout = self.transformers_dropout
        inputs = Input(shape=[None, len(self.predictor_columns)])
        x = inputs
        for _ in range(self.transformers_num_blocks):
            x = transformer_encoder(x, self.transformers_num_heads, self.transformers_ff_dim, self.transformers_dropout)

        x = GlobalAveragePooling1D(data_format="channels_last")(x)
        for dim in self.transformers_mlp_units:
            x = Dense(dim, activation="relu", kernel_initializer=HeNormal())(x)
            x = Dropout(mlp_dropout)(x)
        out_dense = []
        for i in range(len(self.target_columns)):
            out_dense.append(Dense(1, activation='linear', kernel_initializer=HeNormal())(x))
        output = Concatenate()(out_dense)

        self.built_model = Model(inputs=inputs, outputs=output)
        self.built_model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
        self.built_model.summary()

    def get_timeseries_ds(self, shuffle=True, shuffle_buffer=1000):
        all_cols = self.predictor_columns + self.target_columns
        ds_list = [tf.data.Dataset.from_tensor_slices(self.x_train[all_cols].values),
                   tf.data.Dataset.from_tensor_slices(self.x_test[all_cols].values)]

        for idx, ds in enumerate(ds_list):
            ds = ds.window(self.win_size + 1, shift=1, drop_remainder=True)
            ds = ds.flat_map(lambda window: window.batch(self.win_size + 1))
            ds = ds.map(lambda w: (w[:-1, :len(self.predictor_columns)], w[-1, len(self.predictor_columns):]))
            # Only shuffle the training set not the test set
            if shuffle and idx == 0:
                ds = ds.shuffle(shuffle_buffer)
            ds_list[idx] = ds.batch(self.bsize).prefetch(tf.data.AUTOTUNE)

            x_batch, y_batch = next(iter(ds_list[idx].take(1)))
            print("x_batch shape:", x_batch.shape)
            print("y_batch shape:", y_batch.shape)

        return ds_list[0], ds_list[1]

    def prepare_train_test_data(self):
        """
        Prepare train and test data
        """
        train_data_size = int(self.df.shape[0] * self.train_data_rate)
        test_data_size = int(self.df.shape[0] * self.test_data_rate)
        self.df_train = self.df[:train_data_size]
        self.df_test = self.df[train_data_size:(train_data_size + test_data_size)]
        self.x_train, in_train_scaler, out_train_scaler = scale_dataframe(self.df_train, self.predictor_columns,
                                                                          self.target_columns)
        self.x_test, in_test_scaler, out_test_scaler = scale_dataframe(self.df_test, self.predictor_columns,
                                                                       self.target_columns)
        self.train_scalers = (in_train_scaler, out_train_scaler)
        self.test_scalers = (in_test_scaler, out_test_scaler)
        self.x_train_set, self.x_test_set = self.get_timeseries_ds(shuffle=True)

    def fit(self):
        """
        Fit the model
        :return: History object generated by fit
        """
        self.fit_history = self.built_model.fit(self.x_train_set, validation_data=self.x_test_set, epochs=self.epoch,
                                                batch_size=self.bsize,
                                                callbacks=[EarlyStopperCallback(mae_threshold=0.2)])

    def predict(self):
        """
        Predict
        :return: Predictions
        """
        self.y_pred = self.built_model.predict(self.x_test_set, batch_size=self.bsize)
        if len(self.y_pred.shape) > 2 and self.y_pred.shape[2] == 1:
            self.y_pred = self.y_pred[:, :, 0]
        self.y_pred = self.test_scalers[1].inverse_transform(self.y_pred)
        x_test_scaled = self.x_test[self.target_columns].iloc[-self.y_pred.shape[0]:].reset_index(drop=True)
        x_test_real = pd.DataFrame(self.test_scalers[1].inverse_transform(x_test_scaled),
                                   columns=self.target_columns)

        self.df_comparative = self.df_test[['Ano', 'Mes', 'Dia', 'Hora', 'Minuto']].iloc[-self.y_pred.shape[0]:] \
            .reset_index(drop=True)
        for idx, column in enumerate(self.target_columns):
            self.df_comparative[f'{column}_pred'] = self.y_pred[:, idx]
            self.df_comparative[f'{column}_test_real'] = x_test_real[column]
        return self.df_comparative

    @abc.abstractmethod
    def plot_results(self, pred_to_plot=336):
        """
        Plot the results. Abstract method to plot the results in the child class
        :param pred_to_plot: Number of predictions to plot. Default is 336 hours (2 weeks)
        """
        pass

    def save_model(self):
        """
        Save the built model and scalers
        """
        self.built_model.save(f'{self.model_basepath}/{self.model_name}.keras')
        joblib.dump(self.train_scalers[0], f'{self.scalers_basepath}/{self.model_name}_in.joblib')
        joblib.dump(self.train_scalers[1], f'{self.scalers_basepath}/{self.model_name}_out.joblib')

    def load_model(self):
        """
        Load the built model and scalers
        """
        self.built_model = keras.models.load_model(f'{self.model_basepath}/{self.model_name}.keras',
                                                   custom_objects={"limit_output": limit_output,
                                                                   "RobustScalerSerializable": RobustScalerSerializable
                                                                   })
        self.train_scalers = (joblib.load(f'{self.scalers_basepath}/{self.model_name}_in.joblib'),
                              joblib.load(f'{self.scalers_basepath}/{self.model_name}_out.joblib'))

    def evaluate(self):
        """
        Calculate MAPE (Mean Absolute Percentage Error)
        :return: The MAPE values for each target column
        """
        df_mape = pd.DataFrame()
        df_mape['datetime'] = pd.to_datetime({'year': self.df_comparative['Ano'], 'month': self.df_comparative['Mes'],
                                              'day': self.df_comparative['Dia']})
        df_mape_daily = pd.DataFrame()
        for column in self.target_columns:
            df_mape_col = df_mape.copy()
            # Add a small value to avoid division by zero
            df_mape_col[f'{column}_real'] = np.abs(self.df_comparative[f'{column}_test_real'].values) + 0.99
            df_mape_col[f'{column}_pred'] = np.abs(self.df_comparative[f'{column}_pred'].values) + 0.99
            # calculates the MAPE grouping by date
            df_mape_col_daily = df_mape_col.groupby(['datetime']).apply(
                lambda x: (abs((x[f'{column}_real'] - x[f'{column}_pred']) / x[f'{column}_real'])).mean() * 100)
            df_mape_col_daily = df_mape_col_daily.reset_index()
            df_mape_col_daily.name = f'{column}_daily_mape'
            # append datetime if not exists
            if 'datetime' not in df_mape_daily.columns:
                df_mape_daily['datetime'] = df_mape_col_daily['datetime']
            df_mape_daily[f'{column}_daily_mape'] = df_mape_col_daily[0]

        print('Mean Absolute Percentage Error (MAPE) last month:')
        print(df_mape_daily.iloc[-300:])

        return df_mape_daily

    def evaluate_nrmse(self):
        """
        Calculate NRMSE (Normalized Root Mean Squared Error)
        :return: The NRMSE values for each target column
        """
        df_nrmse = pd.DataFrame()
        df_nrmse['datetime'] = pd.to_datetime({'year': self.df_comparative['Ano'], 'month': self.df_comparative['Mes'],
                                              'day': self.df_comparative['Dia']})
        df_nrmse_daily = pd.DataFrame()
        for column in self.target_columns:
            df_nrmse_col = df_nrmse.copy()
            df_nrmse_col[f'{column}_real'] = self.df_comparative[f'{column}_test_real'].values
            df_nrmse_col[f'{column}_pred'] = self.df_comparative[f'{column}_pred'].values
            df_nrmse_col_daily = df_nrmse_col.groupby(['datetime']) \
                .apply(
                    lambda x:
                    (np.sqrt(np.mean((x[f'{column}_real'] - x[f'{column}_pred']) ** 2))) /
                    np.abs(np.max(x[f'{column}_real']) - np.min(x[f'{column}_real']))
                )

            df_nrmse_col_daily = df_nrmse_col_daily.reset_index()
            df_nrmse_col_daily.name = f'{column}_daily_mape'
            # append datetime if not exists
            if 'datetime' not in df_nrmse_daily.columns:
                df_nrmse_daily['datetime'] = df_nrmse_col_daily['datetime']
            df_nrmse_daily[f'{column}_daily_mape'] = df_nrmse_col_daily[0]
        last_days = 30
        print(f'Normalized Root Mean Squared Error (NRMSE) last {last_days} days:')
        print(df_nrmse_daily.iloc[-last_days:])
        # mean NRMSE for the last 30 days for all columns
        print(f'Mean NRMSE last {last_days} days for all columns:')
        print(df_nrmse_daily.iloc[-last_days:].mean())
        print(f'Median NRMSE last {last_days} days for all columns:')
        print(df_nrmse_daily.iloc[-last_days:].median())

        return df_nrmse_daily

    def plot_mape_values(self, df_mape: pd.DataFrame, model_title: str, labels: list, logarithmic_scale=True,
                         ylim_bottom=None, scaled_values=False):
        df_mape['datetime'] = pd.to_datetime(df_mape['datetime'])
        df_mape.set_index('datetime', inplace=True)

        # to plot also scaled values
        if len(df_mape.columns) > 1 and scaled_values:
            scaler = MinMaxScaler()
            cols_not_scaled = df_mape.columns
            cols_scaled = [item + '_scaled' for item in cols_not_scaled]
            df_mape_scal = pd.DataFrame(scaler.fit_transform(df_mape), columns=cols_scaled)
            df_mape_scal.index = df_mape.index
            # concatenate the scaled values to the original dataframe
            df_mape = pd.concat([df_mape, df_mape_scal], axis=1)
            array_cols = np.array([cols_not_scaled, cols_scaled], dtype=object)
            titles = [f'MAPE para {model_title}', f'MAPE para {model_title} escalado']
        else:
            array_cols = np.array([df_mape.columns], dtype=object)
            titles = [f'MAPE para {model_title}']

        cm.plot_series(df_mape, array_cols, titles, ['Fecha'] * array_cols.shape[1], ['MAPE'] * array_cols.shape[1],
                       labels=labels, save=False, file_name=f'{self.model_name}_mape', append_datetime=False,
                       logarithmic_scale=logarithmic_scale, ylim_bottom=ylim_bottom)

    def plot_fit_history(self, title_detail=None, filename_suffix=None, min_loss=0.2, max_loss=0.7):
        """
        Plot the fit history
        """
        file_name = f'fit_history_{self.model_name}{filename_suffix}' \
            if filename_suffix is not None else f'fit_history_{self.model_name}'
        cm.plot_fit_history(self.fit_history, save=True, file_name=file_name, title_detail=title_detail,
                            min_loss=min_loss,
                            max_loss=max_loss)
