import os
from datetime import datetime
from enum import Enum
from typing import List

import joblib
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from huggingface_hub import snapshot_download
from numpy import datetime64

import common.dataframe_manager as dm
from common.robust_scaler_serializable import RobustScalerSerializable
from dtos_objects import EmissionsPrediction, EmissionsPredicted

access_token = os.getenv('MODEL_DATA_REPO_TOKEN')


class EmissionPredictionType(Enum):
    CARBON = 'carbon'
    COMBINED_CYCLE = 'combined_cycle'
    COGENERATION = 'cogeneration'
    DIESEL_ENGINES = 'diesel_engines'
    GAS_TURBINE = 'gas_turbine'
    STEAM_TURBINE = 'steam_turbine'

    @classmethod
    def value_from_string(cls, value):
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"No member found with value '{value}' in {cls.__name__}")


predictor_target_cols = {
    EmissionPredictionType.CARBON: ['Carbon_gen', 'Carbon_emi', 'Carbón_gen', 'Carbón_emi'],
    EmissionPredictionType.COMBINED_CYCLE: ['Ciclo_combinado_gen', 'Ciclo_combinado_emi', 'Ciclo_combinado_gen',
                                            'Ciclo_combinado_emi'],
    EmissionPredictionType.COGENERATION: ['Cogeneracion_y_residuos_gen', 'Cogeneracion_y_residuos_emi',
                                          'Cogeneración_y_residuos_gen', 'Cogeneración_y_residuos_emi'],
    EmissionPredictionType.DIESEL_ENGINES: ['Motores_diesel_gen', 'Motores_diesel_emi', 'Motores_diésel_gen',
                                            'Motores_diésel_emi'],
    EmissionPredictionType.GAS_TURBINE: ['Turbina_de_gas_gen', 'Turbina_de_gas_emi', 'Turbina_de_gas_gen',
                                         'Turbina_de_gas_emi'],
    EmissionPredictionType.STEAM_TURBINE: ['Turbina_de_vapor_gen', 'Turbina_de_vapor_emi',
                                           'Turbina_de_vapor_gen', 'Turbina_de_vapor_emi']
}


def retrieve_emission_pred_type(col_name):
    """
    Retrieve emission prediction type
    :param col_name: The column name
    :return: The emission prediction type
    """
    pred_type_out = None
    for pred_type in EmissionPredictionType:
        if col_name in predictor_target_cols[pred_type]:
            pred_type_out = pred_type
    return pred_type_out


def retrieve_predictor_columns(model_name, target_columns: list):
    predictor_columns = []
    if model_name == 'Transformer_dem_model' or model_name == 'Transformer_gen_model' \
            or model_name == 'Transformer_emi_model':
        predictor_columns = ['Ano', 'Mes', 'Dia', 'Hora', 'is_weekend', 'Solar_altitude',
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
        # Additional columns with the previous target values
        predictor_columns = predictor_columns + [col + '_1day_before' for col in target_columns]
        predictor_columns = predictor_columns + [col + '_2days_before' for col in target_columns]
        predictor_columns = predictor_columns + [col + '_3days_before' for col in target_columns]
    elif model_name == 'ANN_simu_emi_model':
        predictor_columns = []
    return predictor_columns


def retrieve_target_columns(model_name):
    """
    Retrieve target columns
    :param model_name: The model name
    :return: The target columns
    """
    target_columns = []
    if model_name == 'Transformer_dem_model':
        target_columns = ['Real']
    elif model_name == 'Transformer_gen_model':
        target_columns = ['Eolica_gen', 'Hidraulica_gen', 'Solar_fotovoltaica_gen', 'Solar_termica_gen']
    elif model_name == 'Transformer_emi_model':
        target_columns = ['Carbon_emi', 'Ciclo_combinado_emi', 'Motores_diesel_emi', 'Turbina_de_vapor_emi',
                          'Turbina_de_gas_emi', 'Cogeneracion_y_residuos_emi']
    elif model_name == 'ANN_simu_emi_model':
        target_columns = ['Carbon_emi', 'Ciclo_combinado_emi', 'Cogeneracion_y_residuos_emi', 'Motores_diesel_emi',
                          'Turbina_de_gas_emi', 'Turbina_de_vapor_emi']
    return target_columns


class Predictor:
    """
    Predictor class
    """
    # Class attributes
    df = None

    def __init__(self, model_name: str, win_size: int = 60, bsize: int = 60):
        self.x_pred_set = None
        self.x_predict = None
        self.df_pred = None
        self.model_name = model_name
        self.target_columns = retrieve_target_columns(self.model_name)
        self.predictor_columns = retrieve_predictor_columns(self.model_name, self.target_columns)
        self.real_columns = ['Real'] if self.model_name == 'Transformer_dem_model' else self.target_columns
        self.y_pred = None
        self.win_size = win_size
        self.bsize = bsize
        self.scalers = None
        self.model = None
        self.emissions_models = {}
        self.emission_scalers = {}
        self.model_basepath = snapshot_download(repo_id="JoseManuelR/tfm-models-repo", use_auth_token=access_token)
        data_path = snapshot_download(repo_id="JoseManuelR/tfm-data-repo", repo_type="dataset",
                                      use_auth_token=access_token)
        self.data_filepath = f'{data_path}/energy_meteo_data.parquet'

    def load_parquet_data(self):
        """
        Load parquet data
        """
        Predictor.df = pd.read_parquet(self.data_filepath)
        Predictor.df = dm.append_datetime_from_date_and_time_cols(Predictor.df)
        Predictor.df = dm.append_weekend_column(self.df)

    def append_data_energy_days_before(self):
        """
        Append data energy days before
        """
        for col in self.target_columns:
            Predictor.df[f'{col}_1day_before'] = Predictor.df[col].shift(1 * 24).fillna(Predictor.df[col])
            Predictor.df[f'{col}_2days_before'] = Predictor.df[col].shift(2 * 24).fillna(Predictor.df[col])
            Predictor.df[f'{col}_3days_before'] = Predictor.df[col].shift(3 * 24).fillna(Predictor.df[col])

    def load_model(self):
        """
        Load the built model and scalers
        """
        if self.model_name == 'ANN_simu_emi_model':
            for pred_type in EmissionPredictionType:
                self.emissions_models[pred_type] = keras.models.load_model(
                    f'{self.model_basepath}/{self.model_name}_{pred_type.value}.keras',
                    custom_objects={"RobustScalerSerializable": RobustScalerSerializable})
                self.emission_scalers[pred_type] = (
                    joblib.load(f'{self.model_basepath}/{self.model_name}_{pred_type.value}_in.joblib'),
                    joblib.load(f'{self.model_basepath}/{self.model_name}_{pred_type.value}_out.joblib'))
        else:
            self.model = keras.models.load_model(f'{self.model_basepath}/{self.model_name}.keras',
                                                 custom_objects={"RobustScalerSerializable": RobustScalerSerializable})
            self.scalers = (joblib.load(f'{self.model_basepath}/{self.model_name}_in.joblib'),
                            joblib.load(f'{self.model_basepath}/{self.model_name}_out.joblib'))

    def print_parquet_data(self):
        """
        Print parquet data
        """
        print(Predictor.df.head())
        print(Predictor.df.info())
        print(Predictor.df.describe())
        print(Predictor.df.columns)

    def get_timeseries_ds(self, pred_cols=None):
        if pred_cols is None:
            pred_cols = self.predictor_columns
        ds = tf.data.Dataset.from_tensor_slices(self.x_predict[pred_cols].values)
        ds = ds.window(self.win_size + 1, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda x: x.batch(self.win_size + 1))
        ds = ds.batch(self.bsize)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    def scale_dataframe(self, df, pred_cols=None):
        if pred_cols is None:
            pred_cols = self.predictor_columns
            scaler = self.scalers[0]
        else:
            scaler = self.emission_scalers[retrieve_emission_pred_type(pred_cols)][0]
            pred_cols = [pred_cols]
        normalized_array_in = scaler.fit_transform(df[pred_cols])
        return pd.DataFrame(normalized_array_in, columns=[pred_cols])

    def prepare_prediction_data(self, start_date: datetime64, end_date: datetime64):
        """
        Prepare train and test data
        """
        self.df_pred = Predictor.df[(Predictor.df['datetime'] >= start_date) & (Predictor.df['datetime'] <= end_date)]
        self.x_predict = self.scale_dataframe(self.df_pred)
        self.x_pred_set = self.get_timeseries_ds()

    def predict(self, start_date: datetime, end_date: datetime):
        start_date = np.datetime64(start_date)
        end_date = np.datetime64(end_date)
        self.prepare_prediction_data(start_date - pd.Timedelta(hours=self.win_size), end_date)
        self.y_pred = self.model.predict(self.x_pred_set, batch_size=self.bsize)
        self.y_pred = self.scalers[1].inverse_transform(self.y_pred)
        df_out = pd.DataFrame()
        df_out['datetime'] = Predictor.df[(Predictor.df['datetime'] >= start_date) &
                                          (Predictor.df['datetime'] <= end_date)]['datetime']
        df_out.reset_index(drop=True, inplace=True)
        df_out = df_out.iloc[-self.y_pred.shape[0]:]
        for idx, (tar_col, real_col) in enumerate(zip(self.target_columns, self.real_columns)):
            df_out[f'{tar_col}_pred'] = self.y_pred[:, idx]
            df_out[f'{real_col}_real'] = Predictor.df[(Predictor.df['datetime'] >= start_date) &
                                                      (Predictor.df['datetime'] <= end_date)][real_col].values
        return df_out, self.retrieve_mape(df_out)

    def predict_emissions(self, energy_to_redict: EmissionsPrediction):
        start_date = np.datetime64(energy_to_redict.date_instant)
        self.df_pred = Predictor.df[(Predictor.df['datetime'] >= start_date - pd.Timedelta(hours=self.win_size)) &
                                    (Predictor.df['datetime'] <= start_date)]
        self.df_pred.iloc[-1, self.df_pred.columns.get_loc('Carbon_gen')] = energy_to_redict.Carbon_gen
        self.df_pred.iloc[
            -1, self.df_pred.columns.get_loc('Ciclo_combinado_gen')] = energy_to_redict.Ciclo_combinado_gen
        self.df_pred.iloc[-1, self.df_pred.columns.get_loc('Motores_diesel_gen')] = energy_to_redict.Motores_diesel_gen
        self.df_pred.iloc[-1, self.df_pred.columns.get_loc('Turbina_de_gas_gen')] = energy_to_redict.Turbina_de_gas_gen
        self.df_pred.iloc[
            -1, self.df_pred.columns.get_loc('Turbina_de_vapor_gen')] = energy_to_redict.Turbina_de_vapor_gen
        self.df_pred.iloc[-1, self.df_pred.columns.get_loc(
            'Cogeneracion_y_residuos_gen')] = energy_to_redict.Cogeneracion_y_residuos_gen

        # initialise the y_pred numpy array empty
        self.y_pred = np.empty((1, 0))
        for pred_type in EmissionPredictionType:
            pred_cols = predictor_target_cols.get(pred_type)[0]
            self.x_predict = self.scale_dataframe(self.df_pred, pred_cols)
            self.x_pred_set = self.get_timeseries_ds(pred_cols=pred_cols)
            y_prediction = self.emissions_models[pred_type].predict(self.x_pred_set, batch_size=self.bsize)
            y_prediction = y_prediction[:, :, 0]
            y_prediction = self.emission_scalers[pred_type][1]. \
                inverse_transform(y_prediction)[-1].reshape(-1, 1)[-1].reshape(1, -1)
            self.y_pred = np.concatenate((self.y_pred, y_prediction), axis=1)

        df_out = pd.DataFrame()
        df_out['date_instant'] = Predictor.df[(Predictor.df['datetime'] == start_date)]['datetime']
        df_out.reset_index(drop=True, inplace=True)
        df_out = df_out.iloc[-self.y_pred.shape[0]:]
        for idx, (tar_col, real_col) in enumerate(zip(self.target_columns, self.real_columns)):
            df_out[f'{tar_col}_pred'] = np.clip(self.y_pred[:, idx], a_min=0, a_max=None)
            df_out[f'{real_col}_real'] = Predictor.df[(Predictor.df['datetime'] == start_date)][real_col].values
        return [EmissionsPredicted(**row) for row in df_out.to_dict(orient="records")]

    def get_predictor_columns(self, start_date: datetime, end_date: datetime) -> List[EmissionsPrediction]:
        """
        Get predictor columns for a given date range
        :param start_date: The start date
        :param end_date: The end date
        :return: The predictor columns for the given date range
        """
        start_date = np.datetime64(start_date)
        end_date = np.datetime64(end_date)
        df_out = pd.DataFrame()
        cols_energy_predictor = [pred_col[0] for pred_col in predictor_target_cols.values()]
        df_out['date_instant'] = Predictor.df[(Predictor.df['datetime'] >= start_date) &
                                              (Predictor.df['datetime'] <= end_date)]['datetime']
        df_out['date_instant'] = pd.to_datetime(df_out['date_instant'])
        for col in cols_energy_predictor:
            df_out[col] = Predictor.df[(Predictor.df['datetime'] >= start_date) &
                                       (Predictor.df['datetime'] <= end_date)][col].values

        return [EmissionsPrediction(**row) for row in df_out.to_dict(orient="records")]

    def retrieve_mape(self, df: pd.DataFrame):
        """
        Calculate MAPE (Mean Absolute Percentage Error)
        :return: The MAPE values for each target column
        """
        df_mape = {}
        for idx, (tar_col, real_col) in enumerate(zip(self.target_columns, self.real_columns)):
            # Add a small value to avoid division by zero
            y_real = df[f'{tar_col}_pred'].values + 0.000001
            y_pred = df[f'{real_col}_real'].values + 0.000001
            mape = np.mean(np.abs((y_real - y_pred[:]) / y_real)) * 100
            df_mape[f'{tar_col}_MAPE'] = mape
        # Calculate de general MAPE
        cols_pred = [item + '_pred' for item in self.target_columns]
        cols_real = [item + '_real' for item in self.real_columns]
        y_real = df[cols_pred].values + 0.000001
        y_pred = df[cols_real].values + 0.000001
        df_mape['General_MAPE'] = np.mean(np.abs((y_real - y_pred) / y_real)) * 100

        return df_mape


if __name__ == '__main__':
    pred = Predictor('Transformer_gen_model')
    pred.load_parquet_data()
    pred.print_parquet_data()
    pred.load_model()
