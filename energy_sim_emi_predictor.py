import numpy as np
import pandas as pd

import common.chart_manager as cm
from enum import Enum

from keras import Input, Model
from keras.src.initializers import HeNormal
from keras.src.layers import Dense, Dropout
from keras.src.optimizers import Adam

from energy_base_predictor import EnergyBasePredictor


class EmissionPredictionType(Enum):
    CARBON = 'carbon'
    COMBINED_CYCLE = 'combined_cycle'
    COGENERATION = 'cogeneration'
    DIESEL_ENGINES = 'diesel_engines'
    GAS_TURBINE = 'gas_turbine'
    STEAM_TURBINE = 'steam_turbine'


class EnergySimulateEmiPredictor(EnergyBasePredictor):
    """
    Energy simulate emission predictor
    """

    def __init__(self, prediction_type=EmissionPredictionType.CARBON):
        """
        Constructor
        """
        super().__init__()
        self.prediction_type = prediction_type
        self.model_name = 'ANN_simu_emi_model_' + self.prediction_type.value
        self.predictor_target_cols = {
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
        self.predictor_columns = [self.predictor_target_cols.get(self.prediction_type)[0]]
        self.target_columns = [self.predictor_target_cols.get(self.prediction_type)[1]]
        self.epoch = 15
        self.win_size = 1
        self.bsize = 256

    def get_custom_correlation_names(self):
        """
        Get custom correlation names
        """
        self.cols_corr = self.predictor_columns + self.target_columns
        self.cols_corr_new = [self.predictor_target_cols.get(self.prediction_type)[2],
                              self.predictor_target_cols.get(self.prediction_type)[3]]

    def build_model(self):
        mlp_dropout = self.transformers_dropout
        ffn_layers = [256, 128]
        inputs = Input(shape=[None, len(self.predictor_columns)])
        x = inputs
        for dim in ffn_layers:
            x = Dense(dim, activation="relu", kernel_initializer=HeNormal())(x)
            x = Dropout(mlp_dropout)(x)
        out_dense = Dense(1, activation='linear', kernel_initializer=HeNormal())(x)

        self.built_model = Model(inputs=inputs, outputs=out_dense)
        self.built_model.compile(optimizer=Adam(), loss='mae', metrics=['mae'])
        self.built_model.summary()

    def plot_results(self, pred_to_plot=336):
        """
        Plot results
        :param pred_to_plot: The number of predictions to plot
        :return: None
        """
        cm.plot_series(self.df_comparative.iloc[-pred_to_plot:],
                       np.array([['Carbon_emi_pred', 'Carbon_emi_test_real'],
                                 ['Ciclo_combinado_emi_pred', 'Ciclo_combinado_emi_test_real']], dtype=object),
                       ['Carbón', 'Ciclo combinado'], ['Fecha'] * 2, ['Emisiones simuladas en t$CO_2$ equivalente'] * 2,
                       save=False, file_name=f'Carbon_sim_emi_and_Ciclo_combinado_sim_emi_test_vs_pred')
        cm.plot_series(self.df_comparative.iloc[-pred_to_plot:],
                       np.array([['Motores_diesel_emi_pred', 'Motores_diesel_emi_test_real'],
                                 ['Turbina_de_vapor_emi_pred', 'Turbina_de_vapor_emi_test_real']], dtype=object),
                       ['Motores diésel', 'Turbina de vapor'], ['Fecha'] * 2,
                       ['Emisiones simuladas en t$CO_2$ equivalente'] * 2,
                       save=False, file_name=f'Motores_diesel_sim_emi_and_Turbina_de_vapor_sim_emi_test_vs_pred')
        cm.plot_series(self.df_comparative.iloc[-pred_to_plot:],
                       np.array([['Turbina_de_gas_emi_pred', 'Turbina_de_gas_emi_test_real'],
                                 ['Cogeneracion_y_residuos_emi_pred', 'Cogeneracion_y_residuos_emi_test_real']],
                                dtype=object), ['Turbina de gas', 'Cogeneración y residuos'], ['Fecha'] * 2,
                       ['Emisiones simuladas en t$CO_2$ equivalente'] * 2, save=False,
                       file_name=f'Turbina_de_gas_sim_emi_and_Cogeneracion_y_residuos_sim_emi_test_vs_pred')


if __name__ == '__main__':
    # Train, predict and evaluate
    # for prediction_type in EmissionPredictionType:
    #     esep = EnergySimulateEmiPredictor(prediction_type=prediction_type)
    #     esep.load_parquet_data()
    #     esep.print_parquet_data()
    #     esep.build_model()
    #     esep.prepare_train_test_data()
    #     esep.fit()
    #     esep.plot_fit_history(title_detail='con ANN de simulación de emisiones de $CO_2$',
    #                           filename_suffix=f'_{prediction_type.value}', min_loss=0.0, max_loss=1.0)
    #     esep.predict()
    #     esep.evaluate()
    #     esep.save_model()
    # Correlations
    # esep.generate_correlations(all_cols=False, annot=True)
    # Load, predict and evaluate
    df_plot = pd.DataFrame()
    df_mape = pd.DataFrame()
    df_nrmse = pd.DataFrame()
    pred_to_plot = 720
    for prediction_type in EmissionPredictionType:
        esep = EnergySimulateEmiPredictor(prediction_type=prediction_type)
        esep.load_parquet_data()
        esep.print_parquet_data()
        esep.prepare_train_test_data()
        esep.load_model()
        df_predicted = esep.predict()
        if df_plot.empty:
            df_plot = df_predicted[['Ano', 'Mes', 'Dia', 'Hora', 'Minuto']]
        # concatenate all columns from df_predicted except Ano, Mes, Dia, Hora, Minuto if not empty
        df_plot = pd.concat([df_plot, df_predicted.iloc[:, 5:]], axis=1)
        df_mape_eval = esep.evaluate()
        if df_mape.empty:
            df_mape = df_mape_eval[['datetime']]
        df_mape = pd.concat([df_mape, df_mape_eval.iloc[:, 1:]], axis=1)
        df_nrmse_eval = esep.evaluate_nrmse()
        if df_nrmse.empty:
            df_nrmse = df_nrmse_eval[['datetime']]
        df_nrmse = pd.concat([df_nrmse, df_nrmse_eval.iloc[:, 1:]], axis=1)

    cm.plot_series(df_plot.iloc[-pred_to_plot:],
                   np.array([['Carbon_emi_pred', 'Carbon_emi_test_real'],
                             ['Ciclo_combinado_emi_pred', 'Ciclo_combinado_emi_test_real']], dtype=object),
                   ['Carbón', 'Ciclo combinado'], ['Fecha'] * 2, ['Emisiones en t$CO_2$ equivalente'] * 2,
                   save=False, file_name=f'Carbon_emi_and_Ciclo_combinado_emi_test_vs_pred')
    cm.plot_series(df_plot.iloc[-pred_to_plot:],
                   np.array([['Motores_diesel_emi_pred', 'Motores_diesel_emi_test_real'],
                             ['Turbina_de_vapor_emi_pred', 'Turbina_de_vapor_emi_test_real']], dtype=object),
                   ['Motores diésel', 'Turbina de vapor'], ['Fecha'] * 2, ['Emisiones en t$CO_2$ equivalente'] * 2,
                   save=False, file_name=f'Motores_diesel_emi_and_Turbina_de_vapor_emi_test_vs_pred')
    cm.plot_series(df_plot.iloc[-pred_to_plot:],
                   np.array([['Turbina_de_gas_emi_pred', 'Turbina_de_gas_emi_test_real'],
                             ['Cogeneracion_y_residuos_emi_pred', 'Cogeneracion_y_residuos_emi_test_real']],
                            dtype=object), ['Turbina de gas', 'Cogeneración y residuos'], ['Fecha'] * 2,
                   ['Emisiones en t$CO_2$ equivalente'] * 2, save=False,
                   file_name=f'Turbina_de_gas_emi_and_Cogeneracion_y_residuos_emi_test_vs_pred')

    esep = EnergySimulateEmiPredictor()
    esep.plot_mape_values(df_mape, 'Simulación de emisiones de $CO_2$',
                          ['Carbón', 'Ciclo combinado', 'Cogeneración', 'Motores diésel', 'Turbina de gas',
                           'Turbina de vapor'])
    print(df_nrmse)
    last_days = 30
    # mean NRMSE for the last 30 days for all columns
    print(f'Mean NRMSE last {last_days} days for all columns:')
    print(df_nrmse.iloc[-last_days:].mean())
    print(f'Median NRMSE last {last_days} days for all columns:')
    print(df_nrmse.iloc[-last_days:].median())
