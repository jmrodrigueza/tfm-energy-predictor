import numpy as np

import common.chart_manager as cm
from energy_base_predictor import EnergyBasePredictor


class EnergyEmissionPredictor(EnergyBasePredictor):
    """
    Energy emission predictor
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.model_name = 'Transformer_emi_model'
        self.target_columns = ['Carbon_emi', 'Ciclo_combinado_emi', 'Motores_diesel_emi', 'Turbina_de_vapor_emi',
                               'Turbina_de_gas_emi', 'Cogeneracion_y_residuos_emi']
        self.predictor_columns = self.predictor_columns + [col + '_1day_before' for col in self.target_columns]
        self.predictor_columns = self.predictor_columns + [col + '_2days_before' for col in self.target_columns]
        self.predictor_columns = self.predictor_columns + [col + '_3days_before' for col in self.target_columns]
        self.epoch = 10
        self.win_size = 60
        self.bsize = self.win_size
        self.transformers_num_heads = 10
        self.transformers_ff_dim = 256
        self.transformers_num_blocks = 1
        self.transformers_dropout = 0.2
        self.transformers_mlp_units = [512, 256, 32]

    def get_custom_correlation_names(self):
        """
        Get custom correlation names
        """
        self.cols_corr = ['Ano', 'Mes', 'Dia', 'Hora', 'Solar_altitude', '5906X_Humedad_relativa',
                          '5906X_Precipitacion', '5906X_Presion', '5906X_Temperatura', '5906X_Viento_pred', 'Asturias',
                          'Cantabria', 'Navarra', 'País Vasco', 'Cataluña', 'Aragón', 'Galicia', 'Islas Baleares',
                          'La Rioja', 'Valencia', 'Castilla y León', 'Castilla La Mancha', 'Extremadura', 'Andalucía',
                          'Murcia', 'Madrid', 'Carbon_emi', 'Ciclo_combinado_emi', 'Motores_diesel_emi',
                          'Turbina_de_vapor_emi', 'Turbina_de_gas_emi', 'Cogeneracion_y_residuos_emi'] + \
                         [col + '_1day_before' for col in self.target_columns] + \
                         [col + '_2days_before' for col in self.target_columns] + \
                         [col + '_3days_before' for col in self.target_columns]
        self.cols_corr_new = ['Año', 'Mes', 'Día', 'Hora', 'Solar_altitude', '5906X_Humedad_relativa',
                              '5906X_Precipitación', '5906X_Presión', '5906X_Temperatura', '5906X_Viento_pred',
                              'Asturias', 'Cantabria', 'Navarra', 'País Vasco', 'Cataluña', 'Aragón', 'Galicia',
                              'Islas Baleares', 'La Rioja', 'Valencia', 'Castilla y León', 'Castilla La Mancha',
                              'Extremadura', 'Andalucía', 'Murcia', 'Madrid', 'Carbon_emi', 'Ciclo_combinado_emi',
                              'Motores_diesel_emi', 'Turbina_de_vapor_emi', 'Turbina_de_gas_emi',
                              'Cogeneracion_y_residuos_emi'] + \
                             [col + '_1day_before' for col in self.target_columns] + \
                             [col + '_2days_before' for col in self.target_columns] + \
                             [col + '_3days_before' for col in self.target_columns]

    def plot_results(self, pred_to_plot=336):
        """
        Plot results
        :param pred_to_plot: The number of predictions to plot
        :return: None
        """
        cm.plot_series(self.df_comparative.iloc[-pred_to_plot:],
                       np.array([['Carbon_emi_pred', 'Carbon_emi_test_real'],
                                 ['Ciclo_combinado_emi_pred', 'Ciclo_combinado_emi_test_real']], dtype=object),
                       ['Carbón', 'Ciclo combinado'], ['Fecha'] * 2, ['Emisiones en t$CO_2$ equivalente'] * 2,
                       save=False, file_name=f'Carbon_emi_and_Ciclo_combinado_emi_test_vs_pred')
        cm.plot_series(self.df_comparative.iloc[-pred_to_plot:],
                       np.array([['Motores_diesel_emi_pred', 'Motores_diesel_emi_test_real'],
                                 ['Turbina_de_vapor_emi_pred', 'Turbina_de_vapor_emi_test_real']], dtype=object),
                       ['Motores diésel', 'Turbina de vapor'], ['Fecha'] * 2, ['Emisiones en t$CO_2$ equivalente'] * 2,
                       save=False, file_name=f'Motores_diesel_emi_and_Turbina_de_vapor_emi_test_vs_pred')
        cm.plot_series(self.df_comparative.iloc[-pred_to_plot:],
                       np.array([['Turbina_de_gas_emi_pred', 'Turbina_de_gas_emi_test_real'],
                                 ['Cogeneracion_y_residuos_emi_pred', 'Cogeneracion_y_residuos_emi_test_real']],
                                dtype=object), ['Turbina de gas', 'Cogeneración y residuos'], ['Fecha'] * 2,
                       ['Emisiones en t$CO_2$ equivalente'] * 2, save=False,
                       file_name=f'Turbina_de_gas_emi_and_Cogeneracion_y_residuos_emi_test_vs_pred')


if __name__ == '__main__':
    # eep = EnergyEmissionPredictor()
    # eep.load_parquet_data()
    # eep.print_parquet_data()
    # eep.build_model()
    # eep.prepare_train_test_data()
    # eep.fit()
    # eep.plot_fit_history(title_detail='con el Transformer de Emisiones de CO₂', min_loss=0.0, max_loss=1.0)
    # eep.predict()
    # eep.evaluate()
    # eep.save_model()

    # Correlations
    # eep.generate_correlations(all_cols=True, annot=False)
    # Load, predict and evaluate
    eep = EnergyEmissionPredictor()
    eep.load_parquet_data()
    eep.print_parquet_data()
    eep.prepare_train_test_data()
    eep.load_model()
    eep.predict()
    eep.plot_results(pred_to_plot=720)
    df_mape = eep.evaluate()
    eep.plot_mape_values(df_mape, 'las emisiones de $CO_2$',
                         ['Carbón', 'Ciclo combinado', 'Motores diésel', 'Turbina de vapor', 'Turbina de gas',
                          'Cogeneración'])

