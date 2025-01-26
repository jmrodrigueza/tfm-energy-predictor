import numpy as np

import common.chart_manager as cm
from energy_base_predictor import EnergyBasePredictor


class EnergyDemandPredictor(EnergyBasePredictor):
    """
    Energy demand predictor class
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.model_name = 'Transformer_dem_model'
        self.target_columns = ['Real']
        self.predictor_columns = self.predictor_columns + [col + '_1day_before' for col in self.target_columns]
        self.predictor_columns = self.predictor_columns + [col + '_2days_before' for col in self.target_columns]
        self.predictor_columns = self.predictor_columns + [col + '_3days_before' for col in self.target_columns]
        self.epoch = 10
        self.win_size = 1
        self.bsize = 64

    def get_custom_correlation_names(self):
        """
        Get custom correlation names
        """
        self.cols_corr = ['Ano', 'Mes', 'Dia', 'Hora', 'Solar_altitude', '5906X_Humedad_relativa',
                          '5906X_Precipitacion', '5906X_Presion', '5906X_Temperatura', '5906X_Viento_pred', 'Asturias',
                          'Cantabria', 'Navarra', 'País Vasco', 'Cataluña', 'Aragón', 'Galicia', 'Islas Baleares',
                          'La Rioja', 'Valencia', 'Castilla y León', 'Castilla La Mancha', 'Extremadura', 'Andalucía',
                          'Murcia', 'Madrid'] + self.target_columns + \
                         [col + '_1day_before' for col in self.target_columns] + \
                         [col + '_2days_before' for col in self.target_columns] + \
                         [col + '_3days_before' for col in self.target_columns]
        self.cols_corr_new = ['Año', 'Mes', 'Día', 'Hora', 'Solar_altitude', '5906X_Humedad_relativa',
                              '5906X_Precipitación', '5906X_Presión', '5906X_Temperatura', '5906X_Viento_pred',
                              'Asturias', 'Cantabria', 'Navarra', 'País Vasco', 'Cataluña', 'Aragón', 'Galicia',
                              'Islas Baleares', 'La Rioja', 'Valencia', 'Castilla y León', 'Castilla La Mancha',
                              'Extremadura', 'Andalucía', 'Murcia', 'Madrid', 'Demanda real'] + \
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
                       np.array([['Real_pred', 'Real_test_real']], dtype=object), ['Demanda'], ['Fecha'],
                       ['Demanda en MW'], save=False, file_name=f'Demanda_test_vs_pred')


if __name__ == '__main__':
    # edp = EnergyDemandPredictor()
    # edp.load_parquet_data()
    # edp.print_parquet_data()
    # edp.build_model()
    # edp.prepare_train_test_data()
    # edp.fit()
    # edp.plot_fit_history(title_detail='con el Transformer de Demanda', min_loss=0.0, max_loss=1.0)
    # edp.predict()
    # edp.evaluate()
    # edp.save_model()

    # Correlations
    # edp.generate_correlations(all_cols=True, annot=False)
    # Load, predict and evaluate
    edp = EnergyDemandPredictor()
    edp.load_parquet_data()
    edp.print_parquet_data()
    edp.prepare_train_test_data()
    edp.load_model()
    edp.predict()
    edp.plot_results(pred_to_plot=720)
    df_mape = edp.evaluate()
    edp.plot_mape_values(df_mape, 'la demanda', ['Demanda'], logarithmic_scale=False)
    evaluate_nrmse = edp.evaluate_nrmse()
