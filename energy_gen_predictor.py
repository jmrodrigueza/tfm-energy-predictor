import numpy as np

import common.chart_manager as cm
from energy_base_predictor import EnergyBasePredictor


class EnergyGenPredictor(EnergyBasePredictor):
    """
    Energy generation predictor class
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.model_name = 'Transformer_gen_model'
        self.target_columns = ['Eolica_gen', 'Hidraulica_gen', 'Solar_fotovoltaica_gen', 'Solar_termica_gen']
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
                          'Murcia', 'Madrid', 'Eolica_gen', 'Hidraulica_gen', 'Solar_fotovoltaica_gen',
                          'Solar_termica_gen', 'Eolica_gen_1day_before', 'Hidraulica_gen_1day_before',
                          'Solar_fotovoltaica_gen_1day_before', 'Solar_termica_gen_1day_before',
                          'Eólica_gen_2days_before', 'Hidráulica_gen_2days_before',
                          'Solar_fotovoltaica_gen_2days_before', 'Solar_térmica_gen_2days_before',
                          'Eólica_gen_3days_before', 'Hidráulica_gen_3days_before',
                          'Solar_fotovoltaica_gen_3days_before', 'Solar_térmica_gen_3days_before']
        self.cols_corr_new = ['Año', 'Mes', 'Día', 'Hora', 'Solar_altitude', '5906X_Humedad_relativa',
                              '5906X_Precipitación', '5906X_Presión', '5906X_Temperatura', '5906X_Viento_pred',
                              'Asturias', 'Cantabria', 'Navarra', 'País Vasco', 'Cataluña', 'Aragón', 'Galicia',
                              'Islas Baleares', 'La Rioja', 'Valencia', 'Castilla y León', 'Castilla La Mancha',
                              'Extremadura', 'Andalucía', 'Murcia', 'Madrid', 'Eólica_gen', 'Hidráulica_gen',
                              'Solar_fotovoltaica_gen', 'Solar_térmica_gen', 'Eólica_gen_1day_before',
                              'Hidráulica_gen_1day_before', 'Solar_fotovoltaica_gen_1day_before',
                              'Solar_térmica_gen_1day_before', 'Eólica_gen_2days_before',
                              'Hidráulica_gen_2days_before', 'Solar_fotovoltaica_gen_2days_before',
                              'Solar_térmica_gen_2days_before', 'Eólica_gen_3days_before',
                              'Hidráulica_gen_3days_before', 'Solar_fotovoltaica_gen_3days_before',
                              'Solar_térmica_gen_3days_before']

    def plot_results(self, pred_to_plot=336):
        """
        Plot results
        :param pred_to_plot: The number of predictions to plot
        :return: None
        """
        cm.plot_series(self.df_comparative.iloc[-pred_to_plot:],
                       np.array([['Eolica_gen_pred', 'Eolica_gen_test_real'],
                                 ['Hidraulica_gen_pred', 'Hidraulica_gen_test_real']], dtype=object),
                       ['Eólica', 'Hidráulica'], ['Fecha'] * 2, ['Potencia generada (MW)'] * 2, save=False,
                       file_name=f'Eolica_gen_and_Hidraulica_gen_test_vs_pred')
        cm.plot_series(self.df_comparative.iloc[-pred_to_plot:],
                       np.array([['Solar_fotovoltaica_gen_pred', 'Solar_fotovoltaica_gen_test_real'],
                                 ['Solar_termica_gen_pred', 'Solar_termica_gen_test_real']], dtype=object),
                       ['Solar fotovoltaica', 'Solar térmica'], ['Fecha'] * 2, ['Potencia generada (MW)'] * 2,
                       save=False, file_name=f'Solar_fotovoltaica_gen_and_Solar_termica_gen_test_vs_pred')


if __name__ == '__main__':
    # Train, predict and evaluate
    # emm = EnergyGenPredictor()
    # emm.load_parquet_data()
    # emm.print_parquet_data()
    # emm.build_model()
    # emm.prepare_train_test_data()
    # emm.fit()
    # emm.plot_fit_history(title_detail='con el Transformer de Generación', min_loss=0)
    # emm.predict()
    # emm.evaluate()
    # emm.save_model()

    # Correlations
    # emm.generate_correlations(all_cols=False, annot=True)
    # Load, predict and evaluate
    emm = EnergyGenPredictor()
    emm.load_parquet_data()
    emm.print_parquet_data()
    emm.prepare_train_test_data()
    emm.load_model()
    emm.predict()
    emm.plot_results(pred_to_plot=720)
    df_mape = emm.evaluate()
    emm.plot_mape_values(df_mape, 'producción', ['Eólica', 'Hidráulica', 'Solar fotovoltaica', 'Solar térmica'])
