import numpy as np
import pandas as pd

from energy_demand_predictor import EnergyDemandPredictor


class ReePredictorComparator(EnergyDemandPredictor):
    """
    Ree predictor comparator
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()

    def evaluate(self):
        """
        Evaluate
        :return: The dataframe with the MAPE
        """
        df_mape_prediction = super().evaluate()
        # calculate mape for Real and Prevista columns in self.df
        # Add a small value to avoid division by zero
        df_mape_ree = self.df_test[['Ano', 'Mes', 'Dia', 'Hora', 'Minuto']].iloc[-self.y_pred.shape[0]:] \
            .reset_index(drop=True)
        df_mape_ree['datetime'] = pd.to_datetime(
            {'year': df_mape_ree['Ano'], 'month': df_mape_ree['Mes'],
             'day': df_mape_ree['Dia']})
        df_mape_ree = df_mape_ree[['datetime']]

        df_mape_ree['Real'] = np.abs(self.df[['Real']].iloc[-df_mape_ree.shape[0]:].values) + 0.99
        df_mape_ree['Prevista'] = np.abs(self.df[['Prevista']].iloc[-df_mape_ree.shape[0]:].values) + 0.99
        # calculates the MAPE grouping by date
        df_mape_ree_daily = df_mape_ree.groupby(['datetime']).apply(
            lambda x: (abs((x['Real'] - x['Prevista']) / x['Real'])).mean() * 100)
        df_mape_ree_daily = df_mape_ree_daily.reset_index()
        df_mape_ree_daily.name = f'REE_daily_mape'
        # append datetime if not exists
        if 'datetime' not in df_mape_prediction.columns:
            df_mape_prediction['datetime'] = df_mape_ree_daily['datetime']
        df_mape_prediction[f'REE_daily_mape'] = df_mape_ree_daily[0]

        return df_mape_prediction


if __name__ == '__main__':
    reepc = ReePredictorComparator()
    reepc.load_parquet_data()
    reepc.prepare_train_test_data()
    reepc.load_model()
    reepc.predict()
    df_mape = reepc.evaluate()
    reepc.plot_mape_values(df_mape, 'MAPE para el modelo de demanda implementado y el de REE',
                           ['Modelo de demanda', 'Modelo de REE'])
