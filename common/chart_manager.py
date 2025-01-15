from os.path import abspath, dirname
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import LogLocator

import common.dataframe_manager as dm

sns.set(rc={"figure.figsize": (10, 6)})
sns.set_style("whitegrid", {'grid.linestyle': ':'})

file_path = abspath(dirname(dirname(__file__)))
figures_path = file_path + '/output_figures'


def plot_series(df: pd.DataFrame, columns_to_plot: np.ndarray, title: list, xlabel: list, ylabel: list,
                file_name: str = None, save: bool = False, labels: list = None, append_datetime: bool = True,
                logarithmic_scale: bool = False, ylim_bottom=None):
    df_copy = dm.append_datetime_from_date_and_time_cols(df, set_as_index=True) if append_datetime else df.copy()
    dim = columns_to_plot.shape[0]
    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=dim, figsize=(12, 5))
    if labels is None:
        labels = ['Predicción', 'Real']

    if dim > 1:
        for i, column_set in enumerate(columns_to_plot):
            data = {}
            for column in column_set:
                data[column] = pd.Series(df_copy[column])
            sns.lineplot(data=pd.DataFrame(data), lw=2, ax=axes[i])
            axes[i].set_title(title[i])
            axes[i].set_xlabel(xlabel[i])
            axes[i].set_ylabel(ylabel[i])
            handles, _ = axes[i].get_legend_handles_labels()
            axes[i].legend(handles=handles, labels=labels)
            axes[i].grid(axis='x', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
            if ylim_bottom is not None:
                axes[i].set_ylim(bottom=ylim_bottom)
            if logarithmic_scale:
                log_locator = LogLocator(base=10.0, subs=None, numticks=10)
                axes[i].yaxis.set_major_locator(log_locator)
                axes[i].grid(True, which="both", linestyle="--", linewidth=0.5)
                axes[i].set_yscale('log')
    else:
        sns.lineplot(data=df_copy[columns_to_plot[0]], lw=2, ax=axes)
        axes.set_title(title[0])
        axes.set_xlabel(xlabel[0])
        axes.set_ylabel(ylabel[0])
        handles, _ = axes.get_legend_handles_labels()
        axes.legend(handles=handles, labels=labels)
        if ylim_bottom is not None:
            axes.set_ylim(bottom=ylim_bottom)
        if logarithmic_scale:
            log_locator = LogLocator(base=10.0, subs=None, numticks=10)
            plt.gca().yaxis.set_major_locator(log_locator)
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.yscale('log')

    plt.gcf().autofmt_xdate(rotation=45)
    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig(f'{figures_path}/{file_name}.png')
    plt.close()


def plot_boxplots_in_line(df, columns, cols_to_show, sup_title, figsize=(15, 5)):
    n_cols = len(columns)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    if n_cols == 1:
        axes = [axes]

    for ax, col, cols_to_show in zip(axes, columns, cols_to_show):
        sns.boxplot(y=df[col], ax=ax)
        ax.set_title(cols_to_show)
        ax.set_ylabel('')

    plt.suptitle(sup_title)
    plt.tight_layout()
    plt.show()


def plot_fit_history(fit_history, save, file_name, title_detail=None, min_loss=0.2, max_loss=0.7):
    """
    Plot the fit history of a model
    """
    sns.set_style("whitegrid", {'grid.linestyle': ':'})
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    sns.lineplot(data=pd.DataFrame(
        {'MSE entrenamiento': fit_history.history['loss'],
         'MSE validación': fit_history.history['val_loss']
         }), lw=2, ax=axes[0])
    axes[0].set_title(f'MSE {title_detail}')
    axes[0].set_xlabel('Nº épocas')
    axes[0].set_ylabel('Loss')
    axes[0].set_ylim(min_loss, max_loss)

    sns.lineplot(data=pd.DataFrame(
        {'MAE entrenamiento': fit_history.history['mae'],
         'MAE validación': fit_history.history['val_mae']
         }), lw=2, ax=axes[1])
    axes[1].set_title(f'MAE {title_detail}')
    axes[1].set_xlabel('Nº épocas')
    axes[1].set_ylabel('Loss')
    axes[1].set_ylim(min_loss, max_loss)
    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig(f'{figures_path}/{file_name}.png')
    plt.close()
