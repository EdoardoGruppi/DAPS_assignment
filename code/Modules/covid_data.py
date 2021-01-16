# Import packages
from covid19dh import covid19
from pandas import to_pickle, read_pickle
import os
from Modules.config import *
from datetime import datetime
import numpy as np


def get_covid_data(filename='Covid'):
    """
    Gets the daily records about the ongoing pandemic due by the proliferation of the coronavirus SARS-CoV-2.

    :param filename: name of the pickle file that will host the data.
    :return: address of the pickle file.
    """
    # Get the data
    dataframe, src = covid19(raw=True)
    # Create the path where to save the data
    dataframe_path = os.path.join(base_dir, f'{filename}.pkl')
    # Save the dataset in a pickle file
    to_pickle(dataframe, dataframe_path)
    return dataframe_path


def covid_preprocessing(df_path, daily_change=True):
    """
    Pre-processes the pandemic dataset dropping and combining columns.

    :param df_path: path of the file where the dataset is saved.
    :param daily_change: if True it computes the daily change of the new global cases. default_value=True
    :return: the pre-processed dataset.
    """
    # Read the original data
    dataframe = read_pickle(df_path)
    # Drop the columns that are not important for the aims of this specific analysis
    dataframe = dataframe[['date', 'confirmed', 'recovered', 'deaths']]
    # Transform the date in datetime objects to delete discrepancies with other datasets
    dataframe.index = dataframe['date'].apply(lambda x: datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S"))
    dataframe = dataframe.drop(['date'], axis=1)
    # Sum all the daily confirmed, recovered, deaths cases recorded by each state
    dataframe = dataframe.groupby(level=0).sum()
    # Consider only the active cases
    dataframe['active'] = dataframe.confirmed - dataframe.recovered - dataframe.deaths
    dataframe = dataframe.drop(['confirmed', 'deaths', 'recovered'], axis=1)
    if daily_change:
        # Compute the daily movement percentage of the active cases after replacing missing values with 0 where detected
        dataframe['Covid'] = dataframe.active.pct_change(periods=1).fillna(0)
        # Since in some situations the cases pass from 0 to an integer the daily change is considered inf. This could
        # create problems subsequently. Hence, those values are replaced with a symbolic daily change of 100%.
        dataframe.Covid = dataframe.Covid.replace(np.inf, 1)
        # The new dataframe is composed only by daily_changes
        dataframe = dataframe.Covid
    return dataframe
