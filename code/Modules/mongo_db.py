# Import packages
from Modules.config import *
from pymongo import MongoClient
from pandas import read_pickle, json_normalize, to_pickle
import os


# Access tp the cloud database hosted by mongo db
client = MongoClient(mongo_key)
database = client['DAPS_assignment']
# Filename of all the files to save inside the cloud
filenames = ['Covid', f'{company}_twitter', 'News', 'Time_series']


def upload_datasets():
    """
    Uploads the dataset that were locally saved.
    """
    print('Uploading datasets...')
    for filename in filenames:
        # Load each dataframe saved in the pickle files
        dataframe_path = os.path.join(base_dir, f'{filename}.pkl')
        dataframe = read_pickle(dataframe_path)
        # Create a collection with the name of the related pickle file
        collection = database[filename]
        # Transform the dataframe in a dictionary
        records = dataframe.to_dict('records')
        # Load the dataset in the dedicated collection
        collection.insert_many(records)
        print(f'Uploaded dataset {filename}')


def download_datasets():
    """
    Downloads dataset from the cloud database.
    """
    print('Downloading datasets...')
    for filename in filenames:
        # For every required collection take all the documents
        dataset = list(database[filename].find({}))
        # Parse the entire collection from json files to a single dataframe
        dataframe = json_normalize(dataset)
        # Save the dataframe in the pickle format
        dataframe_path = os.path.join(base_dir, f'{filename}.pkl')
        to_pickle(dataframe, dataframe_path)
        print(f'Saved dataset {filename} in {dataframe_path}')
