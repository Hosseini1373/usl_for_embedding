import logging
import json
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pickle
import numpy as np
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(filepath):
    """
    Load data from a binary file.
    """
    try:
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        logger.info(f"Data loaded from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data from {filepath}: {e}")
        raise

def save_data(data, filepath):
    """
    Save data to a binary file.
    """
    try:
        with open(filepath, 'wb') as file:
            pickle.dump(data, file)
        logger.info(f"Data saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save data to {filepath}: {e}")
        raise

def partition_data(data, partition_column):
    """
    Partition data based on a specified column.
    """
    partitions = {'train': None, 'val': None, 'test': None}
    for part in partitions.keys():
        partitions[part] = data.loc[data[partition_column] == part]
        logger.info(f"Partitioned {part} data")
    return partitions





# This function should return the embeddings and labels for the specified dataset
def process_data(read_raw, dataset):
    # Load configuration
    config_path = os.getenv('config_path')  
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    if read_raw:
        data = load_data(config['data']['input_filepath'])
        partitions = partition_data(data, config['data']['partititioned_indices_column'])

        for part, data_part in partitions.items():
            save_data(data_part, config['data'][f'{part}_filepath'])

    if dataset in ['train', 'val', 'test']:
        data = load_data(config['data'][f'{dataset}_filepath'])
        embeddings = np.array(data[config['data']['embedding_column']].tolist())
        labels = data[config['data']['target_variable']].tolist()
        return embeddings, labels




if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())
    # dataset can be either 'train', 'val', or 'test'
    process_data(read_raw=False, dataset="train")
