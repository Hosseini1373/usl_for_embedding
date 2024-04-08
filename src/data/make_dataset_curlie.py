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

# Load configuration
config_path = os.getenv('config_path')
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

read_raw = config['data_curlie'].get('read_raw', False)







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






def process_data(dataset):
    """
    Process the data based on the dataset type and whether to read raw data.
    """
    if read_raw:
        print("Reading raw data and processing from start...")
        data = load_data(config['data_curlie']['input_filepath'])
        partitions = partition_data(data, config['data_curlie']['partitioned_indices_column'])

        for part, data_part in partitions.items():
            save_data(data_part, config['data_curlie'][f'{part}_filepath'])
    else:
        print("Reading already processed data...")
        
    if dataset in ['train', 'val', 'test']:
        data = load_data(config['data_curlie'][f'{dataset}_filepath'])
        embeddings = np.array(data[config['data_curlie']['embedding_column']].tolist())
        
        labels = data[config['data_curlie']['target_variable']].tolist()
        if 'fine_tuned_embedding_predictions' in config['data_curlie'] and dataset != 'train':
            fine_tuned_embeddings = np.array(data[config['data_curlie']['fine_tuned_embedding_predictions']].tolist())
            return embeddings, labels, fine_tuned_embeddings
        else:
            return embeddings, labels, None


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())
    # Adjust according to whether 'read_raw' should be a parameter or directly accessed from config
    # dataset can be either 'train', 'val', or 'test'
    for dataset in ['train', 'val', 'test']:
        process_data(dataset="train")
