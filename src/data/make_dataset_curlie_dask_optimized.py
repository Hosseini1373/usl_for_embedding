import logging
import json
from dask.distributed import Client
import dask.dataframe as dd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import numpy as np

# Configure logging and load environment
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())
client = Client('tcp://10.0.3.171:8786')  # Connect to your Dask scheduler

# Load configuration
config_path = os.getenv('config_path')
with open(config_path, 'r') as config_file:
    config = json.load(config_file)


read_raw = config['data_curlie'].get('read_raw', False)








def load_data_pa(filepath):
    """
    Load data from a Parquet file using Dask.
    """
    try:
        data = dd.read_parquet(filepath)
        logger.info(f"Data loaded from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data from {filepath}: {e}")
        raise



def save_data(data, filepath):
    """
    Save Dask DataFrame to Parquet file.
    """
    try:
        data.to_parquet(filepath, engine='pyarrow')
        logger.info(f"Data saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save data to {filepath}: {e}")
        raise

def partition_data(data, partition_column):
    """
    Partition data based on a specified column using Dask.
    """
    partitions = {part: data[data[partition_column] == part] for part in ['train', 'val', 'test']}
    for part in partitions:
        logger.info(f"Partitioned {part} data")
    return partitions







def process_data(dataset):
    """
    Process the data based on the dataset type and whether to read raw data using Dask.
    """
    if read_raw:
        logger.info("Reading raw data and processing from start...")
        data = load_data_pa(config['data_curlie']['input_filepath'])
        partitions = partition_data(data, config['data_curlie']['partitioned_indices_column'])

        for part, data_part in partitions.items():
            # Dask operations are lazy; ensure you compute or persist if necessary
            save_data(data_part, config['data_curlie'][f'{part}_filepath'])
    else:
        logger.info("Reading already processed data...")
        
    # Since direct Dask support for complex transformations to numpy arrays is limited,
    # consider keeping data in Dask dataframe as long as possible
    if dataset in ['train', 'val', 'test']:
        data = dd.read_parquet(config['data_curlie'][f'{dataset}_filepath'])  # Adjust according to your data format

        # Handling embeddings can be complex as it typically involves converting to numpy arrays
        # Consider strategies like storing pre-computed embeddings or computing them on the fly with Dask
        # For simplicity, the following line is a placeholder for the logic you'd implement
        embeddings = compute_embeddings_with_dask(data, config['data_curlie']['embedding_column'])

        labels = data[config['data_curlie']['target_variable']].compute().tolist()  # Calling compute() gathers data to a single machine; use carefully

        # Similarly, handle fine-tuned embeddings with consideration for Dask's architecture
        if 'fine_tuned_embedding_predictions' in config['data_curlie'] and dataset != 'train':
            fine_tuned_embeddings = compute_fine_tuned_embeddings_with_dask(data, config['data_curlie']['fine_tuned_embedding_predictions'])
            return embeddings, labels, fine_tuned_embeddings.compute()  # Example placeholder
        else:
            return embeddings, labels, None


def compute_embeddings_with_dask(data, embedding_column):
    """
    Placeholder function to demonstrate handling embeddings with Dask.
    Actual implementation will depend on the specific nature of your embeddings.
    """
    # This is where you'd implement your logic to handle embeddings with Dask
    # For instance, if embeddings are stored directly in the dataframe, you might simply extract them
    # Or, for more complex scenarios, you might apply custom functions with dask.array or dask.delayed
    return data[embedding_column].compute()  # Simplified for demonstration; actual approach may vary

def compute_fine_tuned_embeddings_with_dask(data, column):
    """
    Similar placeholder function for handling fine-tuned embeddings with Dask.
    """
    return data[column]  # Actual implementation may involve custom Dask operations

if __name__ == '__main__':
    # Assuming 'read_raw' is now dynamically set or accessed differently
    for dataset in ['train', 'val', 'test']:
        process_data(dataset)
