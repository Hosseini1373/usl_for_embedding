import argparse

import numpy as np
import pandas as pd


from src.methods import ssl_t, ssl
from src.data.make_dataset_curlie import process_data
from  src.data import make_dataset_curlie
from src.methods import ssl_curlie,ssl_t_curlie
from src.methods import density_reg
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


project_dir = Path(__file__).resolve().parents[1]
load_dotenv(find_dotenv())
# Load configuration
config_path = os.getenv('config_path')
with open(config_path, 'r') as config_file:
    config = json.load(config_file)
    
recalculate_indices=config['data_curlie'].get('recalculate_indices', False)
    
def main():
    parser = argparse.ArgumentParser(description='Run USL/USL-t with SSL for project')
    parser.add_argument('--method', type=str, default='USL', choices=['usl', 'usl-t'],
                        help='Choose between USL and USL-t modes.')
    parser.add_argument('--mode', type=str, default='train',choices=['train', 'eval','test'],
                        help='Choose between training, evaluation, and testing modes.')
    parser.add_argument('--dataset', type=str, default='zhaw',choices=['zhaw', 'curlie'],
                        help='Choose between zhaw or curlie dataset.')
    # parser.add_argument('--epochs', type=int, default=10,
    #                     help='Number of training epochs.')
    # Add other command-line arguments as needed

    args = parser.parse_args()

    print(f"Running in {args.mode} mode with {args.method} method on {args.dataset} dataset...")
    # Add code to run the appropriate method based on the arguments provided
    if args.dataset == 'zhaw':
        if args.method == 'usl':
            print("Running in USL mode...")
            
            if args.mode == 'train':
                embeddings, labels, _ = process_data( dataset='train')
                embeddings_val, labels_val, _ = process_data( dataset='val')
                ssl.train(embeddings, labels,embeddings_val, labels_val)
            elif args.mode == 'eval':
                embeddings_val, labels_val, fine_tuned_embedding_predictions = process_data( dataset='val')
                ssl.evaluate(embeddings_val, labels_val, fine_tuned_embedding_predictions)
            elif args.mode == 'test':
                embeddings_test, labels_test, fine_tuned_embedding_predictions = process_data( dataset='test')
                ssl.test(embeddings_test, labels_test, fine_tuned_embedding_predictions)
            
            
        elif args.method == 'usl-t':
            print("Running in USL-t mode...")
            
            if args.mode == 'train':
                embeddings, labels, _ = make_dataset_curlie.process_data(dataset='train')
                embeddings_val, labels_val, _ = make_dataset_curlie.process_data(dataset='val')
                ssl_t.train(embeddings, labels, embeddings_val, labels_val)
            elif args.mode == 'eval':
                embeddings_val, labels_val, fine_tuned_embedding_predictions = make_dataset_curlie.process_data(dataset='val')
                ssl_t.evaluate(embeddings_val, labels_val, fine_tuned_embedding_predictions)
            elif args.mode == 'test':
                embeddings_test, labels_test, fine_tuned_embedding_predictions = make_dataset_curlie.process_data(dataset='test')
                ssl.test(embeddings_test, labels_test, fine_tuned_embedding_predictions)
    
    elif args.dataset == 'curlie':
        if args.method == 'usl':
            print("Running in USL mode...")
            
            if args.mode == 'train':
                embeddings, labels, _ = make_dataset_curlie.process_data(dataset='train')
                if recalculate_indices:
                    selected_indices,_,_ = density_reg.density_reg(embeddings)
                    make_dataset_curlie.save_selected_indices(selected_indices)
                else:
                    selected_indices=make_dataset_curlie.load_selected_indices()
                embeddings_val, labels_val, _ = make_dataset_curlie.process_data(dataset='val')
                
                ssl_curlie.train(embeddings, labels,embeddings_val, labels_val,selected_indices)
            elif args.mode == 'eval':
                embeddings_val, labels_val, fine_tuned_embedding_predictions = make_dataset_curlie.process_data(dataset='val')
                ssl_curlie.evaluate(embeddings_val, labels_val, fine_tuned_embedding_predictions)
            elif args.mode == 'test':
                embeddings_test, labels_test, fine_tuned_embedding_predictions = make_dataset_curlie.process_data(dataset='test')
                ssl_curlie.test(embeddings_test, labels_test, fine_tuned_embedding_predictions)
            
        elif args.method == 'usl-t':
            print("Running in USL-t mode...")
            
            if args.mode == 'train':
                embeddings, labels, _ = make_dataset_curlie.process_data(dataset='train')                  
                embeddings_val, labels_val, _ = make_dataset_curlie.process_data(dataset='val')
                ssl_t_curlie.train(embeddings, labels, embeddings_val, labels_val,recalculate_indices)
            elif args.mode == 'eval':
                embeddings_val, labels_val, fine_tuned_embedding_predictions = make_dataset_curlie.process_data(dataset='val')
                ssl_t_curlie.evaluate(embeddings_val, labels_val, fine_tuned_embedding_predictions)
            elif args.mode == 'test':
                embeddings_test, labels_test, fine_tuned_embedding_predictions = make_dataset_curlie.process_data(dataset='test')
                ssl_curlie.test(embeddings_test, labels_test, fine_tuned_embedding_predictions)    

if __name__ == '__main__':
    main()
