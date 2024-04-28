import argparse

import numpy as np
import pandas as pd



from  src.data import make_dataset
from  src.data import make_dataset_curlie
from  src.data import make_dataset_segments

from src.methods import ssl_t, ssl
from src.methods import ssl_curlie,ssl_t_curlie
from src.methods import ssl_segments,ssl_t_segments

from src.visualization import visualize

from src.methods import density_reg_curlie
from src.methods import density_reg_segments 

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
    
recalculate_indices=config['data'].get('recalculate_indices', False)
recalculate_indices_curlie=config['data_curlie'].get('recalculate_indices', False)
recalculate_indices_segments=config['data_segments'].get('recalculate_indices', False)

just_ssl=config['data_segments'].get('just_ssl', False)

plot_filepath = config['data'].get('plot_filepath', '')
plot_filepath_curlie = config['data_curlie'].get('plot_filepath', '')
plot_filepath_segments = config['data_segments'].get('plot_filepath', '')






    
def main():
    parser = argparse.ArgumentParser(description='Run USL/USL-t with SSL for project')
    parser.add_argument('--method', type=str, default='USL', choices=['usl', 'usl-t'],
                        help='Choose between USL and USL-t modes.')
    parser.add_argument('--mode', type=str, default='train',choices=['train', 'eval','test'],
                        help='Choose between training, evaluation, and testing modes.')
    parser.add_argument('--dataset', type=str, default='zhaw',choices=['zhaw', 'curlie','zhaw_segments'],
                        help='Choose between zhaw, curlie dataset or zhaw_segments.')
    # parser.add_argument('--epochs', type=int, default=10,
    #                     help='Number of training epochs.')
    # Add other command-line arguments as needed

    args = parser.parse_args()

    print(f"Running in {args.mode} mode with {args.method} method on {args.dataset} dataset...")


    ########## First ZHAW dataset ##########
    if args.dataset == 'zhaw':
        if args.method == 'usl':
            print("Running in USL mode...")
            if args.mode == 'train':
                embeddings, labels, _ = make_dataset.process_data( dataset='train')
                if recalculate_indices:
                    selected_indices,_,_ = ssl.density_reg(embeddings)
                    visualize.visualize_clusters(embeddings,selected_indices,plot_filepath,'selected_points.png')
                    make_dataset.save_selected_indices(selected_indices)
                else:
                    selected_indices=make_dataset.load_selected_indices()
                    visualize.visualize_clusters(embeddings,selected_indices,plot_filepath,'selected_points.png')
                    print("Loaded selected indices: ",selected_indices)
                embeddings_val, labels_val, _ = make_dataset.process_data(dataset='val')
                ssl.train(embeddings, labels,embeddings_val, labels_val,selected_indices)
            elif args.mode == 'eval':
                embeddings_val, labels_val, fine_tuned_embedding_predictions = make_dataset.process_data( dataset='val')
                ssl.evaluate(embeddings_val, labels_val, fine_tuned_embedding_predictions)
            elif args.mode == 'test':
                embeddings_test, labels_test, fine_tuned_embedding_predictions = make_dataset.process_data( dataset='test')
                ssl.test(embeddings_test, labels_test, fine_tuned_embedding_predictions)
            
            
        elif args.method == 'usl-t':
            print("Running in USL-t mode...")
            
            if args.mode == 'train':
                embeddings, labels, _ = make_dataset.process_data(dataset='train')
                embeddings_val, labels_val, _ = make_dataset.process_data(dataset='val')
                ssl_t.train(embeddings, labels, embeddings_val, labels_val,recalculate_indices,plot_filepath,'selected_points_t.png')
            elif args.mode == 'eval':
                embeddings_val, labels_val, fine_tuned_embedding_predictions = make_dataset.process_data(dataset='val')
                ssl_t.evaluate(embeddings_val, labels_val, fine_tuned_embedding_predictions)
            elif args.mode == 'test':
                embeddings_test, labels_test, fine_tuned_embedding_predictions = make_dataset.process_data(dataset='test')
                ssl.test(embeddings_test, labels_test, fine_tuned_embedding_predictions)
    
    
    
    
    
    
    
    
    
    
    ########## Curlie dataset ##########
    elif args.dataset == 'curlie':
        if args.method == 'usl':
            print("Running in USL mode...")
            
            if args.mode == 'train':
                embeddings, labels, _ = make_dataset_curlie.process_data(dataset='train')
                if recalculate_indices_curlie:
                    selected_indices,_,_ = density_reg_curlie.density_reg(embeddings)
                    visualize.visualize_clusters(embeddings,selected_indices,plot_filepath_curlie,'selected_points.png')
                    make_dataset_curlie.save_selected_indices(selected_indices)
                else:
                    selected_indices=make_dataset_curlie.load_selected_indices()
                    visualize.visualize_clusters(embeddings,selected_indices,plot_filepath_curlie,'selected_points.png')
                    print("Loaded selected indices: ",selected_indices)
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
                ssl_t_curlie.train(embeddings, labels, embeddings_val, labels_val,recalculate_indices_curlie,plot_filepath_curlie,'selected_points_t.png')
            elif args.mode == 'eval':
                embeddings_val, labels_val, fine_tuned_embedding_predictions = make_dataset_curlie.process_data(dataset='val')
                ssl_t_curlie.evaluate(embeddings_val, labels_val, fine_tuned_embedding_predictions)
            elif args.mode == 'test':
                embeddings_test, labels_test, fine_tuned_embedding_predictions = make_dataset_curlie.process_data(dataset='test')
                ssl_curlie.test(embeddings_test, labels_test, fine_tuned_embedding_predictions)    










    ########## Second ZHAW dataset ##########
    elif args.dataset == 'zhaw_segments':
        if args.method == 'usl':
            print("Running in USL mode...")
            
            if args.mode == 'train':
                embeddings, labels, _ = make_dataset_segments.process_data(dataset='train')
                if just_ssl:
                    selected_indices=range(len(embeddings)-1) # -1 because we want to have at least one unlabelled point
                    visualize.visualize_clusters(embeddings,selected_indices,plot_filepath_segments,'selected_points.png')
                    print("Loaded selected indices: ",selected_indices)
                    
                elif recalculate_indices_segments:
                    print("Recalculating indices...")
                    selected_indices,_,_ = density_reg_segments.density_reg(embeddings)
                    visualize.visualize_clusters(embeddings,selected_indices,plot_filepath_segments,'selected_points.png')
                    make_dataset_segments.save_selected_indices(selected_indices)
                    
                else:
                    selected_indices=make_dataset_segments.load_selected_indices()
                    visualize.visualize_clusters(embeddings,selected_indices,plot_filepath_segments,'selected_points.png')
                    print("Loaded selected indices: ",selected_indices)
                embeddings_val, labels_val, _ = make_dataset_segments.process_data(dataset='val')
                
                ssl_segments.train(embeddings, labels,embeddings_val, labels_val,selected_indices)
            elif args.mode == 'eval':
                embeddings_val, labels_val, fine_tuned_embedding_predictions = make_dataset_segments.process_data(dataset='val')
                ssl_segments.evaluate(embeddings_val, labels_val)
            elif args.mode == 'test':
                embeddings_test, labels_test, fine_tuned_embedding_predictions = make_dataset_segments.process_data(dataset='test')
                ssl_segments.test(embeddings_test, labels_test)
            
        elif args.method == 'usl-t':
            print("Running in USL-t mode...")
            
            if args.mode == 'train':
                embeddings, labels, _ = make_dataset_segments.process_data(dataset='train')                  
                embeddings_val, labels_val, _ = make_dataset_segments.process_data(dataset='val')
                ssl_t_segments.train(embeddings, labels, embeddings_val, labels_val,recalculate_indices_segments,plot_filepath_segments,'selected_points_t.png')
            elif args.mode == 'eval':
                embeddings_val, labels_val, fine_tuned_embedding_predictions = make_dataset_segments.process_data(dataset='val')
                ssl_t_segments.evaluate(embeddings_val, labels_val)
            elif args.mode == 'test':
                embeddings_test, labels_test, fine_tuned_embedding_predictions = make_dataset_segments.process_data(dataset='test')
                ssl_segments.test(embeddings_test, labels_test)    



if __name__ == '__main__':
    main()
