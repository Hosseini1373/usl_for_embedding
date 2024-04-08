import argparse
import sys

import numpy as np
import pandas as pd
import torch

from src.methods import ssl_t, ssl
from src.data.make_dataset import process_data


def main():
    parser = argparse.ArgumentParser(description='Run USL/USL-t with SSL for project')
    parser.add_argument('--method', type=str, default='USL', choices=['usl', 'usl-t'],
                        help='Choose between USL and USL-t modes.')
    parser.add_argument('--mode', type=str, default='train',choices=['train', 'eval','test'],
                        help='Choose between training, evaluation, and testing modes.')
    # parser.add_argument('--epochs', type=int, default=10,
    #                     help='Number of training epochs.')
    # Add other command-line arguments as needed

    args = parser.parse_args()

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
            embeddings, labels, _ = process_data(dataset='train')
            embeddings_val, labels_val, _ = process_data(dataset='val')
            ssl_t.train(embeddings, labels, embeddings_val, labels_val)
        elif args.mode == 'eval':
            embeddings_val, labels_val, fine_tuned_embedding_predictions = process_data(dataset='val')
            ssl_t.evaluate(embeddings_val, labels_val, fine_tuned_embedding_predictions)
        elif args.mode == 'test':
            embeddings_test, labels_test, fine_tuned_embedding_predictions = process_data(dataset='test')
            ssl.test(embeddings_test, labels_test, fine_tuned_embedding_predictions)
        

if __name__ == '__main__':
    main()
