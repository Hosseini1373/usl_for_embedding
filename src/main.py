import argparse
import sys

import numpy as np
import pandas as pd
import torch

from usl_for_embedding.src.methods import ssl_t, ssl
from usl_for_embedding.src.data.make_dataset import process_data


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
            embeddings, labels = process_data(read_raw=False, dataset='train')
            ssl.train(embeddings, labels)
        elif args.mode == 'eval':
            val_data = process_data(read_raw=False, dataset='val')
            ssl.evaluate(val_data)
        elif args.mode == 'test':
            test_data = process_data(read_raw=False, dataset='test')
            ssl.test(test_data)
        
        
    elif args.method == 'usl-t':
        print("Running in USL-t mode...")
        
        if args.mode == 'train':
            embeddings, labels = process_data(read_raw=False, dataset='train')
            ssl_t.train(embeddings, labels)
        elif args.mode == 'eval':
            val_data = process_data(read_raw=False, dataset='val')
            ssl_t.evaluate(val_data)
        elif args.mode == 'test':
            test_data = process_data(read_raw=False, dataset='test')
            ssl_t.test(test_data)
        

if __name__ == '__main__':
    main()
