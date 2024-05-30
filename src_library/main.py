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
    

method=config.get('method', 'usl')



def usl_for_embedding(embeddings, method='usl'):
    embeddings=np.array(embeddings).tolist()
    
    if method=='usl':
        selected_indices,_,_ = ssl.density_reg(embeddings)
    else:
        selected_indices = ssl_t.train(embeddings)
    return selected_indices

    
def main():
    example_df = pd.DataFrame({'embedding': [[0.7,0.1,0.2], [2,0.3,0.5], [1.5,2.5, 0.5]], })
    usl_for_embedding(example_df['embedding'])

    

if __name__ == '__main__':
    main()
