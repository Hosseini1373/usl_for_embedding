import argparse

import numpy as np
import pandas as pd



from usl_embedding.methods import ssl_t, ssl



def usl_for_embedding(embeddings,method='usl',n_clusters=5,
                              learning_rate=0.001,batch_size=64,n_init=10,m_reg=0.9,k=10,lambda_=0.5,
                              epsilon=1e-5,alpha=0.75,num_epochs_cluster=100,num_heads=3):
    embeddings=np.array(embeddings).tolist()
    
    if method=='usl':
        # Assume 'method' variable determines which config to use ('USL' or 'USL-t')
        selected_indices,_,_ = ssl.density_reg(embeddings,n_init,m_reg,k,lambda_,epsilon,alpha)
    else:
        selected_indices = ssl_t.train(embeddings,learning_rate,batch_size,n_clusters,num_epochs_cluster,num_heads)
    return selected_indices

    
def main():
    # Model and algorithm-specific parameters
    example_df = pd.DataFrame({'embedding': [[0.7,0.1,0.2], [2,0.3,0.5], [1.5,2.5, 0.5]], },method='usl',n_clusters=5,
                              learning_rate=0.001,batch_size=64,n_init=10,m_reg=0.9,k=10,lambda_=0.5,
                              epsilon=1e-5,alpha=0.75,num_epochs_cluster=100,num_heads=3)
    usl_for_embedding(example_df['embedding'])

    

if __name__ == '__main__':
    main()
