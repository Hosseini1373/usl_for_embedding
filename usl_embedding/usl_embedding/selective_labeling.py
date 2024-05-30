import numpy as np
import pandas as pd



from usl_embedding.methods import usl, usl_t



def usl_for_embedding(embeddings, method='usl', n_clusters=None, learning_rate=0.001, batch_size=64, n_init=10, m_reg=0.9, k=10, lambda_=0.5, epsilon=1e-5, alpha=0.75, num_epochs_cluster=100, num_heads=3):
    """
    This function applies the USL or USL-t method to given embeddings.

    Args:
        embeddings (array-like): Input data, embeddings to be clustered.
        method (str): The clustering method to use ('usl' or 'usl-t').
        n_clusters (int): Number of clusters to form.
        learning_rate (float): Learning rate for the optimization.
        batch_size (int): Number of samples per gradient update.
        n_init (int): Number of time the algorithm will be run with different centroid seeds.
        m_reg (float): Regularization term.
        k (int): Number of nearest neighbors.
        lambda_ (float): Trade-off parameter in the loss function.
        epsilon (float): Tolerance for stopping criteria.
        alpha (float): Decay rate.
        num_epochs_cluster (int): Number of epochs for clustering.
        num_heads (int): Number of attention heads in transformer model.
    
    Returns:
        ndarray: Array of indices of selected samples.
    """
    embeddings = np.array(embeddings)
    
    if method == 'usl':
        selected_indices, _, _ = usl.density_reg(embeddings, n_clusters, n_init, m_reg, k, lambda_, epsilon, alpha)
    else:
        selected_indices = usl_t.train(embeddings, learning_rate, batch_size, n_clusters, num_epochs_cluster, num_heads)
    return selected_indices


    
def main():
    # Generate 10 embeddings, each with 10 random floating-point numbers
    data = {'embedding': [list(np.random.rand(10)) for _ in range(10)]}
    
    # method is 'usl' or 'usl-t'
    # n_clusters is the number of samples that should be selected for labeling
    selected_indices = usl_for_embedding(data['embedding'],method='usl',n_clusters=2,
                                learning_rate=0.001,batch_size=64,n_init=10,m_reg=0.9,k=10,lambda_=0.5,
                                epsilon=1e-5,alpha=0.75,num_epochs_cluster=100,num_heads=3)

    print(selected_indices)

if __name__ == '__main__':
    main()
