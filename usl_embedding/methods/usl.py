import numpy as np
from sklearn.cluster import KMeans
# from torch import cdist
from scipy.spatial.distance import cdist
import torch

def get_device():
    # Set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        device = 'cuda'
    else:
        device = 'cpu'
    print("Device: ", device)
    return device
    
def density_reg(embeddings,n_clusters,n_init,m_reg,k,lambda_,epsilon,alpha):
  # Parameters
  # n_clusters # set number of clusters
  # n_init # set number of initializations for stability
  # m_reg # Momentum for EMA
  # k  # Number of nearest neighbors for density estimation
  # lambda_   # Balance hyperparameter for utility function

  # K-Means clustering to partition the dataset into clusters
  kmeans = KMeans(n_clusters=n_clusters, n_init=n_init).fit(embeddings)
  cluster_labels = kmeans.labels_
  centroids = kmeans.cluster_centers_
  # print(centroids)

  # Calculate the pairwise distance matrix between embeddings and centroids
  distances = cdist(embeddings, centroids, 'euclidean')
  closest_clusters = np.argmin(distances, axis=1)

  # Regularization term with EMA

  regularization_term = np.zeros(n_clusters)  # Initialize the regularization term for each cluster

  # Initialize the selected indices for each cluster
  selected_indices = np.zeros(n_clusters, dtype=int)

  # Perform the selection process
  for iteration in range(10):
      new_selection = []
      for cluster_index in range(n_clusters):
          cluster_member_indices = np.where(closest_clusters == cluster_index)[0]
          cluster_distances = distances[cluster_member_indices, cluster_index]

          # Density peak selection using K-NN density estimation

          density = 1 / (np.sort(cluster_distances)[:k].mean() + epsilon)

          # Select the instance with the maximum density peak (minimum distance)
          density_peak_index = cluster_member_indices[np.argmax(density)]

          # In the first iteration, we don't have selected_indices for all clusters yet
          if iteration > 0:
              # Exclude the current cluster's selection from all selections
              other_indices = np.delete(selected_indices, cluster_index)

              # Calculate regularization term with EMA
              if other_indices.size > 0:
                  inter_cluster_distances = cdist(embeddings[density_peak_index].reshape(1, -1),
                                                  embeddings[other_indices].reshape(-1, embeddings.shape[1]),
                                                  'euclidean')
                  current_reg_term = np.sum(1 / (inter_cluster_distances ** alpha + epsilon))
                  regularization_term[cluster_index] = m_reg * regularization_term[cluster_index] + \
                                                      (1 - m_reg) * current_reg_term

          # Utility function to guide the selection within each cluster
          lambda_ = 0.5  # Balance hyperparameter for utility function
          utility = density - lambda_ * regularization_term[cluster_index]

          # Selection based on utility
          new_selection_index = cluster_member_indices[np.argmax(utility)]
          new_selection.append(new_selection_index)

      selected_indices = np.array(new_selection)

  # Find indices of points closest to centroids (i.e., cluster centers)
  cluster_center_indices = np.argmin(distances, axis=0)

  return selected_indices,cluster_center_indices,closest_clusters
