import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score,accuracy_score
# from torch import cdist
from scipy.spatial.distance import cdist
from src.models.ssl_models.embedding_classifier import EmbeddingClassifier
from torch.utils.data import TensorDataset, DataLoader
import os
from joblib import parallel_backend
from fcmeans import FCM
from sklearn.cluster import MiniBatchKMeans



from dotenv import find_dotenv, load_dotenv
import logging
import json

from src.methods.predict_model import predict_curlie

from src.models.file_service import save_model



# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from a JSON file or environment variables
config_path = os.getenv('config_path')
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# Assume 'method' variable determines which config to use ('USL' or 'USL-t')
method = 'usl_curlie' 
config_usl=config[method.lower()]['train']

# Model and algorithm-specific parameters
learning_rate = config_usl.get('learning_rate', 0.001)
batch_size = config_usl.get('batch_size', 64)
num_epochs = config_usl.get('n_epochs', 1000)
n_clusters = config_usl.get('n_clusters', 5) 
num_classes=config_usl.get('num_classes', 2)

early_stoppage = config_usl.get('early_stoppage', True)
patience = config_usl.get('patience', 10)

n_init = config_usl.get('n_init', 10)
m_reg = config_usl.get('m_reg', 0.9)
K = config_usl.get('K', 2)
k= config_usl.get('k', 10)
T=config_usl.get('T',0.5)
lambda_ = config_usl.get('lambda', 0.5)
epsilon = config_usl.get('epsilon', 1e-5)
alpha = config_usl.get('alpha', 0.75)
std_dev = config_usl.get('std_dev', 0.1)
alpha_mixup=config_usl.get('alpha_mixup',0.75)
minibatch_kmenas=config_usl.get('minibatch_kmenas',True)

embedding_column=config['data_curlie']['embedding_column']
target_variable=config['data_curlie']['target_variable']

model_path = config['model_curlie']['output_path']
base_filename = 'model_ssl_usl.pth'
model_filepath=config['usl_curlie']['val']['model_filepath']





















# # This is hard kmeans that is not really suitable for multilabel classification
# def density_reg(embeddings):
#   # Parameters
#   # n_clusters # set number of clusters
#   # n_init # set number of initializations for stability
#   # m_reg # Momentum for EMA
#   # k  # Number of nearest neighbors for density estimation
#   # lambda_   # Balance hyperparameter for utility function
#   print("Density Reg started")
#   # K-Means clustering to partition the dataset into clusters
#   print("Kmeans started")
#   # Inside your density_reg function, replace KMeans with MiniBatchKMeans
#   # Example with a tol value of 0.001 for demonstration; adjust based on your needs
#   if minibatch_kmenas: ## Optional: for faster execution
#     print("Using MiniBatchKMeans")
#     with parallel_backend('loky', n_jobs=-1):  # Optional: for parallel execution
#       kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=n_init, tol=0.001, batch_size=100,random_state=42 ).fit(embeddings)
#   else:
#       kmeans = KMeans(n_clusters=n_clusters, n_init=n_init).fit(embeddings)
      
#   cluster_labels = kmeans.labels_
#   centroids = kmeans.cluster_centers_
#   # print(centroids)
#   # Calculate the pairwise distance matrix between embeddings and centroids
#   print("Calculating distances")
#   distances = cdist(embeddings, centroids, 'euclidean')
#   closest_clusters = np.argmin(distances, axis=1)

#   # Regularization term with EMA

#   regularization_term = np.zeros(n_clusters)  # Initialize the regularization term for each cluster

#   # Initialize the selected indices for each cluster
#   selected_indices = np.zeros(n_clusters, dtype=int)

#   # Perform the selection process iteratively
#   for iteration in range(10):
#       print(f"Iteration of density Reg {iteration + 1}")
#       new_selection = []
#       for cluster_index in range(n_clusters):
#           cluster_member_indices = np.where(closest_clusters == cluster_index)[0]
#           cluster_distances = distances[cluster_member_indices, cluster_index]

#           # Density peak selection using K-NN density estimation

#           density = 1 / (np.sort(cluster_distances)[:k].mean() + epsilon)

#           # Select the instance with the maximum density peak (minimum distance)
#           density_peak_index = cluster_member_indices[np.argmax(density)]

#           # In the first iteration, we don't have selected_indices for all clusters yet
#           if iteration > 0:
#               # Exclude the current cluster's selection from all selections
#               other_indices = np.delete(selected_indices, cluster_index)

#               # Calculate regularization term with EMA
#               if other_indices.size > 0:
#                   inter_cluster_distances = cdist(embeddings[density_peak_index].reshape(1, -1),
#                                                   embeddings[other_indices].reshape(-1, embeddings.shape[1]),
#                                                   'euclidean')
#                   current_reg_term = np.sum(1 / (inter_cluster_distances ** alpha + epsilon))
#                   regularization_term[cluster_index] = m_reg * regularization_term[cluster_index] + \
#                                                       (1 - m_reg) * current_reg_term

#           # Utility function to guide the selection within each cluster
#           lambda_ = 0.5  # Balance hyperparameter for utility function
#           utility = density - lambda_ * regularization_term[cluster_index]

#           # Selection based on utility
#           new_selection_index = cluster_member_indices[np.argmax(utility)]
#           new_selection.append(new_selection_index)

#       selected_indices = np.array(new_selection)

#   # Find indices of points closest to centroids (i.e., cluster centers)
# #   cluster_center_indices = np.argmin(distances, axis=0)

#   print("These are final selections: ",selected_indices)
#   return selected_indices,None, None




def density_reg(embeddings):
    print("Density Reg started")
    print("Fuzzy C-means started")
    
    fcm = FCM(n_clusters=n_clusters)
    fcm.fit(embeddings)
    
    distances = cdist(embeddings, fcm.centers, 'euclidean')
    closest_clusters = np.argmin(distances, axis=1)
    
    regularization_term = np.zeros(n_clusters)  # Initialize the regularization term for each cluster
    selected_indices = np.zeros(n_clusters, dtype=int)
    
    for iteration in range(10):
        print(f"Iteration of density Reg {iteration + 1}")
        new_selection = []
        for cluster_index in range(n_clusters):
            cluster_member_indices = np.where(closest_clusters == cluster_index)[0]
            if len(cluster_member_indices) == 0:
                # Skip clusters with no members
                continue
            
            cluster_distances = distances[cluster_member_indices, cluster_index]
            
            density = 1 / (np.sort(cluster_distances)[:k].mean() + epsilon)
            density_peak_index = cluster_member_indices[np.argmax(density)]
            
            if iteration > 0:
                other_indices = [i for i in selected_indices if i != cluster_index]
                
                if len(other_indices) > 0:
                    inter_cluster_distances = cdist(embeddings[density_peak_index].reshape(1, -1),
                                                    embeddings[other_indices].reshape(-1, embeddings.shape[1]),
                                                    'euclidean')
                    current_reg_term = np.sum(1 / (inter_cluster_distances ** alpha + epsilon))
                    regularization_term[cluster_index] = m_reg * regularization_term[cluster_index] + \
                                                        (1 - m_reg) * current_reg_term
            
            utility = density - lambda_ * regularization_term[cluster_index]
            new_selection_index = cluster_member_indices[np.argmax(utility)]
            new_selection.append(new_selection_index)
        
        if new_selection:
            selected_indices = np.array(new_selection)
    
    print("These are final selections: ", selected_indices)
    return selected_indices, None, None