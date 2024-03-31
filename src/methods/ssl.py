import datetime
import numpy as np
from sklearn.base import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import cdist
import torch
from torch import nn
import torch.nn.functional as F
from src.models.ssl_models.embedding_classifier import EmbeddingClassifier
from torch.utils.data import TensorDataset, DataLoader
import os

from dotenv import find_dotenv, load_dotenv
import logging
import json

from src.methods.predict_model import predict



# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from a JSON file or environment variables
config_path = os.getenv('config_path')
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# Assume 'method' variable determines which config to use ('USL' or 'USL-t')
method = 'usl' 
config_usl=config[method.lower()]['train']

# Model and algorithm-specific parameters
learning_rate = config_usl.get('learning_rate', 0.001)
batch_size = config_usl.get('batch_size', 64)
num_epochs = config_usl.get('n_epochs', 1000)
n_clusters = config_usl.get('n_clusters', 5) 

n_init = config_usl.get('n_init', 10)
m_reg = config_usl.get('m_reg', 0.9)
K = config_usl.get('K', 2)
T=config_usl.get('T',0.5)
lambda_ = config_usl.get('lambda', 0.5)
epsilon = config_usl.get('epsilon', 1e-5)
alpha = config_usl.get('alpha', 0.75)
std_dev = config_usl.get('std_dev', 0.1)
alpha_mixup=config_usl.get('alpha_mixup',0.75)

model_path = config['model']['output_path']


embedding_column=config['data']['embedding_column']

target_variable=config['data']['target_variable']
num_labels=config['data']['num_labels']

model_filepath=config['usl']['val']['model_filepath']



# Step 2 and Step 3: Kmeans, KNN and regularization:
def density_reg(embeddings):
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

  print("These are final selections: ",selected_indices)
  return selected_indices,cluster_center_indices,closest_clusters


def find_duplicates(input_list):
    seen = set()
    duplicates = set()
    for item in input_list:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return list(duplicates)




###### SSL Auxiliary Functions:-----------------
# Custom loss function for soft targets
def kl_divergence_loss(outputs, targets):
    # Ensure outputs are log-probabilities
    log_probabilities = F.log_softmax(outputs, dim=1)
    # Instantiate KLDivLoss
    loss_fn = nn.KLDivLoss(reduction='batchmean')
    # Compute the KL divergence loss
    return loss_fn(log_probabilities, targets)


def sharpen(p, T):
    # Sharpens the distribution by raising it element-wise to the power of 1/T and re-normalizing
    temp = p ** (1 / T)
    return temp / temp.sum(dim=1, keepdim=True)

def mixup(x, y, alpha=0.75):
    # Mixup creates a convex combination of pairs of examples and their labels
    batch_size = x.size(0)
    indices = torch.randperm(batch_size).to(x.device)

    x_mix = x * alpha + x[indices] * (1 - alpha)
    y_mix = y * alpha + y[indices] * (1 - alpha)
    return x_mix, y_mix


def augment_embeddings(embeddings, std_dev=0.1):
    # Add Gaussian noise to the embeddings as a form of augmentation
    noise = torch.randn_like(embeddings) * std_dev
    return embeddings + noise

def mixmatch(labeled_data, labels, unlabeled_data, model, num_classes):
    model.eval()
    batch_size = unlabeled_data.size(0)

    # Step 1: Augment unlabeled data K times and predict to get soft pseudo-labels
    all_outputs = []
    for _ in range(K):
        aug_unlabeled_data = augment_embeddings(unlabeled_data, std_dev)
        outputs_unlabeled = model(aug_unlabeled_data)
        all_outputs.append(outputs_unlabeled)

    # Average the predictions across augmentations
    avg_outputs_unlabeled = torch.mean(torch.stack(all_outputs), dim=0)
    pseudo_labels = sharpen(torch.softmax(avg_outputs_unlabeled, dim=1), T)

     # Ensure labels are in a compatible format for concatenation
    # Assuming 'num_classes' is defined and accessible
    labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()

    # Step 2: Mix labeled and unlabeled data by applying Mixup
    # Concatenate for Mixup
    all_inputs = torch.cat([labeled_data, unlabeled_data], dim=0)
    all_targets = torch.cat([labels_one_hot, pseudo_labels], dim=0)

    mixed_inputs, mixed_labels = mixup(all_inputs, all_targets, alpha_mixup)

    return mixed_inputs, mixed_labels


# Mix Match
def apply_mixmatch(labeled_loader, unlabeled_loader, model, device, optimizer, criterion, num_classes):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        unlabeled_iter = iter(unlabeled_loader)  # Create an iterator for the unlabeled data
        for labeled_batch in labeled_loader:
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                # If unlabeled_loader is exhausted, recreate the iterator
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            labeled_data, labels = labeled_batch
            unlabeled_data = unlabeled_batch[0]  # Only get the data, no labels to unpack

            # Move data to the device
            labeled_data = labeled_data.to(device)
            labels = labels.to(device)
            unlabeled_data = unlabeled_data.to(device)

            # Apply MixMatch
            mixed_inputs, mixed_labels = mixmatch(labeled_data, labels, unlabeled_data, model, T=0.5, alpha=0.75, K=2, std_dev=0.1)
            mixed_inputs = mixed_inputs.to(device)
            mixed_labels = mixed_labels.to(device)

            # Forward pass
            outputs = model(mixed_inputs)
            loss = kl_divergence_loss(outputs, mixed_labels)
            total_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(labeled_loader)}')








###### Training the SSL Model:-----------------


# TODO: Implement early stopping
# TODO: Compare to a baseline model (benchmark)
def train(embeddings, labels):
    
    
    print("Training the  USL SSL model...:  ")
    selected_indices,cluster_center_indices,closest_clusters = density_reg(embeddings)
    print("Selected indices:", selected_indices)
    # Set random seed for reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        device = 'cuda'
    else:
        device = 'cpu'
        
        

    input_dim = embeddings.shape[1]  # Dynamically assign input_dim
    num_classes = len(np.unique(labels))  # Dynamically determine num_classes
    
    # Preparing DataLoaders from the USL step
    # Assuming `labeled_embeddings`, `unlabeled_embeddings`, and `labels` are ready

    # Model Initialization
    model = EmbeddingClassifier(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()  # Might not be directly used if you're sticking with kl_divergence_loss

    # Convert embeddings and labels to DataLoader
    labeled_embeddings = embeddings[selected_indices]
    labeled_labels = np.array(labels)[selected_indices]  # Ensure labels_train is an array for consistent indexing


    all_indices = np.arange(len(embeddings))  # Array of all indices
    unlabeled_indices = np.setdiff1d(all_indices, selected_indices)  # Exclude selected_indices to get unlabeled ones
    unlabeled_embeddings = embeddings[unlabeled_indices]
    # Convert to tensors and create datasets
    labeled_dataset = TensorDataset(torch.tensor(labeled_embeddings, dtype=torch.float32),
                                    torch.tensor(labeled_labels, dtype=torch.long))
    unlabeled_dataset = TensorDataset(torch.tensor(unlabeled_embeddings, dtype=torch.float32))

    # DataLoaders
    labeled_loader = DataLoader(labeled_dataset, batch_size=64, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=True)

    # Apply MixMatch
    apply_mixmatch(labeled_loader, unlabeled_loader, model, device, optimizer, criterion, num_classes)
    
    # Save the trained model
    # Check if the base filename exists, and if so, create a new filename
    base_filename = 'model_ssl_usl.pth'
    model_file_path = os.path.join(model_path, base_filename)
    if os.path.isfile(model_file_path):
        # Generate a unique filename with a timestamp to avoid overwriting
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name, ext = os.path.splitext(base_filename)
        new_filename = f"{base_name}_{timestamp}{ext}"
        model_file_path = os.path.join(model_path, new_filename)
    
    
    
    torch.save(model.state_dict(), model_file_path)
    
    
    
  
###### Evaluate the SSL model on the validation dataset:-----------------   
def evaluate(val_data):
    # Load the trained model

    val_predictions_usl_ssl = predict(val_data, embedding_column, model_filepath, num_labels)
    # True labels for validation data
    val_labels = np.array(val_data[target_variable].tolist())

    
    print("Validation Results on USL model: ")
    # Calculate evaluation metrics for the USL+SSL method
    accuracy_usl_ssl = accuracy_score(val_labels, val_predictions_usl_ssl)
    precision_usl_ssl = precision_score(val_labels, val_predictions_usl_ssl, average='weighted')
    recall_usl_ssl = recall_score(val_labels, val_predictions_usl_ssl, average='weighted')
    f1_usl_ssl = f1_score(val_labels, val_predictions_usl_ssl, average='weighted')

    print(f"USL+SSL Method - Validation Accuracy: {accuracy_usl_ssl}")
    print(f"USL+SSL Method - Validation Precision: {precision_usl_ssl}")
    print(f"USL+SSL Method - Validation Recall: {recall_usl_ssl}")
    print(f"USL+SSL Method - Validation F1 Score: {f1_usl_ssl}")
    
    
    
# TODO: Implement this function to evaluate the model on the test set
def test(test_data):
    pass