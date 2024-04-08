import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score,accuracy_score
# from torch import cdist
from scipy.spatial.distance import cdist
import torch
from torch import nn
import torch.nn.functional as F
from src.models.ssl_models.embedding_classifier import EmbeddingClassifier
from torch.utils.data import TensorDataset, DataLoader
import os

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


embedding_column=config['data_curlie']['embedding_column']
target_variable=config['data_curlie']['target_variable']

model_path = config['model_curlie']['output_path']
base_filename = 'model_ssl_usl.pth'
model_filepath=config['usl_curlie']['val']['model_filepath']




















def get_device():
    # Set random seed for reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        device = 'cuda'
    else:
        device = 'cpu'
    print("Device: ", device)
    
    
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

def mixmatch(labeled_data, labels, unlabeled_data, model):
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
    # Convert labels to integers for one_hot (if not already integers)
    # labels_int = labels.long()

    # Ensure labels are in a compatible format for concatenation
    labels_one_hot = labels

    # Step 2: Mix labeled and unlabeled data by applying Mixup
    # Concatenate for Mixup
    all_inputs = torch.cat([labeled_data, unlabeled_data], dim=0)
    all_targets = torch.cat([labels_one_hot, pseudo_labels], dim=0)

    mixed_inputs, mixed_labels = mixup(all_inputs, all_targets, alpha_mixup)

    return mixed_inputs, mixed_labels




def evaluate_model_on_validation_set(model, validation_loader, device, criterion):
    print("Evaluating model on validation set...")
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, labels in validation_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            # Ensure labels are in the correct format for KL divergence
            labels_one_hot = labels
            labels_one_hot = labels_one_hot.to(device)
            # Adjust if your criterion expects log probabilities for targets
            loss = criterion(outputs, labels_one_hot)
            total_loss += loss.item()
    avg_loss = total_loss / len(validation_loader)
    return avg_loss



def apply_mixmatch_with_early_stopping(labeled_loader, unlabeled_loader, validation_loader, model, device, optimizer, num_epochs, patience):
    print("Applying MixMatch with early stopping...")
    best_loss = float('inf')
    no_improve_epoch = 0
    
    # Assuming kl_divergence_loss is being used as criterion for training, define or replace it accordingly
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        unlabeled_iter = iter(unlabeled_loader)
        for labeled_batch in labeled_loader:
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            labeled_data, labels = labeled_batch
            unlabeled_data = unlabeled_batch[0]

            labeled_data = labeled_data.to(device)
            labels = labels.to(device)
            unlabeled_data = unlabeled_data.to(device)

            mixed_inputs, mixed_labels = mixmatch(labeled_data, labels, unlabeled_data, model)
            mixed_inputs = mixed_inputs.to(device)
            mixed_labels = mixed_labels.to(device)

            outputs = model(mixed_inputs)
            loss = criterion(outputs, mixed_labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        val_loss = evaluate_model_on_validation_set(model, validation_loader, device, criterion)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {total_loss / len(labeled_loader)}, Validation Loss: {val_loss}')
        
        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve_epoch = 0
        else:
            no_improve_epoch += 1
        
        if no_improve_epoch >= patience:
            # Save model
            save_model(model,model_path, base_filename)     
            print("Early stopping triggered after epoch:", epoch+1)
            break
    # Save model at the end of training, if it is not already saved due to early stopping
    if no_improve_epoch < patience:
        save_model(model,model_path, base_filename)     




###### Training the SSL Model:-----------------


# TODO: Implement early stopping
# TODO: Compare to a baseline model (benchmark)
# For the following todos, look at the lecture 05 for more details (Infrastructure Tests, Training Tests,Functionality Tests, Evaluation Tests,Shadow, A/B, Labeling, Expectation testing)
# TODO: Infrastructure Tests (training code): single batch or single epoch tests
# TODO: Training Tests for Integration test: Pull a fixed dataset (==versioned, see lecture 06) and run a full or abbreviated training run Check to make sure model performance remains consistent Consider pulling a sliding window of data and test that Run periodically (nightly for frequently changing codebases)
# TODO: Funcionality Tests: Unit test your prediction code like you would any other code and Load a pretrained model and test prediction
# TODO: Evaluation Tests: Test the evaluation code with a fixed dataset and model and Check that the evaluation metrics are consistent with the expected values and Run every time a new candidate model is created and considered for production
# TODO: Collect high-loss examples for further analysis, Problem could be in the model or it could be in the data!
# TODO: Look at all of the metrics you care about: • Model metrics (precision, recall, accuracy, L2, etc) • Behavioral metrics • Robustness metrics • Privacy and fairness metrics. Do Slice-based evaluation, for website traffic data, slice among gender, mobile vs. desktop
# TODO: Shadow Testing: Run the new model in parallel with the old model and compare the results,  Detect inconsistencies between the offline and online model, Detect issues that appear only on production data or in production environment 
# TODO: A/B Testing: Run the new model in parallel with the old model and compare the results, Test users’ reaction to the new model, UUnderstand how your new model affects user and business metrics
# TODO: • Catch poor quality labels from Feeback of model in production before they corrupt your model. Train and certify the labelers. Trust score.
# TODO: Expectation Testing (unit test for data): Catch data quality issues before they make your way into your pipeline: • Define rules about properties of each of your data tables at each stage in your data cleaning and preprocessing pipeline, Run them when you run batch data pipeline jobs

def train(embeddings, labels, embeddings_val, labels_val):
    
    
    print("Training the  USL SSL model...:  ")
    selected_indices,cluster_center_indices,closest_clusters = density_reg(embeddings)
    print("Selected indices:", selected_indices)

    device=get_device()
        

    input_dim = embeddings.shape[1]  # Dynamically assign input_dim
    # num_classes = len(np.unique(labels))  # Dynamically determine num_classes
    
    # Preparing DataLoaders from the USL step
    # Assuming `labeled_embeddings`, `unlabeled_embeddings`, and `labels` are ready

    # Model Initialization
    model = EmbeddingClassifier(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # criterion = nn.CrossEntropyLoss()  # Might not be directly used if you're sticking with kl_divergence_loss

    # Convert embeddings and labels to DataLoader
    labeled_embeddings = embeddings[selected_indices]
    labeled_labels = np.array(labels)[selected_indices]  # Ensure labels_train is an array for consistent indexing
    

    all_indices = np.arange(len(embeddings))  # Array of all indices
    unlabeled_indices = np.setdiff1d(all_indices, selected_indices)  # Exclude selected_indices to get unlabeled ones
    unlabeled_embeddings = embeddings[unlabeled_indices]
    # Convert to tensors and create datasets
    labeled_dataset = TensorDataset(torch.tensor(labeled_embeddings, dtype=torch.float32),
                                    torch.tensor(labeled_labels, dtype=torch.float32))
    # Validation dataset
    val_dataset = TensorDataset(torch.tensor(embeddings_val, dtype=torch.float32), 
                                torch.tensor(labels_val, dtype=torch.float32))
    
    unlabeled_dataset = TensorDataset(torch.tensor(unlabeled_embeddings, dtype=torch.float32))

    # DataLoaders
    labeled_loader = DataLoader(labeled_dataset, batch_size=64, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    
    if early_stoppage:
        # Apply MixMatch
        apply_mixmatch_with_early_stopping(labeled_loader, unlabeled_loader, validation_loader, model, device, optimizer, num_epochs, patience)
    
    

    
    
    

    
    
  
def evaluate(embeddings_val, labels_val, fine_tuned_embedding_predictions):
    print("Evaluating the USL SSL model...:  ")
    device=get_device()
    # Load the true labels
    val_labels = np.array(labels_val)
    
    # Predictions from the SSL model
    val_predictions_usl_ssl = predict_curlie(embeddings_val, model_filepath, num_classes, device)
    print("Predictions from the SSL model: ",val_predictions_usl_ssl)
    
    # Evaluate SSL-enhanced model
    hamming_loss_ssl = hamming_loss(val_labels, val_predictions_usl_ssl)
    precision_ssl = precision_score(val_labels, val_predictions_usl_ssl, average='micro')
    recall_ssl = recall_score(val_labels, val_predictions_usl_ssl, average='micro')
    f1_ssl = f1_score(val_labels, val_predictions_usl_ssl, average='micro')
    
    
    
    probabilities_fine_tuned = torch.sigmoid(torch.tensor(fine_tuned_embedding_predictions))
    # Apply threshold to get binary predictions for each class
    predictions_fine_tuned = (probabilities_fine_tuned > 0.5).int().cpu().numpy()
    # Evaluate baseline model
    hamming_loss_baseline = hamming_loss(val_labels, predictions_fine_tuned)
    precision_baseline = precision_score(val_labels, predictions_fine_tuned, average='micro')
    recall_baseline = recall_score(val_labels, predictions_fine_tuned, average='micro')
    f1_baseline = f1_score(val_labels, predictions_fine_tuned, average='micro')
    
    # Print results
    print("Validation Results:")
    print(f"SSL Model - Hamming Loss: {hamming_loss_ssl}, Precision: {precision_ssl}, Recall: {recall_ssl}, F1 Score: {f1_ssl}")
    print(f"Baseline Model - Hamming Loss: {hamming_loss_baseline}, Precision: {precision_baseline}, Recall: {recall_baseline}, F1 Score: {f1_baseline}")
    
    # Calculating percentages of baseline reached
    percentage_of_baseline = {
        "Hamming Loss": (hamming_loss_ssl / hamming_loss_baseline) * 100,
        "Precision": (precision_ssl / precision_baseline) * 100,
        "Recall": (recall_ssl / recall_baseline) * 100,
        "F1 Score": (f1_ssl / f1_baseline) * 100
    }

    # Preparing data for DataFrame
    data = {
        "Metric": ["Hamming Loss", "Precision", "Recall", "F1 Score"],
        "Baseline": [hamming_loss_baseline, precision_baseline, recall_baseline, f1_baseline],
        "SSL Model": [hamming_loss_ssl, precision_ssl, recall_ssl, f1_ssl],
        "Percentage of Baseline": [
            percentage_of_baseline["Hamming Loss"], 
            percentage_of_baseline["Precision"], 
            percentage_of_baseline["Recall"], 
            percentage_of_baseline["F1 Score"]
        ]
    }

    # Creating DataFrame
    df = pd.DataFrame(data)

    # Formatting for nicer display
    df_formatted = df.copy()
    df_formatted['Baseline'] = df_formatted['Baseline'].map('{:,.2f}'.format)
    df_formatted['SSL Model'] = df_formatted['SSL Model'].map('{:,.2f}'.format)
    df_formatted['Percentage of Baseline'] = df_formatted['Percentage of Baseline'].map('{:,.2f}%'.format)

    # Display DataFrame
    print(df_formatted)
            
    
# TODO: Implement this function to evaluate the model on the test set
def test(test_data):
    pass