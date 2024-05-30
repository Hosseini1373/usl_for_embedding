import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score,accuracy_score
# from torch import cdist
# from scipy.spatial.distance import cdist
import torch
from torch import nn
import torch.nn.functional as F
from src.models.ssl_models_segments.embedding_classifier import EmbeddingClassifier
from torch.utils.data import TensorDataset, DataLoader
import os

from dotenv import find_dotenv, load_dotenv
import logging
import json
from torch.optim import Adam

from src.methods.predict_model import predict_segments
from src.models.ssl_t_models_segments.clustering_model import ClusteringModel
from src.models.file_service import save_model,load_model
import logging
from  src.data import make_dataset_segments
from src.visualization import visualize

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from a JSON file or environment variables
config_path = os.getenv('config_path')
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# Assume 'method' variable determines which config to use ('USL' or 'USL-t')
method = 'usl-t_segments' 
config_usl=config[method.lower()]['train']

# Model and algorithm-specific parameters
learning_rate = config_usl.get('learning_rate', 0.001)
batch_size = config_usl.get('batch_size', 64)
num_epochs = config_usl.get('n_epochs', 1000)
n_clusters = config_usl.get('n_clusters', 100) 
learning_rate_usl_t = config_usl.get('learning_rate_usl-t', 0.001)
batch_size_usl_t = config_usl.get('batch_size_usl-t',0.001)
num_epochs_cluster = config_usl.get('num_epochs_cluster', 100)

n_init = config_usl.get('n_init', 10)
m_reg = config_usl.get('m_reg', 0.9)
K = config_usl.get('K', 2)
T=config_usl.get('T',0.5)
lambda_ = config_usl.get('lambda', 0.5)
epsilon = config_usl.get('epsilon', 1e-5)
alpha = config_usl.get('alpha', 0.5)
std_dev = config_usl.get('std_dev', 0.1)
num_heads = config_usl.get('num_heads', 3)
alpha_mixup=config_usl.get('alpha_mixup',0.75)
# The emebdding dimension of the cluster model, adjust as needed
cluster_embedding_dim=config_usl.get('cluster_embedding_dim',767)
early_stoppage=config_usl.get('early_stoppage',True)
num_classes=config_usl.get('num_classes',14)
patience = config_usl.get('patience', 10)
patience_cluster = config_usl.get('patience_cluster', 10)

model_path = config['model_segments']['output_path_t']


embedding_column=config['data_segments']['embedding_column']
target_variable=config['data_segments']['target_variable']

base_filename = 'model_ssl_usl-t.pth'
base_filename_cluster = 'clustering_model_usl_t.pth'
model_filepath=config['usl-t_segments']['val']['model_filepath']
















def get_device():
    # Set random seed for reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        device = 'cuda'
    else:
        device = 'cpu'
    print("Device: ", device)
    return device
    
    

def find_duplicates(input_list):
    seen = set()
    duplicates = set()
    for item in input_list:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return list(duplicates)




###### Loss functions for USL-t:-----------------



def kl_divergence_loss(outputs, targets):
    # Ensure outputs are log-probabilities
    log_probabilities = F.log_softmax(outputs, dim=1)
    # Instantiate KLDivLoss
    loss_fn = nn.KLDivLoss(reduction='batchmean')
    # Compute the KL divergence loss
    return loss_fn(log_probabilities, targets)



# Credit to PAWS: https://github.com/facebookresearch/suncet/blob/main/src/losses.py
def sharpen(p, T):  # T: sharpen temperature
    sharp_p = p ** (1. / T)
    sharp_p = sharp_p / torch.sum(sharp_p, dim=1, keepdim=True)
    return sharp_p




class OursLossLocal(nn.Module):
    def __init__(self, num_classes, num_heads, momentum=None, adjustment_weight=None, sharpen_temperature=None):
        super(OursLossLocal, self).__init__()
        self.momentum = momentum

        self.adjustment_weight = adjustment_weight

        self.num_heads = num_heads

        self.register_buffer("prob_ema", torch.ones(
            (num_heads, num_classes)) / num_classes)

        self.sharpen_temperature = sharpen_temperature

    def forward(self, head_id, anchors, neighbors):
        # This is ours v2 with multi_headed prob_ema support
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        head_prob_ema = self.prob_ema[head_id]
        neighbors_adjusted = neighbors - self.adjustment_weight * \
            torch.log(head_prob_ema).view((1, -1))

        anchors_prob = F.softmax(anchors, dim=1)
        positives_prob = F.softmax(neighbors_adjusted, dim=1)
        log_anchors_prob = F.log_softmax(anchors, dim=1)

        positives_original_prob = F.softmax(neighbors, dim=1)
        head_prob_ema = head_prob_ema * self.momentum + \
            positives_original_prob.detach().mean(dim=0) * (1 - self.momentum)
        head_prob_ema = head_prob_ema / head_prob_ema.sum()

        self.prob_ema[head_id] = head_prob_ema

        consistency_loss = F.kl_div(log_anchors_prob, sharpen(
            positives_prob.detach(), T=self.sharpen_temperature), reduction="batchmean")

        # Total loss
        total_loss = consistency_loss

        return total_loss


class OursLossGlobal(nn.Module):
    # From ConfidenceBasedCE
    def __init__(self, threshold, reweight, num_classes, num_heads, mean_outside_mask=False, use_count_ema=False, momentum=0., data_len=None, reweight_renorm=False):
        super(OursLossGlobal, self).__init__()
        self.threshold = threshold
        self.reweight = reweight
        # setting reweight_renorm to True ignores reweight
        self.reweight_renorm = reweight_renorm

        if self.reweight_renorm:
            print("Reweight renorm is enabled")
        else:
            print("Reweight renorm is not enabled")

        self.mean_outside_mask = mean_outside_mask
        self.use_count_ema = use_count_ema

        self.num_classes = num_classes
        self.num_heads = num_heads

        self.momentum = momentum

        if use_count_ema:
            print("Data length:", data_len)
            self.data_len = data_len
            self.register_buffer("count_ema", torch.ones(
                (num_heads, num_classes)) / num_classes)
        self.register_buffer("num_counts", torch.zeros(1, dtype=torch.long))

    # Equivalent to: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    # With one-hot target
    def kl_div_loss(self, input, target, mask, weight, mean_outside_mask):
        if torch.all(mask == 0):
            # Return 0 as loss if nothing is in mask
            return torch.tensor(0., device=input.device)

        b = input.shape[0]

        # Select samples that pass the confidence threshold
        input = torch.masked_select(
            input, mask.view(b, 1)).view((-1, input.shape[1]))
        target = torch.masked_select(target, mask)

        log_prob = -F.log_softmax(input, dim=1)
        if weight is not None:
            # Weighted KL divergence
            log_prob = log_prob * weight.view((1, -1))
        loss = torch.gather(log_prob, 1, target.view((-1, 1))).view(-1)

        if mean_outside_mask:
            # Normalize by a constant (batch size)
            return loss.sum(dim=0) / b
        else:
            if weight is not None:
                # Take care of weighted sum
                weight_sum = weight[target].sum(dim=0)
                return (loss / weight_sum).sum(dim=0)
            else:
                return loss.mean(dim=0)

    def forward(self, head_id, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations
        output: cross entropy
        """
        # Retrieve target and mask based on weakly augmentated anchors

        weak_anchors_prob = F.softmax(anchors_weak, dim=1)

        max_prob, target = torch.max(weak_anchors_prob, dim=1)
        mask = max_prob > self.threshold
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        if self.use_count_ema:
            with torch.no_grad():
                head_count_ema = self.count_ema[head_id]

                # Normalized and adjusted with data_len
                count_in_batch = torch.bincount(
                    target_masked, minlength=c) / n * self.data_len
                head_count_ema = head_count_ema * self.momentum + \
                    count_in_batch * (1 - self.momentum)
                self.count_ema[head_id] = head_count_ema

        if head_id == 0:
            self.num_counts += 1

        # Class balancing weights
        # This is also used for debug purpose

        # reweight_renorm is equivalent to reweight when mean_outside_mask is False
        if self.reweight_renorm:
            idx, counts = torch.unique(target_masked, return_counts=True)
            # if self.use_count_ema:
            #     print("WARNING: count EMA used with class balancing")
            freq = float(n) / len(idx) / counts.float()
            weight = torch.ones(c).cuda()
            weight[idx] = freq
        elif self.reweight:
            idx, counts = torch.unique(target_masked, return_counts=True)
            if self.use_count_ema:
                print("WARNING: count EMA used with class balancing")
            freq = 1/(counts.float()/n)
            weight = torch.ones(c).cuda()
            weight[idx] = freq
        else:
            weight = None

        # Loss

        loss = self.kl_div_loss(input=anchors_strong, target=target, mask=mask,
                                weight=weight, mean_outside_mask=self.mean_outside_mask)

        if head_id == 0 and self.num_counts % 200 == 1:
            with torch.no_grad():
                idx, counts = torch.unique(target_masked, return_counts=True)
            if self.use_count_ema:
                print("use_count_ema max: {:.3f}, min: {:.3f}, median: {:.3f}, mean: {:.3f}".format(head_count_ema.max().item(),
                                                                                                      head_count_ema.min().item(), torch.median(head_count_ema).item(), head_count_ema.mean().item()))
            # print("weak_anchors_prob, mean across batch (from weak anchor of global loss): {}".format(
                # weak_anchors_prob.detach().mean(dim=0)))
            # print("Mask: {} / {} ({:.2f}%)".format(mask.sum(),
                                                    #  mask.shape[0], mask.sum() * 100. / mask.shape[0]))
            # print("idx: {}, counts: {}".format(idx, counts))

            if True:  # Verbose: print max confidence of each class
                m = torch.zeros((self.num_classes,))
                for i in range(self.num_classes):
                    v = max_prob[target == i]
                    if len(v):
                        m[i] = v.max()

                # print("Max of each cluster: {}".format(m))

        return loss








###### SSL Auxiliary Functions:-----------------




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

    # Ensure labels are in a compatible format for concatenation
    labels_one_hot = labels

    # Step 2: Mix labeled and unlabeled data by applying Mixup
    # Concatenate for Mixup
    all_inputs = torch.cat([labeled_data, unlabeled_data], dim=0)
    all_targets = torch.cat([labels_one_hot, pseudo_labels], dim=0)

    mixed_inputs, mixed_labels = mixup(all_inputs, all_targets, alpha_mixup)

    return mixed_inputs, mixed_labels










###### Training the USL-t Model:-----------------

def usl_t_pretrain(embeddings,device):

    # Convert embeddings to PyTorch tensors
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float).to(device)

    # Create a TensorDataset and DataLoader without labels
    dataset = TensorDataset(embeddings_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize your ClusteringModel
    model = ClusteringModel(nclusters=n_clusters, embedding_dim=embeddings_tensor.size(1), nheads=num_heads).to(device)

    # Initialize the optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Define your local and global loss functions
    criterion_local = OursLossLocal(num_classes=n_clusters, num_heads=num_heads, momentum=0.1, adjustment_weight=0.1, sharpen_temperature=0.5).to(device)
    criterion_global = OursLossGlobal(threshold=0.8, reweight=True, num_classes=n_clusters, num_heads=num_heads, mean_outside_mask=False, use_count_ema=False, momentum=0.1, data_len=len(dataset)).to(device)

    # Training loop
    for epoch in range(num_epochs_cluster):
        print(f"Epoch {epoch+1}/{num_epochs_cluster}")
        model.train()
        total_loss, total_local_loss, total_global_loss = 0.0, 0.0, 0.0

        for embeddings_batch in dataloader:
            embeddings_batch = embeddings_batch[0].to(device)

            # Forward pass
            outputs = model(embeddings_batch)

            local_loss_sum = torch.tensor(0.0).to(device)
            global_loss_sum = torch.tensor(0.0).to(device)
            for head_id, output in enumerate(outputs):
                # Calculate local loss
                local_loss = criterion_local(head_id=head_id, anchors=output, neighbors=output)  # Adjust according to your data
                local_loss_sum += local_loss

                # Calculate global loss
                global_loss = criterion_global(head_id=head_id, anchors_weak=output, anchors_strong=output)  # Adjust according to your data
                global_loss_sum += global_loss

            # Combine losses
            loss = local_loss_sum + global_loss_sum

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_local_loss += local_loss_sum.item()
            total_global_loss += global_loss_sum.item()

        print(f"Epoch {epoch+1}/{num_epochs_cluster}, Total Loss: {total_loss/len(dataloader)}, Local Loss: {total_local_loss/len(dataloader)}, Global Loss: {total_global_loss/len(dataloader)}")
    
    # Save the trained model
    # Check if the base filename exists, and if so, create a new filename
    save_model(model, model_path, base_filename_cluster)






## usl_t_pretrain with Early Stopping:-----------------
def validate_model(model, validation_loader, device, criterion_local, criterion_global):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in validation_loader:
            embeddings_batch = batch[0].to(device)

            outputs = model(embeddings_batch)

            local_loss_sum = torch.tensor(0.0).to(device)
            global_loss_sum = torch.tensor(0.0).to(device)
            for head_id, output in enumerate(outputs):
                # Here, we mimic the local and global loss calculations
                # For simplicity, we use the same output as both anchor and neighbor,
                # but you should adjust this based on your validation data's nature
                local_loss = criterion_local(head_id=head_id, anchors=output, neighbors=output)
                global_loss = criterion_global(head_id=head_id, anchors_weak=output, anchors_strong=output)

                local_loss_sum += local_loss
                global_loss_sum += global_loss

            loss = local_loss_sum + global_loss_sum
            total_loss += loss.item()

    avg_loss = total_loss / len(validation_loader)
    return avg_loss




def usl_t_pretrain_with_early_stopping(embeddings, device, validation_loader, patience):
    # Convert embeddings to PyTorch tensors
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float).to(device)

    # Create a TensorDataset and DataLoader without labels
    dataset = TensorDataset(embeddings_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize your ClusteringModel
    model = ClusteringModel(nclusters=n_clusters, embedding_dim=embeddings_tensor.size(1), nheads=num_heads).to(device)

    # Initialize the optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Define your local and global loss functions
    criterion_local = OursLossLocal(num_classes=n_clusters, num_heads=num_heads, momentum=0.1, adjustment_weight=0.1, sharpen_temperature=0.5).to(device)
    criterion_global = OursLossGlobal(threshold=0.8, reweight=True, num_classes=n_clusters, num_heads=num_heads, mean_outside_mask=False, use_count_ema=False, momentum=0.1, data_len=len(dataset)).to(device)

    best_loss = float('inf')
    no_improve_epoch = 0

    # Training loop
    for epoch in range(num_epochs_cluster):
        model.train()
        total_loss, total_local_loss, total_global_loss = 0.0, 0.0, 0.0

        for embeddings_batch in dataloader:
            embeddings_batch = embeddings_batch[0].to(device)

            # Forward pass
            outputs = model(embeddings_batch)

            local_loss_sum = torch.tensor(0.0).to(device)
            global_loss_sum = torch.tensor(0.0).to(device)
            for head_id, output in enumerate(outputs):
                # Calculate local loss
                local_loss = criterion_local(head_id=head_id, anchors=output, neighbors=output)
                local_loss_sum += local_loss

                # Calculate global loss
                global_loss = criterion_global(head_id=head_id, anchors_weak=output, anchors_strong=output)
                global_loss_sum += global_loss

            # Combine losses
            loss = local_loss_sum + global_loss_sum

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_local_loss += local_loss_sum.item()
            total_global_loss += global_loss_sum.item()

        # Early Stopping Check using validation dataset
        val_loss = validate_model(model, validation_loader, device, criterion_local, criterion_global)
        print(f"Epoch {epoch+1}/{num_epochs_cluster}, Validation Loss: {val_loss}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve_epoch = 0
            # Save the best model
             
        else:
            no_improve_epoch += 1

        if no_improve_epoch >= patience:
            print("Early stopping triggered after epoch:", epoch+1)
            save_model(model, model_path, base_filename_cluster) 
            break
        
    # Save model at the end of training, if it is not already saved due to early stopping
    if no_improve_epoch < patience:
        save_model(model,model_path, base_filename_cluster)  





def usl_t_selective_labels(embeddings,device):
    # Initialize the model
    model = ClusteringModel(nclusters=n_clusters, embedding_dim=cluster_embedding_dim, nheads=num_heads)
    model = load_model(model, model_path, base_filename_cluster)
    model.eval()
    model.to(device)

    # Convert embeddings to tensor and move to the specified device
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float).to(device)

    # Function to process embeddings in batches and collect probabilities
    def get_sample_selection_indices(embeddings_tensor, model, final_sample_num):
        all_probs = []
        with torch.no_grad():  # Disable gradient computation for efficiency
            for start_idx in range(0, embeddings_tensor.size(0), 32):  # 32 is the batch size
                end_idx = min(start_idx + 32, embeddings_tensor.size(0))
                embeddings_batch = embeddings_tensor[start_idx:end_idx]
                
                outputs = model(embeddings_batch)
                probs = [torch.softmax(output, dim=1) for output in outputs]
                
                # Calculate the average probability across all heads for the batch
                avg_probs_batch = torch.stack(probs).mean(dim=0)
                all_probs.append(avg_probs_batch)

        # Concatenate all batch probabilities and select top samples
        all_probs = torch.cat(all_probs, dim=0)
        _, selected_indices = torch.topk(all_probs.max(dim=1).values, final_sample_num)

        return selected_indices.cpu().numpy()

    # Get the indices of selected samples
    selected_indices = get_sample_selection_indices(embeddings_tensor, model, final_sample_num=n_clusters)
    return selected_indices
   


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
            # targets = F.log_softmax(labels_one_hot, dim=1)
            loss = criterion(outputs, labels_one_hot)
            total_loss += loss.item()
    avg_loss = total_loss / len(validation_loader)
    return avg_loss


def apply_mixmatch_with_early_stopping(labeled_loader, unlabeled_loader, validation_loader, model, device, optimizer, num_epochs,patience):
    print("Applying MixMatch with early stopping...")
    best_loss = float('inf')
    no_improve_epoch = 0
    
    # Assuming kl_divergence_loss is being used as criterion for training, define or replace it accordingly
    criterion = kl_divergence_loss
    
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

# TODO: Compare different embeddings
def train(embeddings, labels, embeddings_val, labels_val,recalculate_indices,plot_filepath_segments,plot_filename):
    # Set random seed for reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        device = 'cuda'
    else:
        device = 'cpu'
        
    print("Device: ", device)
    print("Training the USL-t SSL model...:  ") 

    # Validation dataset
    val_dataset = TensorDataset(torch.tensor(embeddings_val, dtype=torch.float32), 
                                torch.tensor(labels_val, dtype=torch.float32))
    validation_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    if recalculate_indices:
        print("Recalculating indices...")
        usl_t_pretrain_with_early_stopping(embeddings,device,validation_loader,patience_cluster)
        selected_indices = usl_t_selective_labels(embeddings,device)
        visualize.visualize_clusters(embeddings,labels,selected_indices,plot_filepath_segments,plot_filename)  
        make_dataset_segments.save_selected_indices_usl_t(selected_indices) 
    else:
        print("Loading selected indices...")
        selected_indices = make_dataset_segments.load_selected_indices_usl_t()
        visualize.visualize_clusters(embeddings,labels,selected_indices,plot_filepath_segments,plot_filename)
    print("Selected indices:", selected_indices)       

    input_dim = embeddings.shape[1]  # Dynamically assign input_dim
    # num_classes = len(np.unique(labels))  # Dynamically determine num_classes
    
    # Preparing DataLoaders from the USL step
    # Assuming `labeled_embeddings`, `unlabeled_embeddings`, and `labels` are ready

    # Model Initialization
    model = EmbeddingClassifier(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Convert embeddings and labels to DataLoader
    labeled_embeddings = embeddings[selected_indices]
    labeled_labels = np.array(labels)[selected_indices]  # Ensure labels_train is an array for consistent indexing


    all_indices = np.arange(len(embeddings))  # Array of all indices
    unlabeled_indices = np.setdiff1d(all_indices, selected_indices)  # Exclude selected_indices to get unlabeled ones
    unlabeled_embeddings = embeddings[unlabeled_indices]
    # Convert to tensors and create datasets
    labeled_dataset = TensorDataset(torch.tensor(labeled_embeddings, dtype=torch.float32).to(device),
                                    torch.tensor(labeled_labels, dtype=torch.float32).to(device))
    unlabeled_dataset = TensorDataset(torch.tensor(unlabeled_embeddings, dtype=torch.float32).to(device))


    
    # DataLoaders
    labeled_loader = DataLoader(labeled_dataset, batch_size=64, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=True)




    
    if early_stoppage:
        # Apply MixMatch
        apply_mixmatch_with_early_stopping(labeled_loader, unlabeled_loader, validation_loader, model, device, optimizer, num_epochs,patience)
    
    
    
   
   
   
   
###### Evaluate the SSL model on the validation dataset:----------------- 
    
def evaluate(embeddings_val, labels_val,data):
    print("Evaluating the USL SSL model on the {data} dataset...:  ")
    device=get_device()
    # Load the true labels
    val_labels = np.array(labels_val)
    print("True labels: ",val_labels)
    
    # Predictions from the SSL model
    val_predictions_usl_ssl = predict_segments(embeddings_val, model_filepath, num_classes, device)
    print("Predictions from the SSL model: ",val_predictions_usl_ssl)
    
    precision_macro = precision_score(val_labels, val_predictions_usl_ssl, average='macro')
    recall_macro = recall_score(val_labels, val_predictions_usl_ssl, average='macro')
    f1_macro = f1_score(val_labels, val_predictions_usl_ssl, average='macro')

    precision_weighted = precision_score(val_labels, val_predictions_usl_ssl, average='weighted')
    recall_weighted = recall_score(val_labels, val_predictions_usl_ssl, average='weighted')
    f1_weighted = f1_score(val_labels, val_predictions_usl_ssl, average='weighted')
    
    

    # Preparing data for DataFrame
    data = {
        "Metric": ["precision_macro", "recall_macro", "f1_macro", "precision_weighted","recall_weighted", "f1_weighted"],
        "SSL Model": [precision_macro, recall_macro, f1_macro, precision_weighted, recall_weighted, f1_weighted],
    }

    # Creating DataFrame
    df = pd.DataFrame(data)

    # Formatting for nicer display
    df_formatted = df.copy()
    df_formatted['SSL Model'] = df_formatted['SSL Model'].map('{:,.2f}'.format)

    # Display DataFrame
    print(df_formatted)
           
    



