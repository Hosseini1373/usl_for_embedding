import torch
from torch import nn
from src.models.ssl_models.embedding_classifier import EmbeddingClassifier
import numpy as np
from src.models.file_service import load_model

def predict(embeddings_val, model_filepath, num_labels,device):
    # Set random seed for reproducibility
    
    print(f"Using device: {device}")
    model = EmbeddingClassifier(embeddings_val.shape[1], num_labels)
    # 
    # Assuming model, val_data, and device are already defined and available
    embeddings_val = np.array(embeddings_val)  # convert string lists to actual lists

    # Convert validation embeddings to a tensor and move to the appropriate device
    val_embeddings_tensor = torch.tensor(np.array(embeddings_val), dtype=torch.float32).to(device)

    model=load_model(model, model_filepath,base_filename=None)
    model.to(device)
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Obtain model predictions
        val_outputs = model(val_embeddings_tensor)
        # Convert softmax outputs to predicted class indices
        val_predictions_ssl = torch.argmax(val_outputs, dim=1).cpu().numpy()
        
    return val_predictions_ssl




def predict_curlie(embeddings_val, model_filepath, num_labels,device):
    # Set random seed for reproducibility
    
    print(f"Using device: {device}")
    model = EmbeddingClassifier(embeddings_val.shape[1], num_labels)
    # 
    # Assuming model, val_data, and device are already defined and available
    embeddings_val = np.array(embeddings_val)  # convert string lists to actual lists

    # Convert validation embeddings to a tensor and move to the appropriate device
    val_embeddings_tensor = torch.tensor(np.array(embeddings_val), dtype=torch.float32).to(device)

    model=load_model(model, model_filepath,base_filename=None)
    model.to(device)
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Obtain model predictions
        val_outputs = model(val_embeddings_tensor)
        # Convert softmax outputs to predicted class indices
        val_predictions_ssl = torch.argmax(val_outputs, dim=1).cpu().numpy()
        
    return val_predictions_ssl