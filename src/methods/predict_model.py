import torch
from torch import nn
from src.models.ssl_models.embedding_classifier import EmbeddingClassifier
from src.models.ssl_models_curlie.embedding_classifier import EmbeddingClassifier as EmbeddingClassifier_curlie
from src.models.ssl_models_segments.embedding_classifier import EmbeddingClassifier as EmbeddingClassifier_segments

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
    model = EmbeddingClassifier_curlie(embeddings_val.shape[1], num_labels)
    # 
    # Assuming model, val_data, and device are already defined and available
    embeddings_val = np.array(embeddings_val)  # convert string lists to actual lists

    # Convert validation embeddings to a tensor and move to the appropriate device
    val_embeddings_tensor = torch.tensor(np.array(embeddings_val), dtype=torch.float32).to(device)

    model=load_model(model, model_filepath,base_filename=None)
    model.to(device)
    threshold=0.5
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Obtain model predictions (logits)
        val_outputs = model(val_embeddings_tensor)
        # Convert logits to probabilities
        probabilities = torch.sigmoid(val_outputs)
        print("Probabilities:")
        print(probabilities)
        # Apply threshold to get binary predictions for each class
        predictions = (probabilities > threshold).int().cpu().numpy()
    return predictions



def predict_segments(embeddings_val, model_filepath, num_labels,device):
    # Set random seed for reproducibility
    
    print(f"Using device: {device}")
    model = EmbeddingClassifier_segments(embeddings_val.shape[1], num_labels)
    # 
    # Assuming model, val_data, and device are already defined and available
    embeddings_val = np.array(embeddings_val)  # convert string lists to actual lists

    # Convert validation embeddings to a tensor and move to the appropriate device
    val_embeddings_tensor = torch.tensor(np.array(embeddings_val), dtype=torch.float32).to(device)

    model=load_model(model, model_filepath,base_filename=None)
    model.to(device)
    threshold=0.5
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Obtain model predictions (logits)
        val_outputs = model(val_embeddings_tensor)
        # Convert logits to probabilities
        probabilities = torch.sigmoid(val_outputs)
        print("Probabilities:")
        print(probabilities)
        # Apply threshold to get binary predictions for each class
        predictions = (probabilities > threshold).int().cpu().numpy()
    return predictions