import torch
from torch import nn
from src.models.ssl_models.embedding_classifier import EmbeddingClassifier
import numpy as np

def predict(val_data, embedding_column, model_filepath, num_labels=2):
    # Set random seed for reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        device = 'cuda'
    else:
        device = 'cpu'


    # Assuming model, val_data, and device are already defined and available
    embeddings_val = np.array(val_data[embedding_column].tolist())  # convert string lists to actual lists

    # Convert validation embeddings to a tensor and move to the appropriate device
    val_embeddings_tensor = torch.tensor(np.array(embeddings_val), dtype=torch.float32).to(device)

    #Reading the model:
    model = EmbeddingClassifier(embeddings_val.shape[1], num_labels)
    model_state_dict = torch.load(model_filepath)
    model.load_state_dict(model_state_dict)
    model.to(device)

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Obtain model predictions
        val_outputs = model(val_embeddings_tensor)
        # Convert softmax outputs to predicted class indices
        val_predictions_ssl = torch.argmax(val_outputs, dim=1).cpu().numpy()
        
    return val_predictions_ssl