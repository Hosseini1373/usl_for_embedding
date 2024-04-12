import os
import torch
from datetime import datetime  # Adjusted import

def save_model(model, model_path, base_filename):    
    # Save the trained model
    # Check if the base filename exists, and if so, create a new filename
    model_file_path = os.path.join(model_path, base_filename)
    if os.path.isfile(model_file_path):
        #overwrite the last model:
        print("SSL model is saved under: ", model_file_path)
        # Generate a unique filename with a timestamp to avoid overwriting
        torch.save(model.state_dict(), base_filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # Adjusted call
        base_name, ext = os.path.splitext(base_filename)
        new_filename = f"{base_name}_{timestamp}{ext}"
        model_file_path = os.path.join(model_path, new_filename)
    
    print("SSL model is saved to: ", model_file_path)
    torch.save(model.state_dict(), model_file_path)
    # It seems like you're saving the model twice, once with the potentially new filename and once with the base filename.
    # You likely only need to save it once, so you might want to remove the following line:




def load_model(model,model_path, base_filename):
    # Load the trained model
    model_file_path=model_path
    if base_filename:
        model_file_path = os.path.join(model_path, base_filename)
    print("Loading model from: ", model_file_path)
    model.load_state_dict(torch.load(model_file_path))
    print("Model is loaded from: ", model_file_path)
    return model