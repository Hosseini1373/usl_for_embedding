import os
import torch
import datetime

def save_model(model,model_path, base_filename):    
    # Save the trained model
    # Check if the base filename exists, and if so, create a new filename
    model_file_path = os.path.join(model_path, base_filename)
    if os.path.isfile(model_file_path):
        # Generate a unique filename with a timestamp to avoid overwriting
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name, ext = os.path.splitext(base_filename)
        new_filename = f"{base_name}_{timestamp}{ext}"
        model_file_path = os.path.join(model_path, new_filename)
        
    torch.save(model.state_dict(), model_file_path)