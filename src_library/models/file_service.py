import os
from matplotlib import pyplot as plt
import torch
from datetime import datetime  # Adjusted import

def save_model(model, model_path, base_filename):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name, ext = os.path.splitext(base_filename)
    new_filename = f"{base_name}_{timestamp}{ext}"
    model_file_path = os.path.join(model_path, new_filename)
    
    torch.save(model.state_dict(), model_file_path)
    print("SSL model is saved to: ", model_file_path)







def load_model(model, model_path, base_filename):
    # Adjust to find the latest file based on the modified naming convention
    if base_filename:
        latest_file = None
        latest_time = None
        print("base_filename: ",base_filename)
        print("model_path: ",model_path)
        base_filename = base_filename.replace('.pth', '')  # Remove .pth to match files correctly

        for file in os.listdir(model_path):
            if file.startswith(base_filename) and file.endswith('.pth'):
                file_path = os.path.join(model_path, file)
                file_time = os.path.getmtime(file_path)
                if latest_time is None or file_time > latest_time:
                    latest_file = file_path
                    latest_time = file_time

        if latest_file is None:
            raise FileNotFoundError(f"No model file found for pattern '{base_filename}' in '{model_path}'")

        print("Loading model from: ", latest_file)
        model.load_state_dict(torch.load(latest_file))
        print("Model is loaded from: ", latest_file)
        return model
    else:
        latest_file = None
        latest_time = None
        print("original base_filename: ",base_filename)
        print("model_path: ",model_path)
        base_filename = model_path.split("/")[-1]
        model_path = model_path.replace(base_filename, "")
        print("base_filename: ",base_filename)
        base_filename = base_filename.replace('.pth', '')  # Remove .pth to match files correctly

        for file in os.listdir(model_path):
            if file.startswith(base_filename) and file.endswith('.pth'):
                file_path = os.path.join(model_path, file)
                file_time = os.path.getmtime(file_path)
                if latest_time is None or file_time > latest_time:
                    latest_file = file_path
                    latest_time = file_time

        if latest_file is None:
            raise FileNotFoundError(f"No model file found for pattern '{base_filename}' in '{model_path}'")

        print("Loading model from: ", latest_file)
        model.load_state_dict(torch.load(latest_file))
        print("Model is loaded from: ", latest_file)
        return model        