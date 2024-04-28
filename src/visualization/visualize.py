
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


import os
import matplotlib.pyplot as plt
from datetime import datetime


def save_plot(plot_path, base_filename):
    # Generate the full path for the plot file
    print("Saving the plot to: ", plot_path,base_filename)
    model_file_path = os.path.join(plot_path, base_filename)
    
    # Check if the file already exists
    if os.path.isfile(model_file_path):
        print("File already exists. Saving with a new timestamp.")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name, ext = os.path.splitext(base_filename)
        new_filename = f"{base_name}_{timestamp}{ext}"
        model_file_path = os.path.join(plot_path, new_filename)
    else:
        print("Saving new plot.")

    # Save the plot to the determined path
    plt.savefig(model_file_path)
    print("Plot is saved to: ", model_file_path)


def visualize_clusters(embeddings, selected_indices, plot_path, base_filename):
    # Reduce the dimensionality for visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, label='All Points')
    selected_points = embeddings_2d[selected_indices]
    plt.scatter(selected_points[:, 0], selected_points[:, 1], color='red', marker='x', label='Selected Points', s=100)

    plt.legend()
    plt.title('PCA Projection of Embeddings')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    save_plot(plot_path, base_filename)




    
    
    if __name__ == '__main__':
        visualize_clusters(embeddings,selected_indices)