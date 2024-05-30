
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


import os
import matplotlib.pyplot as plt
from datetime import datetime


def save_plot(plot_path, base_filename):
    # Generate the full path for the plot file
    print("Saving the plot to: ", plot_path+base_filename)
    model_file_path = os.path.join(plot_path, base_filename)
    
    # Check if the file already exists
    if os.path.isfile(model_file_path):
        plt.savefig(model_file_path)
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


# def visualize_clusters(embeddings, labels,selected_indices, plot_path, base_filename):
#     # Reduce the dimensionality for visualization
#     pca = PCA(n_components=2)
#     embeddings_2d = pca.fit_transform(embeddings)

#     plt.figure(figsize=(10, 8))
#     plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, label='All Points')
#     selected_points = embeddings_2d[selected_indices]
#     plt.scatter(selected_points[:, 0], selected_points[:, 1], color='red', marker='x', label='Selected Points', s=100)

#     plt.legend()
#     plt.title('PCA Projection of Embeddings')
#     plt.xlabel('PCA Component 1')
#     plt.ylabel('PCA Component 2')

#     save_plot(plot_path, base_filename)

import numpy as it
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

def visualize_clusters(embeddings, labels, selected_indices, plot_path, base_filename):
    # Reduce the dimensionality for visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Setup plot
    plt.figure(figsize=(10, 8))
    
    # Handle both single-label and multi-label scenarios
    if all(isinstance(label, np.ndarray) for label in labels):
        # Multilabel scenario: convert label lists to tuple for uniqueness
        labels=[label.tolist() for label in labels]
        unique_labels = list(set(tuple(label) for label in labels))
        label_to_color = {label: i for i, label in enumerate(unique_labels)}
        colors = [label_to_color[tuple(label)] for label in labels]
    else:
        # Single-label scenario
        unique_labels = list(set(labels))
        label_to_color = {label: i for i, label in enumerate(unique_labels)}
        colors = [label_to_color[label] for label in labels]

    # Generate a color map
    cmap = plt.cm.get_cmap('viridis', len(unique_labels))

    # Plot all points with coloring by labels
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, cmap=cmap, alpha=0.5)
    
    # Highlight selected points
    selected_points = embeddings_2d[selected_indices]
    plt.scatter(selected_points[:, 0], selected_points[:, 1], color='red', marker='x', label='Selected Points', s=100)

    # Add colorbar to the plot
    cbar = plt.colorbar(scatter, ticks=range(len(unique_labels)))
    cbar.set_label('Class Labels')
    cbar.set_ticklabels([str(label) for label in unique_labels])

    plt.legend()
    plt.title('PCA Projection of Embeddings')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    # Save the plot
    save_plot(plot_path, base_filename)



    
    
    if __name__ == '__main__':
        visualize_clusters(embeddings,labels,selected_indices)