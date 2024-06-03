import torch
import torch.onnx
from torch import nn
from graphviz import Digraph

# Define your neural network architecture for SSL
class EmbeddingClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(EmbeddingClassifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def visualize_embedding_classifier():
    dot = Digraph()
    
    # Adding nodes for layers
    dot.node('Input', 'Input\n(embedding_dim)')
    dot.node('Linear1', 'Linear1\n(embedding_dim, 256)')
    dot.node('ReLU1', 'ReLU')
    dot.node('Linear2', 'Linear2\n(256, 128)')
    dot.node('ReLU2', 'ReLU')
    dot.node('Linear3', 'Linear3\n(128, num_classes)')
    dot.node('Output', 'Output\n(num_classes)')
    
    # Adding edges between nodes
    dot.edge('Input', 'Linear1')
    dot.edge('Linear1', 'ReLU1')
    dot.edge('ReLU1', 'Linear2')
    dot.edge('Linear2', 'ReLU2')
    dot.edge('ReLU2', 'Linear3')
    dot.edge('Linear3', 'Output')

    # Render the graph to a file
    dot.format = 'png'
    dot.render('EmbeddingClassifier_simple')

if __name__ == "__main__":
    visualize_embedding_classifier()