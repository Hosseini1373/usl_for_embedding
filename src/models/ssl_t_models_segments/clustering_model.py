from torch import nn
import torch
from torchviz import make_dot
from graphviz import Digraph

class ClusteringModel(nn.Module):
    def __init__(self, nclusters=100, embedding_dim=1024, nheads=3):
        super(ClusteringModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.nheads = nheads

        self.cluster_heads = nn.ModuleList([nn.Linear(embedding_dim, nclusters) for _ in range(nheads)])

    def forward(self, x):
        # Since we are directly using embeddings, there's no backbone model involved here.
        features = x
        outputs = [cluster_head(features) for cluster_head in self.cluster_heads]
        return outputs
    
    
    
def visualize_model():
    dot = Digraph()
    
    # Adding nodes for layers
    dot.node('Input', 'Input\n(1, 1024)')
    dot.node('Linear1', 'Linear1\n(1024, 100)')
    dot.node('Linear2', 'Linear2\n(1024, 100)')
    dot.node('Linear3', 'Linear3\n(1024, 100)')
    dot.node('Output1', 'Output1\n(1, 100)')
    dot.node('Output2', 'Output2\n(1, 100)')
    dot.node('Output3', 'Output3\n(1, 100)')
    
    # Adding edges between nodes
    dot.edge('Input', 'Linear1')
    dot.edge('Input', 'Linear2')
    dot.edge('Input', 'Linear3')
    dot.edge('Linear1', 'Output1')
    dot.edge('Linear2', 'Output2')
    dot.edge('Linear3', 'Output3')

    # Render the graph to a file
    dot.format = 'png'
    dot.render('USL_t_model_simple')

if __name__ == "__main__":
    visualize_model()