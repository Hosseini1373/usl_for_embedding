from torch import nn

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