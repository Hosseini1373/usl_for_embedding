from torch import nn

class EmbeddingClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, dropout_rate=0.5):
        super(EmbeddingClassifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout layer after first ReLU
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout layer after second ReLU
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)  # Applying dropout after first activation
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)  # Applying dropout after second activation
        x = self.fc3(x)
        return x
