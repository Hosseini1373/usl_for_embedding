from torch import nn



# Model Output Layer: Ensure that the output layer 
# of your EmbeddingClassifier has an activation function 
# suitable for multilabel classification. In your current setup, 
# it seems the final layer (self.fc3) does not specify an activation. 
# For multilabel classification with BCEWithLogitsLoss, it’s appropriate 
# to not have an activation because the loss function handles the sigmoid 
# activation internally.

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
    
   
# #Enhanced Veriosion
# class EmbeddingClassifier(nn.Module):
#     def __init__(self, embedding_dim, num_classes):
#         super(EmbeddingClassifier, self).__init__()
#         # Input to hidden layer 1
#         self.fc1 = nn.Linear(embedding_dim, 512)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(0.5)  # Add dropout for regularization
        
#         # Hidden layer 1 to hidden layer 2
#         self.fc2 = nn.Linear(512, 256)
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(0.3)  # Add dropout for regularization
        
#         # Hidden layer 2 to hidden layer 3
#         self.fc3 = nn.Linear(256, 128)
#         self.relu3 = nn.ReLU()
#         self.dropout3 = nn.Dropout(0.2)  # Add dropout for regularization

#         # Hidden layer 3 to output
#         self.fc4 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu1(x)
#         x = self.dropout1(x)
        
#         x = self.fc2(x)
#         x = self.relu2(x)
#         x = self.dropout2(x)
        
#         x = self.fc3(x)
#         x = self.relu3(x)
#         x = self.dropout3(x)
        
#         x = self.fc4(x)  # Output layer does not have an activation function
#         return x


    