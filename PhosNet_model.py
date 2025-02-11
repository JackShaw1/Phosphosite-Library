import torch
import torch.nn as nn
import torch.nn.functional as F

class PhosNetBinaryClassifier(nn.Module):
    
    def __init__(self, num_aa_types=20, max_len=20):  
        super(PhosNetBinaryClassifier, self).__init__()

        self.max_len = max_len  # Store max_len for reference

        # Embedding for amino acid type
        self.aa_embedding = nn.Embedding(num_aa_types, 4)  

        # 1D convolutions
        self.conv1 = nn.Conv1d(12, 36, 1)
        self.conv2 = nn.Conv1d(36, 72, 1)
        self.conv3 = nn.Conv1d(72, 144, 1)

        self.bn1 = nn.InstanceNorm1d(36, affine=True)
        self.bn2 = nn.InstanceNorm1d(72, affine=True)
        self.bn3 = nn.InstanceNorm1d(144, affine=True)

        # **Fix input size based on max_len**
        fc_input_size = 144 * max_len  # Ensure consistency with expected input shape
        self.fc1 = nn.Linear(fc_input_size, 288)
        self.fc2 = nn.Linear(288, 144)
        self.fc3 = nn.Linear(144, 1)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        batch_size, num_points, num_features = x.shape

        # Extract atom coordinates
        xyz = x[:, :, :3]

        # Extract chain id, amino acid type, and atom element for atom
        chain_id = F.one_hot(x[:, :, 3].long(), num_classes=2).float()
        aa_type = self.aa_embedding(x[:, :, 4].long()) 
        atom_type = F.one_hot(x[:, :, 5].long(), num_classes=3).float()

        # Concatenate spatial and chemical features
        features = torch.cat([xyz, chain_id, aa_type, atom_type], dim=2).permute([0, 2, 1])

        # Layer 1
        features = self.conv1(features)
        features = self.bn1(features)
        features = F.relu(features)

        # Layer 2
        features = self.conv2(features)
        features = self.bn2(features)
        features = F.relu(features)

        # Layer 3
        features = self.conv3(features)
        features = self.bn3(features)

        # Flatten to match fc1's expected input size
        features = features.view(batch_size, -1)  # Shape: (batch_size, 144 * max_len)

        # Fully connected layers
        connected_features = F.relu(self.fc1(features))
        connected_features = self.dropout(F.relu(self.fc2(connected_features)))
        connected_features = self.fc3(connected_features)

        # Remove sigmoid if using BCEWithLogitsLoss
        output = torch.sigmoid(connected_features)  
        
        return output.squeeze(1)
