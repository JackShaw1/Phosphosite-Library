import torch
import torch.nn as nn
import torch.nn.functional as F

class PhosNetBinaryClassifier(nn.Module):
    
    def __init__(self, num_aa_types=20):  
        super(PhosNetBinaryClassifier, self).__init__()

        # Embedding for amino acid type
        self.aa_embedding = nn.Embedding(num_aa_types, 4)  

        # 1D convolutions
        self.conv1 = nn.Conv1d(12, 36, 1)
        self.conv2 = nn.Conv1d(36, 72, 1)
        self.conv3 = nn.Conv1d(72, 144, 1)

        self.bn1 = nn.BatchNorm1d(36)
        self.bn2 = nn.BatchNorm1d(72)
        self.bn3 = nn.BatchNorm1d(144)

        # Fully connected layers (Fixed input size to 144 instead of 144 * max_len)
        self.fc1 = nn.Linear(144, 288)  # Now 144 instead of 2880
        self.fc2 = nn.Linear(288, 144)
        self.fc3 = nn.Linear(144, 1)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        batch_size, num_points, num_features = x.shape

        # If num_points == 0, return a trainable tensor instead of a detached constant
        if num_points == 0:
            empty_features = torch.zeros((batch_size, 144), device=x.device, dtype=x.dtype, requires_grad=True)
            connected_features = self.fc3(empty_features)  # Pass through fc3 to keep computation graph
            return torch.sigmoid(connected_features).squeeze(-1)

        # Extract atom coordinates
        xyz = x[:, :, :3]  # (batch, num_points, 3)

        # Extract chain ID (one-hot)
        chain_id = F.one_hot(x[:, :, 3].long(), num_classes=2).float()  # (batch, num_points, 2)

        # Extract amino acid type (embedding)
        aa_type = self.aa_embedding(x[:, :, 4].long())  # (batch, num_points, 4)

        # Extract atom type (one-hot)
        atom_type = F.one_hot(x[:, :, 5].long(), num_classes=3).float()  # (batch, num_points, 3)

        # Concatenate features
        features = torch.cat([xyz, chain_id, aa_type, atom_type], dim=2)  # (batch, num_points, 12)

        # Permute to match Conv1d expected input (batch, channels, num_points)
        features = features.permute(0, 2, 1)  # (batch, 12, num_points)

        # Layer 1
        if num_points > 1:
            features = F.relu(self.bn1(self.conv1(features)))  # (batch, 36, num_points)
        else:
            features = F.relu(self.conv1(features))  # (batch, 36, 1)

        # Layer 2
        if num_points > 1:
            features = F.relu(self.bn2(self.conv2(features)))  # (batch, 72, num_points)
        else:
            features = F.relu(self.conv2(features))  # (batch, 72, 1)

        # Layer 3
        if num_points > 1:
            features = self.bn3(self.conv3(features))  # (batch, 144, num_points)
        else:
            features = self.conv3(features)  # (batch, 144, 1)

        # Global Max Pooling
        features = F.adaptive_max_pool1d(features, 1).squeeze(-1)  # (batch, 144)

        # Fully connected layers
        connected_features = F.relu(self.fc1(features))
        connected_features = self.dropout(F.relu(self.fc2(connected_features)))
        connected_features = self.fc3(connected_features)  # (batch, 1)

        return torch.sigmoid(connected_features).squeeze(-1)  # Correct final shape: (batch,)
