import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetBinaryClassifier(nn.Module):
    
    def __init__(self, num_aa_types=20, num_atom_types=3, num_chain_ids=2):  # Changed num_atom_types back to 16
        super(PointNetBinaryClassifier, self).__init__()

        # Embedding layers
        self.aa_embedding = nn.Embedding(num_aa_types + 1, 3)  
        self.atom_embedding = nn.Embedding(num_atom_types + 1, 3)  
        self.chain_embedding = nn.Embedding(num_chain_ids + 1, 1)

        # PointNet feature extractor (1D Convolutions + InstanceNorm)
        self.conv1 = nn.Conv1d(10, 20, 1)
        self.conv2 = nn.Conv1d(20, 40, 1)
        self.conv3 = nn.Conv1d(40, 80, 1)

        self.bn1 = nn.InstanceNorm1d(20, affine=True)
        self.bn2 = nn.InstanceNorm1d(40, affine=True)
        self.bn3 = nn.InstanceNorm1d(80, affine=True)

        # Fully connected layers
        self.fc1 = nn.Linear(80, 160)
        self.fc2 = nn.Linear(160, 80)  # Increased from 20 → 80 to prevent bottleneck
        self.fc3 = nn.Linear(80, 1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        batch_size, num_points, num_features = x.shape

        # Extract spatial features (continuous)
        xyz = x[:, :, :3].permute(0, 2, 1) 

        # Extract chemical features (discrete categoricals encoded as integers)
        chain_id = x[:, :, 3].unsqueeze(1).float()
        aa_type = self.aa_embedding(x[:, :, 4].long()).permute(0, 2, 1)  
        atom_type = self.atom_embedding(x[:, :, 5].long()).permute(0, 2, 1)  

        # Concatenate spatial and chemical features
        features = torch.cat([xyz, chain_id, aa_type, atom_type], dim=1)

        # Apply InstanceNorm1d only if num_points > 1
        features = self.conv1(features)
        if num_points > 1:
            features = self.bn1(features)
        features = F.relu(features)

        features = self.conv2(features)
        if num_points > 1:
            features = self.bn2(features)
        features = F.relu(features)

        features = self.conv3(features)
        if num_points > 1:
            features = self.bn3(features)

        # Global max pooling
        features = torch.max(features, 2, keepdim=False)[0]

        # Fully connected layers
        x = F.relu(self.fc1(features))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        # Remove sigmoid if using BCEWithLogitsLoss
        x = torch.sigmoid(x)  
        
        return x.squeeze(1)
