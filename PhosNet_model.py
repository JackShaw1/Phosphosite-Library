import torch
import torch.nn as nn
import torch.nn.functional as F

class PhosNetBinaryClassifier(nn.Module):
    
    def __init__(self, num_aa_types=20):  # Changed num_atom_types back to 16
        super(PhosNetBinaryClassifier, self).__init__()

        # Embedding for amino acid type
        self.aa_embedding = nn.Embedding(num_aa_types, 4)  

        # 1D convolutions
        self.conv1 = nn.Conv1d(12, 36, 1)
        self.conv2 = nn.Conv1d(36, 72, 1)
        self.conv3 = nn.Conv1d(72, 144, 1)

        self.bn1 = nn.InstanceNorm1d(36, affine=True)
        self.bn2 = nn.InstanceNorm1d(72, affine=True)
        self.bn3 = nn.InstanceNorm1d(144, affine=True)

        # Fully connected layers
        self.fc1 = nn.Linear(144, 288)
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
