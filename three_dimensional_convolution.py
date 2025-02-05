import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Modified 3D convolution model for binary classification.
    - Uses 3d convolution to process spatial and chemical features together with embeddings.
    - Embeds categorical features (chain id, amino acid name, and atom element) encoded as integers.
    - Merges all features and outputs a single score for classification.
"""

class three_dim_Classifier(nn.Module):

    def __init__(self, num_aa_types=20, num_atom_types=16, num_chain_ids=2, dropout_rate=0.2):
        super(three_dim_Classifier, self).__init__()

        # embeddings
        self.aa_embedding = nn.Embedding(num_aa_types + 1, 16)  # Amino acid embedding (1-20)
        self.atom_embedding = nn.Embedding(num_atom_types + 1, 8)  # Atom type embedding (1-36)
        self.chain_embedding = nn.Embedding(num_chain_ids + 1, 8)  # Chain ID embedding (0,1)

        # The total number of features
        self.feature_dim = 3 + 16 + 8 + 8  # (x, y, z) + Chain ID + AA Type + Atom Type

        # X, Y, and Z feature extraction
        self.conv_xyz_1 = nn.Conv3d(1, 32, kernel_size=(3, 1, self.feature_dim), padding=(1, 0, 0))
        self.conv_xyz_2 = nn.Conv3d(32, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv_xyz_3 = nn.Conv3d(64, 128, kernel_size=(3, 1, 1), padding=(1, 0, 0))

        self.bn_xyz_1 = nn.BatchNorm3d(32)
        self.bn_xyz_2 = nn.BatchNorm3d(64)
        self.bn_xyz_3 = nn.BatchNorm3d(128)

        # Prepare for merge
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        self.bn_fc1 = nn.BatchNorm1d(256)
        self.bn_fc2 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size, num_points, input_dim = x.shape

        xyz = x[:, :, :3]  # Extract (x, y, z)
        chain_id = self.chain_embedding(x[:, :, 3].long())
        aa_type = self.aa_embedding(x[:, :, 4].long())
        atom_type = self.atom_embedding(x[:, :, 5].long())
        combined_features = torch.cat([xyz, chain_id, aa_type, atom_type], dim=2)
        combined_features = combined_features.view(batch_size, 1, num_points, 1, self.feature_dim)

        # 3D Convolution Processing
        combined_features = F.relu(self.bn_xyz_1(self.conv_xyz_1(combined_features)))
        combined_features = F.relu(self.bn_xyz_2(self.conv_xyz_2(combined_features)))
        combined_features = F.relu(self.bn_xyz_3(self.conv_xyz_3(combined_features)))

        # Global Max Pooling
        combined_features = torch.max(combined_features, dim=2)[0] 
        combined_features = combined_features.view(batch_size, -1)

        # Connect
        x = F.relu(self.bn_fc1(self.fc1(combined_features)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)  # binary output

        return torch.sigmoid(x).squeeze(1)  # Probability score (0 to 1)
