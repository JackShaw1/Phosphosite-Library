import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import csv
from sep_tpo_PDB_dataset import PDBDataset
from transformations import PointCloudTransform
from pseudo_pointnet import PointNetBinaryClassifier

pdb_folder = 'pdb_structures_sep_tpo'
true_csv = 'norep_positives_ungrouped_fold_job_jan_25.csv'
false_csv = 'sampled_negatives_jan_25.csv'

# params
epochs = 4
batch_size = 1
learning_rate = 0.0001

# Create dataset and split into train/test
full_dataset = PDBDataset(pdb_folder, true_csv, false_csv)
indices = list(range(len(full_dataset)))
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

train_transform = PointCloudTransform(rotation=True, wiggle=True)
train_dataset = Subset(PDBDataset(pdb_folder, true_csv, false_csv, transform=train_transform), train_indices)
test_dataset = Subset(PDBDataset(pdb_folder, true_csv, false_csv), test_indices)

def custom_collate(batch):
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        raise ValueError("All samples in the batch are invalid.")
    max_len = max(len(sample['coordinates']) for sample in batch)
    
    padded_coords = []
    labels = []
    filenames = []
    
    for sample in batch:
        coords = sample['coordinates']
        label = sample['label']
        coord_padding = np.zeros((max_len - len(coords), coords.shape[1]))
        padded_coords.append(torch.tensor(np.vstack((coords, coord_padding)), dtype=torch.float32))
        labels.append(torch.tensor(label, dtype=torch.float32))  # 🔹 Change label to float for BCEWithLogitsLoss
        filenames.append(sample['filename'])
    
    coords_batch = torch.stack(padded_coords)
    labels_batch = torch.stack(labels)
    
    return {'coordinates': coords_batch, 'labels': labels_batch, 'filename': filenames}

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

# get model
model = PointNetBinaryClassifier()

# Binary Cross-Entropy Loss
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with tqdm
loss_values = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
    
    for batch in progress_bar:
        coords = batch['coordinates']
        labels = batch['labels']
        
        batch_size, max_len, input_dim = coords.shape
        coords = coords.view(batch_size, max_len, input_dim)
        outputs = model(coords)

        # optimizer
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    
    avg_epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}")
    loss_values.append(avg_epoch_loss)

torch.save(model.state_dict(), 'experimental_PhosNet.pth')

# Generate loss figure
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid(True)
plt.savefig('training_loss_plot.png')
plt.show()

print("Training complete. Model saved as 'experimental_pseudo_PointNet_PhosNet.pth'")
