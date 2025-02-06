import torch
import numpy as np
import random

class PointCloudTransform:
    
    def __init__(self, rotation=True, wiggle=True, wiggle_factor=0.15):
        self.rotation = rotation
        self.wiggle = wiggle
        self.wiggle_factor = wiggle_factor

    def __call__(self, sample):
        coordinates = torch.tensor(sample['coordinates'], dtype=torch.float32)

        # Apply random rotation
        if self.rotation:
            theta_x, theta_y, theta_z = random.uniform(0, 2 * np.pi), random.uniform(0, 2 * np.pi), random.uniform(0, 2 * np.pi)
            rotation_x = torch.tensor([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]], dtype=torch.float32)
            rotation_y = torch.tensor([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]], dtype=torch.float32)
            rotation_z = torch.tensor([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]], dtype=torch.float32)

            rotation_matrix = torch.matmul(rotation_x, torch.matmul(rotation_y, rotation_z))
            coordinates[:, :3] = torch.matmul(coordinates[:, :3], rotation_matrix.T)

        if self.wiggle:
            wiggle_offsets = torch.randn_like(coordinates[:, :3]) * self.wiggle_factor
            coordinates[:, :3] += wiggle_offsets

        sample['coordinates'] = coordinates.numpy()
        return sample
