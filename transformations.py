import torch
import numpy as np
import random

class PointCloudTransform:
    
    def __init__(self, rotation=True, wiggle=True, wiggle_factor=0.15, reflection=True, num_rotations=3):
        self.rotation = rotation
        self.wiggle = wiggle
        self.wiggle_factor = wiggle_factor
        self.reflection = reflection
        self.num_rotations = num_rotations  # Number of different rotations + wiggles per sample

    def apply_rotation_and_wiggle(self, coordinates):
        # Apply random rotation
        theta_x, theta_y, theta_z = random.uniform(0, 2 * np.pi), random.uniform(0, 2 * np.pi), random.uniform(0, 2 * np.pi)
        rotation_x = torch.tensor([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]], dtype=torch.float32)
        rotation_y = torch.tensor([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]], dtype=torch.float32)
        rotation_z = torch.tensor([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]], dtype=torch.float32)

        rotation_matrix = torch.matmul(rotation_x, torch.matmul(rotation_y, rotation_z))
        coordinates[:, :3] = torch.matmul(coordinates[:, :3], rotation_matrix.T)

        # Apply small random wiggle
        if self.wiggle:
            wiggle_offsets = torch.randn_like(coordinates[:, :3]) * self.wiggle_factor
            coordinates[:, :3] += wiggle_offsets

        return coordinates

    def __call__(self, sample):
        original_coords = torch.tensor(sample['coordinates'], dtype=torch.float32)

        # Randomly select one of the transformations
        transformation_choice = random.randint(0, self.num_rotations + 3)  # +3 for reflections

        if transformation_choice == 0:
            transformed_coords = original_coords.clone()  # Keep original
        elif 1 <= transformation_choice <= self.num_rotations:
            transformed_coords = self.apply_rotation_and_wiggle(original_coords.clone())  # Apply rotation + wiggle
        elif transformation_choice == self.num_rotations + 1:
            transformed_coords = original_coords.clone()
            transformed_coords[:, 0] *= -1  # Reflect across X-axis
        elif transformation_choice == self.num_rotations + 2:
            transformed_coords = original_coords.clone()
            transformed_coords[:, 1] *= -1  # Reflect across Y-axis
        else:
            transformed_coords = original_coords.clone()
            transformed_coords[:, 2] *= -1  # Reflect across Z-axis

        return {  
            'coordinates': transformed_coords.numpy(),
            'label': sample['label'],  # Preserve label
            'filename': sample['filename']  # Preserve filename
        }
