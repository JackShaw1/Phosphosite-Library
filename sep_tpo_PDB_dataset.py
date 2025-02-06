import os
import csv
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from Bio.PDB import PDBParser, NeighborSearch

acceptible_residues = [
    'ARG', 'ASP', 'GLU', 'LYS', 'SEP', 'SER', 'TPO', 'THR'
]

acceptible_residues_self = [
    'ARG', 'ASP', 'GLU', 'LYS', 'ASN', 'CYS', 'GLN', 'GLY', 'HIS', 
    'ILE', 'LEU', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 
    'TYR', 'VAL'
]

acceptible_atoms_self = ['NE', 'NH1', 'NH2', 'OD1', 'OD2', 'ND2', 
                         'OE1', 'OE2', 'NE1', 'NE2', 'ND1', 
                         'NZ', 'OG', 'OG1', 'OH', 'CA']

acceptible_atoms = ['NE', 'NH1', 'NH2', 'OD1', 'OD2', 'ND2', 
                    'OE1', 'OE2', 'NE1', 'NE2', 'ND1', 
                    'NZ', 'OG', 'OG1', 'OH']

atom_mapping = {name: idx for idx, name in enumerate(acceptible_atoms)}

residue_mapping = {name: idx for idx, name in enumerate(acceptible_residues_self)}

# encode names
def encode_names(phos_id, bind_id, residue_name, atom_name):
    if residue_name not in ['SEP', 'PTR', 'TPO']:
        return (0 if phos_id == bind_id else 1), residue_mapping.get(residue_name, 0), atom_mapping.get(atom_name, 0)
    return (0 if phos_id == bind_id else 1), residue_mapping.get({'SEP': 'SER', 'TPO': 'THR', 'PTR': 'TYR'}[residue_name], 0), atom_mapping.get(atom_name, 0)

class PDBDataset(Dataset):
    def __init__(self, pdb_folder, true_csv, false_csv, radius=10.0, transform=None):
        self.pdb_folder = pdb_folder
        self.transform = transform
        self.radius = radius
        self.samples = []
        self._load_csv(true_csv, label=1)
        self._load_csv(false_csv, label=0)

    def _load_csv(self, csv_path, label):
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header if present
            for row in reader:
                filename, chain, residue_index = row[0], row[1], int(row[2])
                self.samples.append((filename, chain, residue_index, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pdb_code, chain_id, residue_index, label = self.samples[idx]
        pdb_path = os.path.join(self.pdb_folder, pdb_code)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_code, f"{pdb_path}.pdb")

        focal_atom = None
        model = structure[0] 
        for chain in model:
            if chain.get_id() == chain_id:
                for residue in chain:
                    if residue.get_id()[1] == int(residue_index):
                        for atom in residue:
                            if residue.get_resname() in ['SEP', 'SER'] and atom.get_name() == 'OG':
                                focal_atom = atom
                            if residue.get_resname() in ['TPO', 'THR'] and atom.get_name() == 'OG1':
                                focal_atom = atom
                            if residue.get_resname() in ['PTR', 'TYR'] and atom.get_name() == 'OH':
                                focal_atom = atom

        if focal_atom is None:
            return None

        # Neighbors
        neighbor_search = NeighborSearch(list(structure.get_atoms()))
        neighbors = neighbor_search.search(focal_atom.coord, self.radius)
        point_cloud = []

        for atom in neighbors:
            if atom.get_parent().get_resname() in acceptible_residues and atom.get_name() in acceptible_atoms:
                if atom.get_parent().get_resname() in ['SEP', 'SER', 'TPO', 'THR'] and atom.get_parent().get_id()[1] != int(residue_index):
                    continue
                res_name = atom.get_parent().get_resname()
                if res_name == 'SEP':
                    res_name = 'SER'
                if res_name == 'TPO':
                    res_name = 'THR'
                if res_name == 'PTR':
                    res_name = 'TYR'
                atom_name = atom.get_name()
                coord = (atom.coord - focal_atom.coord) / 10  # normalize coordinates
                chain_encoded, res_encoded, atom_encoded = encode_names(chain.get_id(), atom.get_parent().get_parent().get_id(), res_name, atom_name)
                point_cloud.append(np.concatenate((coord, [chain_encoded, res_encoded, atom_encoded])))

        if not point_cloud:
            return None

        sample = {'coordinates': np.array(point_cloud, dtype=np.float32), 'label': label, 'filename': pdb_code}
        if self.transform:
            sample = self.transform(sample)

        return sample
