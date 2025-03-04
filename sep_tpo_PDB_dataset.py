import os
import csv
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from Bio.PDB import PDBParser, NeighborSearch

acceptable_residues = [
    'ARG', 'LYS', 'ASP', 'GLU',
    'ALA', 'CYS', 'GLN', 'GLY', 'HIS', 'ILE', 
    'LEU', 'MET', 'PHE', 'PRO', 'SER', 'THR', 
    'TRP', 'TYR', 'VAL', 'ASN', 'SEP', 'TPO'
]

acceptable_residues_strict = [
    'ARG', 'LYS', 'ASP', 'GLU',
]

acceptable_atoms = ['NZ', 'NE', 'NH1', 'NH2', 'OD1', 'OD2', 'OE1', 'OE2', 'CA']

residue_mapping = {
    'ARG': 0, 'LYS': 1, 'ASP': 2, 'GLU': 3,
    'ALA': 4, 'CYS': 5, 'GLN': 6, 'GLY': 7, 
    'HIS': 8, 'ILE': 9, 'LEU': 10, 'MET': 11, 
    'PHE': 12, 'PRO': 13, 'SER': 14, 'THR': 15, 
    'TRP': 16, 'TYR': 17, 'VAL': 18, 'ASN': 19,
    'SEP': 14, 'TPO': 15
}

atom_mapping = {
    'NZ': 0, 'NE': 0, 'NH1': 0, 'NH2': 0, 'OD1': 1, 'OD2': 1, 'OE1': 1, 'OE2': 1, 'CA': 2
}

# Encode names
def encode_names(phos_id, bind_id, residue_name, atom_name):
    return (0 if phos_id == bind_id else 1), residue_mapping[residue_name], atom_mapping[atom_name]

class PDBDataset(Dataset):
    def __init__(self, true_pdb_folder, false_pdb_folder, true_csv, false_csv, transform=None):
        self.true_pdb_folder = true_pdb_folder
        self.false_pdb_folder = false_pdb_folder
        self.transform = transform
        self.samples = []
        self._load_csv(true_csv, label=1, pdb_folder=true_pdb_folder)
        self._load_csv(false_csv, label=0, pdb_folder=false_pdb_folder)

    def _load_csv(self, csv_path, label, pdb_folder):
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header if present
            for row in reader:
                filename, chain, residue_index = row[0], row[1], int(row[2])
                self.samples.append((filename, chain, residue_index, label, pdb_folder))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pdb_code, chain_id, residue_index, label, pdb_folder = self.samples[idx]
        pdb_path = os.path.join(pdb_folder, pdb_code)
        parser = PDBParser(QUIET=True)
        if label == 1:
            structure = parser.get_structure(pdb_code, f"{pdb_path}.pdb")
        if label == 0:
            structure = parser.get_structure(pdb_code, f"{pdb_path}")

        focal_atom = None
        for model in structure:
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
            break

        if focal_atom is None:
            return None

        # Neighbors
        neighbor_search = NeighborSearch(list(structure.get_atoms()))
        neighbors_wide = neighbor_search.search(focal_atom.coord, 12)
        neighbors_narrow = neighbor_search.search(focal_atom.coord, 6)

        point_cloud = []

        for atom in neighbors_wide:

            # Handle side chain atoms
            if atom.get_parent().get_resname() in acceptable_residues_strict and \
                atom.get_name() in acceptable_atoms and atom.get_name() != 'CA' \
                    and atom.get_parent().get_parent().get_id() != chain_id:
                    res_name = atom.get_parent().get_resname()
                    atom_name = atom.get_name()
                    coord = (atom.coord - focal_atom.coord) / 12  # normalize coordinates
                    chain_encoded, res_encoded, atom_encoded = encode_names(chain.get_id(), atom.get_parent().get_parent().get_id(), res_name, atom_name)
                    point_cloud.append(np.concatenate((coord, [chain_encoded, res_encoded, atom_encoded])))

        for atom in neighbors_narrow:

            # Handle side chain atoms
            if atom.get_parent().get_resname() in acceptable_residues_strict and \
                atom.get_name() in acceptable_atoms and atom.get_name() != 'CA' \
                    and atom.get_parent().get_parent().get_id() == chain_id:
                        res_name = atom.get_parent().get_resname()
                        atom_name = atom.get_name()
                        coord = (atom.coord - focal_atom.coord) / 12  # normalize coordinates
                        chain_encoded, res_encoded, atom_encoded = encode_names(chain.get_id(), atom.get_parent().get_parent().get_id(), res_name, atom_name)
                        point_cloud.append(np.concatenate((coord, [chain_encoded, res_encoded, atom_encoded])))

        # for model in structure:
        #     for chain in model:
        #         if chain.get_id() == chain_id:
        #             for residue in chain:
        #                 if residue.get_resname() in acceptable_residues and \
        #                     abs(residue.get_id()[1] - int(residue_index)) <= 10:
        #                         for atom in residue:
        #                             if atom.get_name() == 'CA':
        #                                 res_name = residue.get_resname()
        #                                 atom_name = atom.get_name()
        #                                 coord = (atom.coord - focal_atom.coord) / 12  # normalize coordinates
        #                                 chain_encoded, res_encoded, atom_encoded = encode_names(chain.get_id(), atom.get_parent().get_parent().get_id(), res_name, atom_name)
        #                                 point_cloud.append(np.concatenate((coord, [chain_encoded, res_encoded, atom_encoded])))
        #     break

        if not point_cloud:
            return None
        
        point_cloud.sort(key=lambda x: np.linalg.norm(x[:3]))  # Sort by Euclidean distance

        if len(point_cloud) > 20:
            point_cloud = point_cloud[:20]

        sample = {'coordinates': np.array(point_cloud, dtype=np.float32), 'label': label, 'filename': pdb_code}
        if self.transform:
            sample = self.transform(sample)

        return sample
