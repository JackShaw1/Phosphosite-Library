import torch
import numpy as np
from Bio.PDB import PDBParser, NeighborSearch
from pseudo_pointnet import PointNetBinaryClassifier  # Import trained model
import argparse
import os
import csv
import lzma
import shutil

def decompress_xz(input_path, output_path):
    try:
        with lzma.open(input_path, 'rb') as compressed_file:
            with open(output_path, 'wb') as decompressed_file:
                # Copy decompressed content to the output file
                shutil.copyfileobj(compressed_file, decompressed_file)
        print(f"Decompressed file saved to: {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

MODEL_PATH = "experimental_PhosNet.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PointNetBinaryClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval() 
model.to(device)

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

# Encode residue and atom names
def encode_names(phos_id, bind_id, residue_name, atom_name):
    return (0 if phos_id == bind_id else 1), residue_mapping.get(residue_name, 0), atom_mapping.get(atom_name, 0)

# Generate point cloud
def extract_residue_point_cloud(pdb_path, chain_id, residue_index, radius=10.0):
    """Extracts atomic coordinates and feature encodings for a given residue."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("input_pdb", pdb_path)
    
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
        raise ValueError(f"⚠️ Residue {residue_index} in chain {chain_id} not found or does not have the expected focal atom.")

    neighbor_search = NeighborSearch(list(structure.get_atoms()))
    neighbors = neighbor_search.search(focal_atom.coord, radius)

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
        raise ValueError(f"⚠️ No neighbors found within {radius} Å for residue {residue_index} in chain {chain_id}.")

    return np.array(point_cloud, dtype=np.float32)

def test_residue(pdb_path, chain_id, residue_index):
    """Processes a residue and predicts its score using the trained model."""
    try:
        point_cloud = extract_residue_point_cloud(pdb_path, chain_id, residue_index)
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dim

        with torch.no_grad():
            score = model(point_cloud).item()  # Get single output score

        print(f"Score for {os.path.basename(pdb_path)}, Chain {chain_id}, Residue {residue_index}: {score:.4f}")
        return score

    except ValueError as e:
        print(f"{e}")
        return None

if __name__ == "__main__":

    # # Optional command line
    # parser = argparse.ArgumentParser(description="Score a specific residue in a PDB file using a trained PointNet model.")
    # parser.add_argument("pdb_path", type=str, help="Path to the PDB file")
    # parser.add_argument("chain_id", type=str, help="Chain ID (e.g., 'A')")
    # parser.add_argument("residue_index", type=int, help="Residue index (e.g., 100)")

    # args = parser.parse_args()
    # predict_residue(args.pdb_path, args.chain_id, args.residue_index)

    predictome = os.listdir('high_spoc_predictome')

    with open('giga_upregulator_oct_5_no_histidines.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[4][0] in ['T', 'S']:
                parts = row[0].split(' / ')
                model_str = f'model_{row[2][0]}'
                print(model_str)
                for filename in predictome:
                    if parts[0] in filename and parts[1] in filename and model_str in filename and '.pdb' in filename and parts[0] != parts[1] or parts[0] in filename and parts[1] in filename and model_str in filename and '.pdb' in filename and filename.count(parts[0]) == 2:
                        output_path = f"tmp/{filename[:-3]}"
                        input_path = f"high_spoc_predictome/{filename}"
                        decompress_xz(input_path, output_path)
                        score = test_residue(output_path, row[3], int(row[4][1:]))
                        os.remove(output_path)
                        with open('giga_upregulator_oct_5_w_sep_tpo_pseudo_pointnet_phosnet.csv', 'a', newline='') as outfile:
                            writer = csv.writer(outfile)
                            writer.writerow(row[:11] + [score])
