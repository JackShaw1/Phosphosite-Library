import torch
import numpy as np
import os
from Bio.PDB import PDBParser, NeighborSearch
from PhosNet_model import PhosNetBinaryClassifier
from sep_tpo_PDB_dataset import encode_names, acceptable_residues_strict, acceptable_atoms
import csv
import lzma
import shutil

# Load trained model
model_path = 'experimental_PhosNet.pth'
model = PhosNetBinaryClassifier()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # Set model to evaluation mode

def decompress_xz(input_path, output_path):
    try:
        with lzma.open(input_path, 'rb') as compressed_file:
            with open(output_path, 'wb') as decompressed_file:
                # Copy decompressed content to the output file
                shutil.copyfileobj(compressed_file, decompressed_file)
        print(f"Decompressed file saved to: {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Function to process residue
def process_residue(pdb_file, chain_id, residue_index):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("test", pdb_file)

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
        print(f"No valid focal atom found in {pdb_file} at {chain_id}{residue_index}")
        return None

    neighbor_search = NeighborSearch(list(structure.get_atoms()))
    neighbors_wide = neighbor_search.search(focal_atom.coord, 12)
    neighbors_narrow = neighbor_search.search(focal_atom.coord, 6)

    point_cloud = []

    for atom in neighbors_wide:
        if atom.get_parent().get_resname() in acceptable_residues_strict and atom.get_name() in acceptable_atoms and atom.get_name() != 'CA':
            res_name = atom.get_parent().get_resname()
            atom_name = atom.get_name()
            coord = (atom.coord - focal_atom.coord) / 12  # Normalize coordinates
            chain_encoded, res_encoded, atom_encoded = encode_names(chain.get_id(), atom.get_parent().get_parent().get_id(), res_name, atom_name)
            point_cloud.append(np.concatenate((coord, [chain_encoded, res_encoded, atom_encoded])))

    for atom in neighbors_narrow:
        if atom.get_parent().get_resname() in acceptable_residues_strict and atom.get_name() in acceptable_atoms and atom.get_name() != 'CA':
            res_name = atom.get_parent().get_resname()
            atom_name = atom.get_name()
            coord = (atom.coord - focal_atom.coord) / 12
            chain_encoded, res_encoded, atom_encoded = encode_names(chain.get_id(), atom.get_parent().get_parent().get_id(), res_name, atom_name)
            point_cloud.append(np.concatenate((coord, [chain_encoded, res_encoded, atom_encoded])))

    if not point_cloud:
        print(f"No valid point cloud found for {pdb_file} at {chain_id}{residue_index}")
        return None

    # Pad to 35 points
    max_len = 20
    if len(point_cloud) > max_len:
        point_cloud = point_cloud[:max_len]
    else:
        pad = np.zeros((max_len - len(point_cloud), len(point_cloud[0])))
        point_cloud = np.vstack((point_cloud, pad))

    return torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# Function to test a residue
def test_residue(pdb_file, chain_id, residue_index):
    print(f"Testing residue {chain_id}{residue_index} in {pdb_file}")
    input_data = process_residue(pdb_file, chain_id, residue_index)
    
    if input_data is None:
        return

    with torch.no_grad():
        output = model(input_data)
        probability = output.item()
        prediction = "Regulatory" if probability > 0.5 else "Non-regulatory"

    print(f"Prediction: {prediction} (Confidence: {probability:.4f})")

    return probability

# Example usage
if __name__ == "__main__":


    # pdb_file = "CCNA2_HUMAN__CDK2_HUMAN__730aa_unrelaxed_af2mv3_model_1 (4).pdb"  # Replace with your .pdb file
    # chain_id = "B"  # Replace with your target chain
    # residue_index = 41  # Replace with your target residue number
    # test_residue(pdb_file, chain_id, residue_index)


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
