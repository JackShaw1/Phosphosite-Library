from Bio.PDB import PDBParser, NeighborSearch
import os
from collections import Counter
import csv

"""

This script extracts data from each phosphoserine and phosphothreonine deposited to the PDB.
Each of these phosphosites is within 6 Angstroms of exactly one foreign protein chain. In the output
file, output_sep_tpo.csv, the pdb id, chain id for phosphorylated protein, residue index of phosphorylated
amino acid (w/ the first amino acid in the chain starting as index=1), the chain id for the lone binding
protein, and binding amino acid index (again starting with index=1 at N terminus) are written across one
line per phosphosite. 

These phosphosites were used to train The Phosphosite Library's PhosNet model. Known stimulatory sites 
from the set were also identified using these data.

Requirement: pdb_structures_SEP_TPO data folder must be present in cwd where this script is executed!

"""

amino_acids = [
    "ALA",  # Alanine
    "ARG",  # Arginine
    "ASN",  # Asparagine
    "ASP",  # Aspartic acid
    "CYS",  # Cysteine
    "GLN",  # Glutamine
    "GLU",  # Glutamic acid
    "GLY",  # Glycine
    "HIS",  # Histidine
    "ILE",  # Isoleucine
    "LEU",  # Leucine
    "LYS",  # Lysine
    "MET",  # Methionine
    "PHE",  # Phenylalanine
    "PRO",  # Proline
    "SER",  # Serine
    "THR",  # Threonine
    "TRP",  # Tryptophan
    "TYR",  # Tyrosine
    "VAL",  # Valine

    # Phosphorylation
    "SEP", # Phosphoserine
    "TPO", # Phosphothreonine
    "PTR", # Phosphotyrosine

    # Methylation
    "MLY",  # Methylated Lysine → Lysine
    "MLZ",  # Dimethylated Lysine → Lysine
    "M3L",  # Trimethylated Lysine → Lysine
    "MRA",  # Methylated Arginine → Arginine

    # Acetylation
    "ALY",  # N-acetylated Lysine → Lysine
    "ACE",  # N-acetylated Serine → Serine

    # Glycosylation
    "NAG",  # N-acetylglucosaminyl-Asparagine → Asparagine
    "GAL",  # O-linked N-acetylgalactosamine (Ser/Thr) → Serine

    # Ubiquitination & SUMOylation
    "UBI",  # Ubiquitinated Lysine → Lysine
    "SUM",  # SUMOylated Lysine → Lysine

    # Hydroxylation
    "HYP",  # Hydroxyproline → Proline
    "HYL",  # Hydroxylysine → Lysine

    # Other PTMs
    "SEC",  # Selenocysteine → Cysteine
    "PYL",  # Pyrrolysine → Lysine
    "CGU",  # Carboxylated Glutamate → Glutamate
    "FME"
]

def get_index(chain, res_index):
    counter = 0
    for residue in chain:
        for atom in residue:
            if atom.get_name() == "CA":
                counter += 1
                if residue.get_id()[1] == res_index:
                    return counter

def analyze_struct(structure, ns):
    total_num_phos_holder = 0
    total_num_inter_holder = 0
    contacts_holder = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in ['SEP', 'TPO']:
                    for atom in residue:
                        if atom.get_name() == 'P':
                            total_num_phos_holder += 1
                            neighbors = ns.search(atom.get_coord(), 6.0)
                            taken = []
                            inter_id = False
                            for atom2 in neighbors:
                                if atom2.get_parent().get_parent().get_id() != chain.get_id() and atom2.get_parent().get_resname() in amino_acids:
                                    inter_id = True
                                if f"{atom2.get_parent().get_parent().get_id()} - {atom2.get_parent().get_id()[1]}" not in taken and atom2.get_parent().get_parent().get_id() != chain.get_id() and atom2.get_parent().get_resname() in amino_acids:
                                    taken.append(f"{atom2.get_parent().get_parent().get_id()} - {atom2.get_parent().get_id()[1]}")
                                    contacts_holder.append(atom2.get_parent().get_resname())
                            chain_neighbor_counter = []
                            inter_distance = 100
                            inter_index = None
                            for atom2 in neighbors:
                                if atom2.get_parent().get_parent().get_id() != chain.get_id() and atom2.get_parent().get_resname() in amino_acids and atom2.get_parent().get_parent().get_id() not in chain_neighbor_counter:
                                    chain_neighbor_counter.append(atom2.get_parent().get_parent().get_id())
                            for atom2 in neighbors:
                                if atom2.get_parent().get_parent().get_id() != chain.get_id() and atom2.get_parent().get_resname() in amino_acids:
                                    if atom2 - atom < inter_distance:
                                        inter_distance = atom2 - atom
                                        inter_index = get_index(atom2.get_parent().get_parent(), atom2.get_parent().get_id()[1])
                            if len(chain_neighbor_counter) == 1:
                                with open('output_sep_tpo.csv', 'a', newline='') as outfile:
                                    writer = csv.writer(outfile)
                                    writer.writerow([structure.get_id(), chain.get_id(), get_index(chain, residue.get_id()[1]), chain_neighbor_counter[0], inter_index])
                            if inter_id == True:
                                total_num_inter_holder += 1
        break
    return [total_num_phos_holder, total_num_inter_holder, contacts_holder]

if __name__ == "__main__":
    counter = 0
    total_num_inter = 0
    total_num_phos = 0
    contacts = []
    parser = PDBParser(QUIET = True)
    print(f'\nTotal number of files to analyze: {len(os.listdir('pdb_structures_SEP_TPO'))}\n')
    file_counter = 0
    for file in os.listdir('pdb_structures_SEP_TPO'):
        print(file_counter)
        file_counter += 1
        structure = parser.get_structure(file.split('.')[0], f"pdb_structures_SEP_TPO/{file}")
        ns = NeighborSearch(list(structure.get_atoms()))
        results = analyze_struct(structure, ns)
        total_num_phos += results[0]
        total_num_inter += results[1]
        contacts += results[2]

    print(f"Total number of p-THR and p-SER: {total_num_phos}")
    print(f"Total number of interfacial p-THR and p-SER: {total_num_inter}")
    print('Frequency of aa contacts:')
    counts = Counter(contacts)
    for string, count in counts.items():
        print(f"{string}: {count}")
    print("\n")
