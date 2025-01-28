from Bio.PDB import PDBParser, NeighborSearch
import os
from collections import Counter

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
    "VAL"   # Valine
]

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
                                if f"{atom2.get_parent().get_parent().get_id()} - {atom2.get_parent().get_id()[1]}" not in taken and atom2.get_parent().get_parent().get_id() != chain.get_id() and atom2.get_name() not in ['N', 'CA', 'C', 'O'] and atom2.get_parent().get_resname() in amino_acids:
                                    taken.append(f"{atom2.get_parent().get_parent().get_id()} - {atom2.get_parent().get_id()[1]}")
                                    contacts_holder.append(atom2.get_parent().get_resname())
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
    for file in os.listdir('pdb_structures_SEP_TPO'):
        structure = parser.get_structure('struct', f"pdb_structures_sep_tpo/{file}")
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
