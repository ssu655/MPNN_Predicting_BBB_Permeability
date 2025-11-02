import os
import pandas as pd
from rdkit import Chem
import torch_geometric
from torch_geometric.data import Data
import torch
from torch.utils.data import Dataset
from collections import Counter
import numpy as np

class MoleculeGraphProcessor:
    def __init__(self, folder_path):
        """
        Initializes the processor with the provided folder of CSV files.
        Reads the data from all CSV files and stores the SMILES strings.

        Parameters:
            folder_path (str): Path to the folder containing the CSV files.
        """
        self.folder_path = folder_path
        self.load_data()

    def load_data(self):
        """
        Loads assay data from all CSV files in the provided folder.
        Extracts SMILES strings and their associated logP (Standard Value) values.
        """
        self.smiles_list = []
        self.logp_list = []

        for file_name in os.listdir(self.folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(self.folder_path, file_name)
                data = pd.read_csv(file_path)

                # --- Identify SMILES column (case-insensitive search) ---
                smiles_col = None
                for col in data.columns:
                    if 'smiles' in col.lower():   # match variations like "canonical_smiles" or "Smiles"
                        smiles_col = col
                        break

                if smiles_col is None:
                    print(f"Skipping {file_name}: No SMILES-like column found.")
                    continue

                # --- Check if 'Standard Value' column exists (for logP) ---
                if 'Permeability' not in data.columns:
                    print(f"Skipping {file_name}: 'Permeability' column not found.")
                    continue

                # --- Drop missing SMILES or logP values ---
                filtered_data = data.dropna(subset=[smiles_col, 'Permeability'])

                # --- Append extracted values ---
                self.smiles_list.extend(filtered_data[smiles_col].tolist())

                # Convert 'Standard Value' to floats, invalid entries become NaN
                logp_values = pd.to_numeric(filtered_data['Permeability'], errors='coerce')
                # Keep all entries, including NaN
                self.logp_list.extend(logp_values.tolist())

        print(f"Loaded {len(self.smiles_list)} SMILES with Permeability values from {self.folder_path}")

    def smiles_to_mol(self, smiles):
        """
        Converts a SMILES string to an RDKit molecule object.
        
        Parameters:
            smiles (str): The SMILES string of the compound.

        Returns:
            rdkit.Chem.Mol: The RDKit molecule object.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        return mol

    def mol_to_graph(self, mol):
        """
        Converts an RDKit molecule to a molecular graph suitable for MPNN.

        Parameters:
            mol (rdkit.Chem.Mol): The RDKit molecule object.

        Returns:
            torch_geometric.data.Data: A molecular graph in PyTorch Geometric format.
        """
        # === Atom-level features ===
        atom_features = []
        hyb_map = {
                Chem.rdchem.HybridizationType.SP: 0,
                Chem.rdchem.HybridizationType.SP2: 1,
                Chem.rdchem.HybridizationType.SP3: 2,
                Chem.rdchem.HybridizationType.SP3D: 3,
                Chem.rdchem.HybridizationType.SP3D2: 4
            }
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            degree = atom.GetDegree()
            formal_charge = atom.GetFormalCharge()
            aromatic = 1 if atom.GetIsAromatic() else 0
            NumH = atom.GetTotalNumHs()
            total_valence = atom.GetTotalValence()

            # Map hybridization to integer encoding
            hyb = atom.GetHybridization()
            hyb_idx = hyb_map.get(hyb, -1)

            atom_features.append([
                atomic_num,
                degree,
                hyb_idx,
                formal_charge,
                aromatic,
                NumH,
                total_valence
            ])

        x = torch.tensor(atom_features, dtype=torch.float)

        # === Bond-level features ===
        bond_indices = []
        bond_features = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

            # Undirected → add both directions, so message passing happens both ways, important for molecular graph
            bond_indices += [[i, j], [j, i]]

            edge_feat = [
                bond.GetBondTypeAsDouble(),
                bond.GetIsConjugated(),
                bond.IsInRing(),
                bond.GetIsAromatic()
            ]
            bond_features += [edge_feat, edge_feat]

        # Convert to PyTorch tensors
        edge_index = torch.tensor(bond_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(bond_features, dtype=torch.float)

        # Create and return the Data object
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return graph

    def process_molecules(self):
        """
        Processes all molecules from the loaded SMILES list, converts them to graphs,
        and stores them in a list.

        save the targets to be predicted in a dictionary called target. Each property
        is an item in dict.
        """
        target = {}
        target["logP"] = self.logp_list

        molecule_graphs = []
        for smiles in self.smiles_list:
            mol = self.smiles_to_mol(smiles)
            graph = self.mol_to_graph(mol)
            molecule_graphs.append(graph)
        return molecule_graphs, target

class MPNNDataset(Dataset):
    """
    Dataset for MPNN models containing molecular graphs and their associated
    target properties (e.g., logP, permeability).

    Parameters:
        graphs (list): List of torch_geometric.data.Data objects.
        targets (list): List or tensor of property values corresponding to each graph.
    """
    def __init__(self, graphs, targets=None):
        if targets is not None:
            # check that all target lists match the number of graphs
            for key, vals in targets.items():
                assert len(vals) == len(graphs), f"Target '{key}' length mismatch with graphs."
        self.graphs = graphs
        self.targets = targets  

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def save(self, path):
        """Save list of PyG Data objects, each with its own .y attribute."""
        if self.targets is not None:
            for i, graph in enumerate(self.graphs):
                y = torch.tensor(self.targets['logP'][i], dtype=torch.float).view(1)
                graph.y = y
        torch.save(self.graphs, path)

    @classmethod
    def load(cls, path):
        graphs = torch.load(path, weights_only=False)
        return cls(graphs)

    def summary(self):
        """Prints dataset statistics for atoms, bonds, bond types, and targets."""
        num_mols = len(self.graphs)
        atom_counts, bond_counts, bond_types, targets = [], [], [], []

        for graph in self.graphs:
            atom_counts.append(graph.num_nodes)
            bond_counts.append(graph.num_edges // 2)

            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                bt = graph.edge_attr[:, 0].tolist() if graph.edge_attr.ndim > 1 else graph.edge_attr.tolist()
                bond_types.extend(bt)

            if hasattr(graph, 'y'):
                targets.append(graph.y.item() if graph.y.numel() == 1 else graph.y.mean().item())

        print("📊 Dataset Summary")
        print("-------------------------------")
        print(f"Total molecules: {num_mols}")
        print(f"Atoms per molecule: mean={np.mean(atom_counts):.2f}, min={np.min(atom_counts)}, max={np.max(atom_counts)}")
        print(f"Bonds per molecule: mean={np.mean(bond_counts):.2f}, min={np.min(bond_counts)}, max={np.max(bond_counts)}")

        bond_type_names = {1: "single", 2: "double", 3: "triple", 4: "quadruple", 12: "aromatic"}
        if bond_types:
            counts = Counter(bond_types)
            print("\nBond type distribution:")
            for btype, count in counts.items():
                name = bond_type_names.get(int(btype), str(btype))
                print(f"  Type {btype} ({name}): {count}")
        else:
            print("\nNo bond type information found.")

        if targets:
            arr = np.array(targets)
            print("\nTarget permeability statistics:")
            print(f"  mean={np.mean(arr):.3f}, min={np.min(arr):.3f}, max={np.max(arr):.3f}")
        else:
            print("\nNo target values found.")

if __name__ == "__main__":
    # Folder path relative to the script
    folder_path = "PAMPAassays"  # This folder should be in the same directory as the script
    
    # Initialize the processor and process the molecules
    processor = MoleculeGraphProcessor(folder_path)
    molecule_graphs, targets = processor.process_molecules()

    # Create an MPNN dataset and optionally save
    dataset = MPNNDataset(molecule_graphs, targets)
    dataset.save("PAMPA_dataset.pt")

    print(f"Processed {len(molecule_graphs)} molecules into molecular graphs, and saved PAMPA permiability metric.\n")
    dataset.summary()