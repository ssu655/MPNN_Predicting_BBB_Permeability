import torch
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from DataLoading import MPNNDataset
import umap.umap_ as umap
import random

class MoleculeClusterAnalysis:
    def __init__(self, model, dataset, batch_size=32, device='cpu'):
        """
        Parameters:
            model: Trained MPNN model with forward supporting return_embedding=True
            dataset: MPNNDataset
            batch_size: DataLoader batch size
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size

        self.embeddings = None
        self.tsne_2d = None
        self.umap_2d = None
        self.values = None

    def extract_embeddings(self):
        """Run dataset through the model to get graph embeddings."""
        self.model.eval()

        n = len(self.dataset)
        indices = list(range(n))

        # Fixed test split seed
        random.seed(42)
        random.shuffle(indices)

        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        idx = indices[val_end:]

        loader = DataLoader([self.dataset[i] for i in idx], batch_size=self.batch_size, shuffle=False)

        self.values = np.array([self.dataset.graphs[i].y.item() for i in idx])

        # loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        all_emb = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                emb = self.model(batch, return_embedding=True)  # returns graph-level embedding
                all_emb.append(emb.cpu())

        self.embeddings = torch.cat(all_emb, dim=0)
        return self.embeddings, self.values

    def compute_tsne(self, n_components=2, perplexity=30, random_state=42):
        """Compute 2D t-SNE projection of the graph embeddings."""
        if self.embeddings is None:
            raise ValueError("Call extract_embeddings() first.")

        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        self.tsne_2d = tsne.fit_transform(self.embeddings.numpy())
        return self.tsne_2d

    def compute_umap(self, n_components=2, random_state=42):
        """ Compute 2D UMAP projection of molecule embeddings. """
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        self.umap_2d = reducer.fit_transform(self.embeddings)
        return self.umap_2d


    def plot_embeddings(self, color_by=None, figsize=(12,6), title="Molecule Embeddings"):
        """
        Plot both t-SNE and UMAP in one figure with 2 subplots.
        
        Parameters:
            color_by (str or array-like): Optional target values for coloring points.
            figsize (tuple): Figure size.
            title (str): Overall figure title.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for ax, emb, name in zip(
            axes, 
            [self.tsne_2d, self.umap_2d],
            ["t-SNE", "UMAP"]):

            if color_by is not None:
                sc = ax.scatter(emb[:,0], emb[:,1], c=self.values, cmap='viridis', s=20)
                fig.colorbar(sc, ax=ax, label=color_by)
            else:
                ax.scatter(emb[:,0], emb[:,1], s=20)

            ax.set_title(name)
            ax.set_xlabel(f"{name} 1")
            ax.set_ylabel(f"{name} 2")

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":

    dataset = MPNNDataset.load("PAMPA_dataset.pt")

    # Load the full model object
    model = torch.load("mpnn_model_full_trained.pt",weights_only=False)
    
    # Suppose `model` is trained and dataset is MPNNDataset
    cluster_analysis = MoleculeClusterAnalysis(model, dataset, device='cpu')

    # Extract embeddings
    embeddings = cluster_analysis.extract_embeddings()

    # Compute 2D projection
    cluster_analysis.compute_tsne()
    cluster_analysis.compute_umap()

    # Plot t-SNE colored by Permeability
    cluster_analysis.plot_embeddings(color_by='Permeability')