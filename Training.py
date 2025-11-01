import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import random
import matplotlib.pyplot as plt
from DataLoading import MPNNDataset
from MPNN import MPNNModel

class MPNNTrainer:
    def __init__(self, model, dataset, epochs=20, batch_size=32, lr=1e-3, Huber_beta = 1, toy_dataset=False):
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.toy_dataset = toy_dataset
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)

        # Loss & optimizer
        self.criterion = nn.SmoothL1Loss(beta=Huber_beta)  # Huber loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        # Prepare data
        self._prepare_data()

        # unified histories dictionary
        self.histories = {
            "train_loss": [],
            "val_loss": [],
            "val_mse": [],
            "val_rmse": [],
            "val_r2": []
        }

    def _prepare_data(self):
        """Split dataset into train/val/test."""
        n = len(self.dataset)
        indices = list(range(n))

        # Fixed test split seed
        random.seed(42)
        random.shuffle(indices)

        if self.toy_dataset:
            train_idx = indices[:100]
            val_idx = indices[100:110]
            test_idx = indices[110:120]
        else:
            train_end = int(0.7 * n)
            val_end = int(0.85 * n)
            train_idx = indices[:train_end]
            val_idx = indices[train_end:val_end]
            test_idx = indices[val_end:]

        self.train_loader = DataLoader([self.dataset[i] for i in train_idx], batch_size=self.batch_size, shuffle=True)
        self.val_loader   = DataLoader([self.dataset[i] for i in val_idx],   batch_size=self.batch_size, shuffle=False)
        self.test_loader  = DataLoader([self.dataset[i] for i in test_idx],  batch_size=self.batch_size, shuffle=False)

    def _evaluate(self, loader):
        """Compute MSE, RMSE, R² metrics."""
        self.model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch).squeeze()
                target = batch.y.float().to(self.device)
                preds.append(out.cpu())
                trues.append(target.cpu())

        preds = torch.cat([p for p in preds], dim=0)
        trues = torch.cat([t for t in trues], dim=0)

        # Huber loss for validation
        huber_loss = self.criterion(preds, trues)

        mse = mean_squared_error(trues, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(trues, preds)
        return huber_loss, mse, rmse, r2

    def train(self, early_stopping_patience=30):
        """Full training loop with early stopping and exponentially decaying learning rate."""
        best_val_loss = float('inf')
        epochs_no_improve = 0

        self.scheduler.step()

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0

            for batch in self.train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                out = self.model(batch).squeeze()
                target = batch.y.float().to(self.device)
                loss = self.criterion(out, target)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * batch.num_graphs

            train_loss /= len(self.train_loader.dataset)

            # Evaluate
            val_loss, val_mse, val_rmse, val_r2 = self._evaluate(self.val_loader)

            # Update history dictionary
            self.histories["train_loss"].append(train_loss)
            self.histories["val_loss"].append(val_loss)
            self.histories["val_mse"].append(val_mse)
            self.histories["val_rmse"].append(val_rmse)
            self.histories["val_r2"].append(val_r2)

            print(f"Epoch {epoch:03d} | "
                  f"Train Huber={train_loss:10.4f} | "
                  f"Val Huber={val_loss:10.4f} | "
                  f"Val MSE={val_mse:10.4f} | "
                  f"Val RMSE={val_rmse:10.4f} | "
                  f"Val R²={val_r2:10.4f}")
        
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # optionally save the best model here
                self.best_model_state = self.model.state_dict()
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Val Huber Loss={best_val_loss:.4f}")
                break

        # At the end, you can load the best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)

    def test(self):
        """Evaluate on held-out test set (never used during training)."""
        loss, mse, rmse, r2 = self._evaluate(self.test_loader)
        print(f"\nFinal Results on held-out test set — Loss={loss:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
        return loss, mse, rmse, r2
    
    def plot_training_history(self):
        """Plot training/validation loss and validation metrics."""
        if not self.histories:
            print("No training history found.")
            return

        epochs = range(1, len(self.histories["train_loss"]) + 1)

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # -----------------------
        # Subplot 1: Train vs Val Huber loss
        # -----------------------
        axs[0].plot(epochs, self.histories["train_loss"], label="Train Huber Loss", marker='o')
        axs[0].plot(epochs, self.histories["val_loss"], label="Val Huber Loss", marker='x')
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Huber Loss")
        axs[0].set_title("Train vs Validation Huber Loss")
        axs[0].legend()
        axs[0].grid(True)

        # -----------------------
        # Subplot 2: Val loss + metrics
        # -----------------------
        axs[1].plot(epochs, self.histories["val_loss"], label="Val Huber Loss", marker='x')
        # axs[1].plot(epochs, self.histories["val_mse"], label="Val MSE", marker='o')
        axs[1].plot(epochs, self.histories["val_rmse"], label="Val RMSE", marker='s')
        axs[1].plot(epochs, self.histories["val_r2"], label="Val R²", marker='^')
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Metric Value")
        axs[1].set_title("Validation Loss & Metrics")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":

    dataset = MPNNDataset.load("PAMPA_dataset.pt")
    graph = dataset[0] # read the first graph from dataset 
    
    # get the input dimension of node and edges
    node_in_dim = graph.x.shape[1]
    edge_in_dim = graph.edge_attr.shape[1] if graph.edge_attr is not None else 0

    # Recreate model architecture exactly as before
    model = MPNNModel(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim, hidden_dim=64, num_layers=4, out_dim=1)

    trainer = MPNNTrainer(model, dataset, epochs=150, batch_size=32, lr=5e-4, Huber_beta=2, toy_dataset=False)
    trainer.train()

    # plot training and validation losses, and several validation metrics
    trainer.plot_training_history()

    # After training, optionally save:
    torch.save(model, "mpnn_model_full_trained.pt")

    model.eval()
    trainer.test()  # Only after you’re satisfied with validation