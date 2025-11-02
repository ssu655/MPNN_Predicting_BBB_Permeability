import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, GlobalAttention
from DataLoading import MPNNDataset

class MPNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim, dropout=0.2):
        super().__init__(aggr="add")  # sum messages from neighbors

        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, edge_index, edge_attr): # propagate is the PyG build-in function
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr): # compute the messages from node and edge embeddings of neighbors
        return self.message_mlp(torch.cat([x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x): # aggr_out is aggragated messages from neighbors
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))
    
# --- Full MPNN Model ---
class MPNNModel(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim,
                 node_emb_dim=64, edge_emb_dim=32, hidden_dim=64,
                 num_layers=3, out_dim = 1, dropout=0.2):
        super().__init__()

        # 1 Initial Embedding
        self.node_emb = nn.Linear(node_in_dim, node_emb_dim)
        self.edge_emb = nn.Linear(edge_in_dim, edge_emb_dim)

        # 2 Stack of MPNN layers
        self.layers = nn.ModuleList([
            MPNNLayer(node_dim=(node_emb_dim if i == 0 else hidden_dim),
                      edge_dim=edge_emb_dim,
                      hidden_dim=hidden_dim,
                      dropout=dropout)
            for i in range(num_layers)
        ])

        # 3 Readout (graph-level prediction)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, out_dim)  # single regression target (logP)
        )

        ''' 
        # code for attention mechanism, replacing the global_add_pool and readout methods above.
        node_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.readout_attention = GlobalAttention(
            gate_nn = nn.Linear(hidden_dim, 1),  # compute attention score per node
            nn = node_transform  # optional transformation on node features before pooling
        )

        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, out_dim)
        )
        '''

    def forward(self, data, return_embedding=False):
        # data = Batch object from PyG DataLoader
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 1 embedding of nodes and edges
        x = self.node_emb(x)                  # shape: [num_nodes, node_emb_dim]
        edge_attr = self.edge_emb(edge_attr)  # shape: [num_edges, edge_emb_dim]
        
        # 2 Message passing
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        # 3 Pooling all the nodes in one graph, into graph-level embedding
        graph_emb = global_add_pool(x, batch)  # [num_graphs, hidden_dim]
        if return_embedding:
            return graph_emb  # return embedding only
    
        # 4 Final prediction
        out = self.readout(graph_emb)  # [num_graphs, 1]
        return out
    
        '''
        # code for attention mechanism of the graph embedding, or graph summing
        graph_emb = self.readout_attention(x, batch)  # [num_graphs, hidden_dim]

        if return_embedding:
            return graph_emb  # just return embeddings

        # final prediction
        out = self.mlp_out(graph_emb)  # [num_graphs, 1]
        return out
        '''

if __name__ == "__main__":
    dataset = MPNNDataset.load("PAMPA_dataset.pt")
    graph = dataset[0] # read the first graph from dataset 
    
    # get the input dimension of node and edges
    node_in_dim = graph.x.shape[1]
    edge_in_dim = graph.edge_attr.shape[1] if graph.edge_attr is not None else 0

    model = MPNNModel(node_in_dim, edge_in_dim, hidden_dim=64, num_layers=3, out_dim=1)

    
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model.eval() # run the model in evaluation mode, dropout disabled
    for batch in loader:
        out = model(batch)
        print(out.shape)  # [batch_size, 1]
        print(out)
        break