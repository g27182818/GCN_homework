# %% Imports
import torch
import networkx as nx
import matplotlib.pyplot as plt
import time
import numpy as np


# %% Visualization function
# Visualization function for NX graph or PyTorch tensor
def visualize(h, color, epoch=None, loss=None):
    # plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")    


# %% Declare karateClub
from torch_geometric.datasets import KarateClub

dataset = KarateClub()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

# %% Detailed visualization

data = dataset[0]  # Get the first graph object.

print(data)
print('==============================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
print(f'Contains self-loops: {data.contains_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

# %% Imrimir edge index
edge_index = data.edge_index
print("This is the edge index for the first 20 edges in the list:")
print(edge_index.t()[0:20])
# %% Visualizacion grafica
from torch_geometric.utils import to_networkx

G = to_networkx(data, to_undirected=True)

plt.figure(figsize=(7, 7))
plt.title("Original graph")
visualize(G, color=data.y)
plt.show()

# %% GCN model implementation
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.

        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out, h


model = GCN()
print(model)

# %% 2D embedings computed with the unntrained model
_, h = model(data.x, data.edge_index)
print(f'Embedding shape: {list(h.shape)}')

plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.title("Original graph")
visualize(G, color=data.y)
plt.subplot(1, 2, 2)
plt.title("Unntrained Graph embeding")
visualize(h, color=data.y)
plt.show()

# Model, optimizer and loss declaration
model = GCN()
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.

# %% Model training function
def train(data):
    optimizer.zero_grad()  # Clear gradients.
    out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, h


max_epoch = 1001
# Loss vec
loss_vec = np.zeros(max_epoch)

# Dynamic plot
plt.figure(figsize=(14, 7))
plt.ion()
plt.show()
for epoch in range(1001):
    loss, h = train(data)
    loss_vec[epoch] = loss
    # Visualize the node embeddings every 10 epochs
    if epoch % 10 == 0:
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.title("2D embeding")
        visualize(h, color=data.y, epoch=epoch, loss=loss)
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(max_epoch)[0:epoch], loss_vec[0:epoch])
        plt.xlim([0, max_epoch])
        plt.ylim([0, 1.5])
        plt.grid()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Evolving Loss function")
        plt.draw()
        plt.pause(0.3)

