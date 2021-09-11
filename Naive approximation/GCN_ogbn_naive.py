#%% Importaciones
import os
import torch
import time
import numpy as np
from torch_scatter import scatter
import networkx as nx
import matplotlib.pyplot as plt
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.nodeproppred import Evaluator
from torch_geometric.utils import to_networkx
from torch.nn import Linear
from torch_geometric.nn import GCNConv

#%% Importacion del dataset de proteinas
dataset = PygNodePropPredDataset(name = "ogbn-proteins")

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
# Define list of indexes
test_idx_lst = test_idx.tolist()
train_idx_lst = train_idx.tolist()
valid_idx_lst = valid_idx.tolist()
data = dataset[0]

# Visualizacion de dataset completo
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

# Visualizacion de grafo especifico
print(data)
print('==============================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
# print(f'Number of training nodes: {data.train_mask.sum()}')
# print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
print(f'Contains self-loops: {data.contains_self_loops()}')
#print(f'Is undirected: {data.is_undirected()}')


#%% Agregacion para obtener features de nodo

node_features_path = "nf_file.pt"
node_features_aggr = 'add'  # Can be 'add', 'mean', 'max'
total_no_of_nodes = data.y.shape[0]

# Save node features in node_features_path
if os.path.isfile(node_features_path):
    print('{} exists'.format(node_features_path))
else:
    if node_features_aggr in ['add', 'mean', 'max']:
        node_features = scatter(data.edge_attr,
                                data.edge_index[0],
                                dim=0,
                                dim_size=total_no_of_nodes,
                                reduce=node_features_aggr)
    else:
        raise Exception('Unknown Aggr Method')
    torch.save(node_features, node_features_path)
    print('Node features extracted are saved into file {}'.format(node_features_path))


#%% Definicion del modelo

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(1)
        self.conv1 = GCNConv(8, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.conv4 = GCNConv(2, 2)
        self.classifier = Linear(2, 112)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.relu()
        h = self.conv4(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.

        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out, h

device = torch.device('cuda')

model = GCN().to(device)
print(model)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.

def train(data):
    optimizer.zero_grad()  # Clear gradients.
    x = torch.load(node_features_path).to(device)
    out, h = model(x, data.edge_index.to(device))  # Perform a single forward pass.
    labels = data.y[train_idx_lst].to(device)
    loss = criterion(out[train_idx_lst].to(device), labels.float())  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, h

def evaluation(dataset, model, evaluator, device, epoch):
    # Put model in eval mode
    model.eval()

    # Obtain splits from dataset
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    data = dataset[0]

    # Define list of indexes
    test_idx_lst = test_idx.tolist()
    train_idx_lst = train_idx.tolist()
    valid_idx_lst = valid_idx.tolist()

    # Define targets of predictions
    target = data.y.to(device).float()
    # Define complete inputs
    x = torch.load(node_features_path).to(device)

    out, h = model(x, data.edge_index.to(device))
    pred = torch.sigmoid(out)
    # if epoch == 100:
    #     breakpoint()
    # #pred = out

    eval_result = {}

    input_dict = {"y_true": target[train_idx_lst], "y_pred": pred[train_idx_lst]}
    eval_result["train"] = evaluator.eval(input_dict)

    input_dict = {"y_true": target[valid_idx_lst], "y_pred": pred[valid_idx_lst]}
    eval_result["valid"] = evaluator.eval(input_dict)

    return eval_result


# Maximum number of epochs
max_epochs = 101
# Vector of losses
loss_vec = np.zeros(max_epochs)
train_metric = np.zeros(1 + max_epochs//10)
valid_metric = np.zeros(1 + max_epochs//10)

# Define oficial evaluator
evaluator = Evaluator(name="ogbn-proteins")

# Starting time
start = time.time()

# Trainning cycle
for epoch in range(max_epochs):
    loss, h = train(data)
    # Intermediate result print
    print("Done epoch " + str(epoch) + " with Loss: " + str(loss.cpu().detach().numpy()))
    # Compute loss
    loss_vec[epoch] = loss.cpu().detach().numpy()
    if np.mod(epoch, 10) == 0:
        # Compute metrics
        eval_dict = evaluation(dataset, model, evaluator, device, epoch)
        train_metric[epoch//10] = eval_dict["train"]['rocauc']
        valid_metric[epoch//10] = eval_dict["valid"]['rocauc']


# End time
end = time.time()

print("Total elapsed time: "+str(end-start))
print("Average time per epoch: "+str((end-start)/max_epochs))

# Parametro para mostrar grafica
show_loss = True

if show_loss:
    # Final plot
    plt.figure()
    plt.plot(np.arange(max_epochs), loss_vec)
    plt.plot(np.arange(0, max_epochs, 10), train_metric, '-o')
    plt.plot(np.arange(0, max_epochs, 10), valid_metric, '-o')
    plt.legend(["Loss", "Train ROC-AUC", "Val ROC-AUC"])
    plt.xlabel("Epoch")
    plt.grid()
    plt.show()






# print("Train metric: " + str(eval_dict["train"]))
# print("Validation metric: " + str(eval_dict["valid"]))

