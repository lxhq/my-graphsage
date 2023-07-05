import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

class node_set(Dataset):
    def __init__(self, nodes, labels, device):
        self.nodes = torch.tensor(nodes, device=device, dtype=torch.float)
        self.labels = torch.tensor(labels, device=device, dtype=torch.long)

    def __getitem__(self, index):
        return self.nodes[index], self.labels[index]

    def __len__(self):
        return len(self.nodes)

class aggregator(torch.nn.Module):
    def __init__(self, feat_dim, device):
        super(aggregator, self).__init__()
        self.device = device
        self.feat_dim = feat_dim
    
    def forward(self, nodes, adj, features):
        neighbors = set()
        for node in nodes:
            neighbors = neighbors.union(adj[node.item()])
            neighbors.add(node.item())
        neighbors = list(neighbors)
        neighbors_dict = {i:j for j, i in enumerate(neighbors)}
        mask = torch.zeros(len(nodes), len(neighbors), device=self.device, dtype=torch.float)
        for idx, node in enumerate(nodes):
            mask[idx][[neighbors_dict[i] for i in adj[node.item()]]] = 1
            mask[idx][neighbors_dict[node.item()]] = 1
        num_neighbors = mask.sum(1, keepdim=True)
        mask = mask.div(num_neighbors)
        neighbor_features = torch.zeros(len(neighbors_dict), self.feat_dim, dtype=torch.float, device=self.device)
        for idx, key in enumerate(neighbors_dict.keys()):
            neighbor_features[idx] = torch.tensor(features[key])
        return mask.mm(neighbor_features)
        
class encoder(torch.nn.Module):
    def __init__(self, device, in_feat, out_feat):
        super(encoder, self).__init__()
        self.device = device
        self.linear = torch.nn.Linear(in_feat, out_feat)
    
    def forward(self, agg):
        return torch.nn.functional.relu(self.linear(agg))

class graph_sage(torch.nn.Module):
    def __init__(self, features, node_feat, embedding_feat, classes, layers, device):
        super(graph_sage, self).__init__()
        self.layers = layers
        self.features = features
        self.agg1 = aggregator(node_feat, device)
        self.enc1 = encoder(device, node_feat, embedding_feat)
        self.agg2 = aggregator(embedding_feat, device)
        self.enc2 = encoder(device, embedding_feat, embedding_feat)
        self.linear = torch.nn.Linear(embedding_feat, classes)

    def forward(self, input_layers, adj):
        out_agg1 = self.agg1(input_layers[0], adj, self.features)
        out_enc1 = self.enc1(out_agg1)
        out_agg2 = self.agg2(input_layers[1], adj, {input_layers[0][idx].item(): feature for idx, feature in enumerate(out_enc1)})
        out_enc2 = self.enc2(out_agg2)
        return self.linear(out_enc2)

def load_cora():
    with open("/home/lxhq/workspace/my-graphsage/cora/cora.content") as content:
        content = pd.read_csv(content, sep='\t', header=None)
        labels_raw = content.iloc[:, -1]
        labels_map = {i: j for j, i in enumerate(set(labels_raw))}
        labels = np.array([labels_map[i] for i in labels_raw], dtype=np.int32)
        features = np.array(content.iloc[:, 1:-1], dtype=np.int32)
        features = {idx: feature for idx, feature in enumerate(features)}
        idx = np.array(content.iloc[:, 0], dtype = np.int32)
        idx_map = {i: j for j, i in enumerate(idx)}

    with open("/home/lxhq/workspace/my-graphsage/cora/cora.cites") as cites:
        cites = pd.read_csv(cites, sep = '\t', header = None)
        edges_index = np.array(list(map(idx_map.get, cites.values.flatten())), dtype=np.int32)
        edges_index = edges_index.reshape(cites.shape)
        adj = defaultdict(set)
        for edge in edges_index:
            adj[edge[0]].add(edge[1])
            adj[edge[1]].add(edge[0])
    return features, adj, labels
    
def get_layers_input(input, adj, layers):
    node_layers = []
    all_nodes = set()
    for layer in range(layers):
        if layer == 0:
            node_layers.append(input)
            all_nodes.union([i.item() for i in input])
        else:
            temp = []
            for node in node_layers[layer - 1]:
                neighbors = adj[node.item()]
                temp.append(node.item())
                for neighbor in neighbors:
                    if neighbor not in all_nodes:
                        temp.append(neighbor)
                        all_nodes.add(neighbor)
            node_layers.append(torch.tensor(temp))
    return list(reversed(node_layers))

def train(train_data_loader:DataLoader, adj, net:graph_sage, optimizer:torch.optim.Adam, layers):
    net.train()
    for input, label in train_data_loader:
        input_layers = get_layers_input(input, adj, layers)
        optimizer.zero_grad()
        output = net(input_layers, adj)
        loss = torch.nn.functional.cross_entropy(output, label)
        loss.backward()
        optimizer.step()

def test(test_data_loader, adj, net:graph_sage, layers):
    net.eval()
    with torch.no_grad():
        total_loss = 0
        for (input, label) in test_data_loader:
            input_layers = get_layers_input(input, adj, layers)
            output = net(input_layers, adj)
            loss = torch.nn.functional.cross_entropy(output, label)
            total_loss += loss.item()
        print("loss", total_loss / len(test_data_loader))

def main():
    node_feat = 1433
    embedding_feat = 128
    epochs = 100
    batch = 256
    layers = 2
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    features, adj, labels = load_cora()
    nodes = np.array(range(len(features)))
    train_dataset = node_set(nodes[1500:], labels[1500:], device)
    test_dataset = node_set(nodes[:1000], labels[:1000], device)
    train_data_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, drop_last=False)
    test_data_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False, drop_last=False)
    net = graph_sage(features, node_feat, embedding_feat, len(set(labels)), layers, device)
    net.to(device)
    optim = torch.optim.Adam(net.parameters())
    for epoch in range(epochs):
        print("------", epoch, "------")
        train(train_data_loader, adj, net, optim, layers)
        test(test_data_loader, adj, net, layers)

if __name__ == "__main__":
    main()