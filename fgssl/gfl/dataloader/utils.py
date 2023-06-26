
import torch
import numpy as np
from torch_geometric.io import read_planetoid_data
from torch_geometric.utils.loop import add_self_loops
from torch_geometric.data import Data


def process_full_batch_data(data):

    print("Processing neighbor_data ")
    nodes = torch.tensor(np.arange(data.num_nodes), dtype=torch.long)

    edge_index, edge_attr = add_self_loops(data.edge_index, data.edge_attr)
    index_orig = data.index_orig
    data = Data(nodes=nodes, edge_index=data.edge_index, edge_attr=data.edge_attr, x=data.x, y=data.y,
                train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask,
                num_nodes=data.num_nodes, neighbor_index=edge_index, neighbor_attr=edge_attr)
    data.index_orig = index_orig
    return data


def process_client_data(data):

    edge_attr = data.edge_attr
    edge_attr = torch.ones(data.edge_index.shape[1]) if edge_attr is None else edge_attr
    data.edge_attr = edge_attr

    data = process_full_batch_data(data)
    return data