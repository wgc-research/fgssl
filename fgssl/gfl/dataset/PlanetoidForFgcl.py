from torch_geometric.datasets import Planetoid
import torch
import numpy as np
from torch_geometric.io import read_planetoid_data
from torch_geometric.utils.loop import add_self_loops
from torch_geometric.data import Data

class PlanetoidForFgcl(Planetoid):
    def process_full_batch_data(self, data):
        print("Processing full batch data")
        nodes = torch.tensor(np.arange(data.num_nodes), dtype=torch.long)

        edge_index, edge_attr = add_self_loops(data.edge_index, data.edge_attr)

        data = Data(nodes=nodes, edge_index=data.edge_index, edge_attr=data.edge_attr, x=data.x, y=data.y,
                    train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask,
                    num_nodes=data.num_nodes, neighbor_index=edge_index, neighbor_attr=edge_attr)

        return [data]

    def process(self):
        data = read_planetoid_data(self.raw_dir, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)

        edge_attr = data.edge_attr
        edge_attr = torch.ones(data.edge_index.shape[1]) if edge_attr is None else edge_attr
        data.edge_attr = edge_attr

        data_list = self.process_full_batch_data(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

