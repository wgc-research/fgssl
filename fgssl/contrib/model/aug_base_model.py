from federatedscope.register import register_model
import torch
import torch.nn.functional as F

from torch.nn import ModuleList
from torch_geometric.data import Data
import pyro
from torch_geometric.nn import GCNConv
import torch.nn as nn
from torch_geometric.utils import to_dense_adj
import torch.distributions as dist
# Build you torch or tf model class here
class GCN_AUG(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0):
        super(GCN_AUG, self).__init__()
        self.convs = ModuleList()
        for i in range(max_depth):
            if i == 0:
                self.convs.append(GCNConv(in_channels, hidden))
            elif (i + 1) == max_depth:
                self.convs.append(GCNConv(hidden, out_channels))
            else:
                self.convs.append(GCNConv(hidden, hidden))
        self.dropout = dropout
        self.bn1 = nn.BatchNorm1d(hidden, momentum  =  0.01)
        self.prelu = nn.PReLU()

    def forward(self, data):

        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        elif isinstance(data, tuple):
            x, edge_index = data
        else:
            raise TypeError('Unsupported data type!')

        adj_sampled, adj_logits, adj_orig = self.sample(0.8, x, edge_index)

        edge_index = adj_sampled.nonzero().t()

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if (i+1) == len(self.convs):
                break
            else:
                z = self.prelu(x)
                x = F.relu(F.dropout(self.bn1(x), p=self.dropout, training=self.training))

        return x, z, adj_sampled , adj_logits, adj_orig

    def sample(self, alpha, x, edge_index):

        for i,conv in enumerate(self.convs):
            if i == 0:
                x = conv(x,edge_index)


        adj_orig = torch.zeros((x.shape[0],x.shape[0])).to('cuda:2')
        for row in edge_index.T:
            i,j = row
            adj_orig[i, j] = 1

        adj_logits = x @ x.T

        edge_probs = adj_logits / torch.max(adj_logits)

        edge_probs = alpha * edge_probs + (1-alpha) * adj_orig

        edge_probs = torch.where(edge_probs < 0, torch.zeros_like(edge_probs), edge_probs)

        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=1,
                                                                         probs=edge_probs).rsample()

        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T

        adj_sampled.fill_diagonal_(1)
        # D_norm = torch.diag(torch.pow(adj_sampled.sum(1), -0.5))
        # adj_sampled = D_norm @ adj_sampled @ D_norm

        return adj_sampled, adj_logits, adj_orig



def gnnbuilder(model_config, input_shape):
    x_shape, num_label, num_edge_features = input_shape
    model = GCN_AUG(x_shape[-1],
                  model_config.out_channels,
                  hidden=model_config.hidden,
                  max_depth=model_config.layer,
                  dropout=model_config.dropout)
    return model


def call_my_net(model_config, local_data):
    if model_config.type == "gnn_gcn_aug":
        model = gnnbuilder(model_config, local_data)
        return model


register_model("gnn_gcn_aug", call_my_net)
