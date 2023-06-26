# federatedscope/contrib/model/my_gcn.py
import pyro
import torch
import torch.nn.functional as F

from torch.nn import ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj

from federatedscope.register import register_model
import torch.nn as nn
from federatedscope.core.mlp import MLP


class MyGCN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0):
        super(MyGCN, self).__init__()
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
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if (i+1) == len(self.convs):
                break
            else:
                z = x
                x = F.relu(F.dropout(self.bn1(x), p=self.dropout, training=self.training))

        return x, z
    def link_predictor(self, x, edge_index):
        x = x[edge_index[0]] * x[edge_index[1]]
        x = self.output(x)
        return x

class MYGIN(torch.nn.Module):
    r"""Graph Isomorphism Network model from the "How Powerful are Graph
    Neural Networks?" paper, in ICLR'19

    Arguments:
        in_channels (int): dimension of input.
        out_channels (int): dimension of output.
        hidden (int): dimension of hidden units, default=64.
        max_depth (int): layers of GNN, default=2.
        dropout (float): dropout ratio, default=.0.

    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0):
        super(MYGIN, self).__init__()
        self.convs = ModuleList()
        self.bn1 = nn.BatchNorm1d(hidden, momentum  =  0.01)
        for i in range(max_depth):
            if i == 0:
                self.convs.append(
                    GINConv(MLP([in_channels, hidden, hidden],
                                batch_norm=True)))
            elif (i + 1) == max_depth:
                self.convs.append(
                    GINConv(
                        MLP([hidden, hidden, out_channels], batch_norm=True)))
            else:
                self.convs.append(
                    GINConv(MLP([hidden, hidden, hidden], batch_norm=True)))
        self.dropout = dropout

    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        elif isinstance(data, tuple):
            x, edge_index = data
        else:
            raise TypeError('Unsupported data type!')

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if (i + 1) == len(self.convs):
                break
            else:
                z = x
                x = F.relu(F.dropout(self.bn1(x), p=self.dropout, training=self.training))
        return x, z
    def link_predictor(self, x, edge_index):
        x = x[edge_index[0]] * x[edge_index[1]]
        x = self.output(x)
        return x
from torch_geometric.nn import SAGEConv
class MYsage(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0):
        super(MYsage, self).__init__()

        self.num_layers = max_depth
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden))
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(hidden, hidden))
        self.convs.append(SAGEConv(hidden, out_channels))
    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        elif isinstance(data, tuple):
            x, edge_index = data
        else:
            raise TypeError('Unsupported data type!')

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if (i + 1) == len(self.convs):
                break
            else:
                z = x
                x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return x, z
    def link_predictor(self, x, edge_index):
        x = x[edge_index[0]] * x[edge_index[1]]
        x = self.output(x)
        return x
from torch_geometric.nn import GATConv



class MYGAT(torch.nn.Module):
    r"""GAT model from the "Graph Attention Networks" paper, in ICLR'18

    Arguments:
        in_channels (int): dimension of input.
        out_channels (int): dimension of output.
        hidden (int): dimension of hidden units, default=64.
        max_depth (int): layers of GNN, default=2.
        dropout (float): dropout ratio, default=.0.

    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0):
        super(MYGAT, self).__init__()
        self.convs = ModuleList()
        self.bn1 = nn.BatchNorm1d(hidden, momentum  =  0.01)
        for i in range(max_depth):
            if i == 0:
                self.convs.append(GATConv(in_channels, hidden))
            elif (i + 1) == max_depth:
                self.convs.append(GATConv(hidden, out_channels))
            else:
                self.convs.append(GATConv(hidden, hidden))
        self.dropout = dropout
        self.prelu = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.augConv = GATConv(in_channels,hidden)
        dim_list = [hidden for _ in range(2)]
        self.output = MLP([hidden] + dim_list + [out_channels],
                          batch_norm=True)
    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()

    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        elif isinstance(data, tuple):
            x, edge_index = data
        else:
            raise TypeError('Unsupported data type!')

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if (i + 1) == len(self.convs):
                break
            else:
                z = x
                x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return x, z

    def link_predictor(self, x, edge_index):
        x = x[edge_index[0]] * x[edge_index[1]]
        x = self.output(x)
        return x



class MyStar(torch.nn.Module):
    r"""GAT model from the "Graph Attention Networks" paper, in ICLR'18

    Arguments:
        in_channels (int): dimension of input.
        out_channels (int): dimension of output.
        hidden (int): dimension of hidden units, default=64.
        max_depth (int): layers of GNN, default=2.
        dropout (float): dropout ratio, default=.0.

    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0):
        super(MyStar, self).__init__()
        self.convs = ModuleList()
        self.bn1 = nn.BatchNorm1d(hidden, momentum  =  0.01)
        for i in range(max_depth):
            if i == 0:
                self.convs.append(GATConv(in_channels, hidden))
            elif (i + 1) == max_depth:
                self.convs.append(GATConv(hidden, out_channels))
            else:
                self.convs.append(GATConv(hidden, hidden))

        self.pre = torch.nn.Sequential(torch.nn.Linear(in_channels, hidden))
        self.embedding_s = torch.nn.Linear(64, hidden)
        self.dropout = dropout
        self.prelu = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.augConv = GATConv(in_channels,hidden)
        dim_list = [hidden for _ in range(2)]
        self.output = MLP([hidden] + dim_list + [out_channels],
                          batch_norm=True)
    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()

    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        elif isinstance(data, tuple):
            x, edge_index = data
        else:
            raise TypeError('Unsupported data type!')

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if (i + 1) == len(self.convs):
                break
            else:
                z = x
                x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return x, z

    def link_predictor(self, x, edge_index):
        x = x[edge_index[0]] * x[edge_index[1]]
        x = self.output(x)
        return x


def gcnbuilder(model_config, input_shape):
    x_shape, num_label, num_edge_features = input_shape
    model = MyGCN(x_shape[-1],
                  model_config.out_channels,
                  hidden=model_config.hidden,
                  max_depth=model_config.layer,
                  dropout=model_config.dropout)
    return model

def ginbuilder(model_config, input_shape):
    x_shape, num_label, num_edge_features = input_shape
    model = MYGIN(x_shape[-1],
                  model_config.out_channels,
                  hidden=model_config.hidden,
                  max_depth=model_config.layer,
                  dropout=model_config.dropout)
    return model

def gatbuilder(model_config, input_shape):
    x_shape, num_label, num_edge_features = input_shape
    model = MYGAT(x_shape[-1],
                  model_config.out_channels,
                  hidden=model_config.hidden,
                  max_depth=model_config.layer,
                  dropout=model_config.dropout)
    return model

def sagebuilder(model_config, input_shape):
    x_shape, num_label, num_edge_features = input_shape
    model = MYsage(x_shape[-1],
                  model_config.out_channels,
                  hidden=model_config.hidden,
                  max_depth=model_config.layer,
                  dropout=model_config.dropout)
    return model

def fedbuilder(model_config, input_shape):
    x_shape, num_label, num_edge_features = input_shape
    model = MyStar(x_shape[-1],
                  model_config.out_channels,
                  hidden=model_config.hidden,
                  max_depth=model_config.layer,
                  dropout=model_config.dropout)
    return model


def call_my_net(model_config, local_data):
    # Please name your gnn model with prefix 'gnn_'
    if model_config.type == "gnn_mygcn":
        model = gcnbuilder(model_config, local_data)
        return model
    if model_config.type == "gnn_mygin":
        model = ginbuilder(model_config, local_data)
        return model
    if model_config.type == "gnn_mygat":
        model = gatbuilder(model_config, local_data)
        return model
    if model_config.type == "gnn_mysage":
        model = sagebuilder(model_config, local_data)
        return model
    if model_config.type == "gnn_fedstar":
        model = fedbuilder(model_config, local_data)
        return model

register_model("gnn_fedstar",call_my_net)
register_model("gnn_mygin", call_my_net)
register_model("gnn_mygcn", call_my_net)
register_model("gnn_mygat", call_my_net)
register_model("gnn_mysage", call_my_net)



from torch_geometric.utils import to_networkx, degree, to_dense_adj, to_scipy_sparse_matrix
from scipy import sparse as sp
def init_structure_encoding(args, gs, type_init):

    if type_init == 'rw':
        for g in gs:
            # Geometric diffusion features with Random Walk
            A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
            D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

            Dinv=sp.diags(D)
            RW=A*Dinv
            M=RW

            SE_rw=[torch.from_numpy(M.diagonal()).float()]
            M_power=M
            for _ in range(args.n_rw-1):
                M_power=M_power*M
                SE_rw.append(torch.from_numpy(M_power.diagonal()).float())
            SE_rw=torch.stack(SE_rw,dim=-1)

            g['stc_enc'] = SE_rw

    elif type_init == 'dg':
        for g in gs:
            # PE_degree
            g_dg = (degree(g.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(1, args.n_dg)
            SE_dg = torch.zeros([g.num_nodes, args.n_dg])
            for i in range(len(g_dg)):
                SE_dg[i,int(g_dg[i]-1)] = 1

            g['stc_enc'] = SE_dg

    elif type_init == 'rw_dg':
        for g in gs:
            # SE_rw
            A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
            D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

            Dinv=sp.diags(D)
            RW=A*Dinv
            M=RW

            SE=[torch.from_numpy(M.diagonal()).float()]
            M_power=M
            for _ in range(args.n_rw-1):
                M_power=M_power*M
                SE.append(torch.from_numpy(M_power.diagonal()).float())
            SE_rw=torch.stack(SE,dim=-1)

            # PE_degree
            g_dg = (degree(g.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(1, args.n_dg)
            SE_dg = torch.zeros([g.num_nodes, args.n_dg])
            for i in range(len(g_dg)):
                SE_dg[i,int(g_dg[i]-1)] = 1

            g['stc_enc'] = torch.cat([SE_rw, SE_dg], dim=1)

    elif type_init == "single":
        A = to_scipy_sparse_matrix(gs.edge_index, num_nodes=g.num_nodes)
        D = (degree(gs.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

        Dinv = sp.diags(D)
        RW = A * Dinv
        M = RW

        SE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(args.n_rw - 1):
            M_power = M_power * M
            SE.append(torch.from_numpy(M_power.diagonal()).float())
        SE_rw = torch.stack(SE, dim=-1)

        # PE_degree
        g_dg = (degree(gs.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(1, args.n_dg)
        SE_dg = torch.zeros([gs.num_nodes, args.n_dg])
        for i in range(len(g_dg)):
            SE_dg[i, int(g_dg[i] - 1)] = 1

        gs['stc_enc'] = torch.cat([SE_rw, SE_dg], dim=1)

    return gs
