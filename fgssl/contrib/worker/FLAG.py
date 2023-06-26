import copy
import logging
import random

import torch
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from tqdm import tqdm

from federatedscope.core.message import Message
from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client

logger = logging.getLogger(__name__)
from sklearn import manifold
import numpy as np
import pandas as pd
# import cca_core
# from CKA import linear_CKA, kernel_CKA
import matplotlib as mpl
import matplotlib.pyplot as plt
import gensim
from torch_geometric.utils import to_dense_adj
import seaborn as sns
import seaborn.objects as so


# Build your worker here.
class FGCLClient(Client):
    def callback_funcs_for_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content

        self.trainer.state = round
        self.trainer.global_model.load_state_dict(content)
        self.trainer.update(content)
        self.state = round
        sample_size, model_para, results = self.trainer.train()
        if self._cfg.federate.share_local_model and not \
                self._cfg.federate.online_aggr:
            model_para = copy.deepcopy(model_para)
        logger.info(
            self._monitor.format_eval_res(results,
                                          rnd=self.state,
                                          role='Client #{}'.format(self.ID)))

        # self.comm_manager.send(
        #     Message(msg_type='model_para',
        #             sender=self.ID,
        #             receiver=[sender],
        #             state=self.state,
        #             content=(sample_size, model_para, self.trainer.ctx.data['test'])))
        #
        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, model_para)))


class FGCLServer(Server):
    def _perform_federated_aggregation(self):
        model_for_e = copy.deepcopy(self.model)
        """
        Perform federated aggregation and update the global model
        """
        train_msg_buffer = self.msg_buffer['train'][self.state]
        for model_idx in range(self.model_num):
            model = self.models[model_idx]
            aggregator = self.aggregators[model_idx]
            msg_list = list()
            staleness = list()

            for client_id in train_msg_buffer.keys():
                if self.model_num == 1:
                    msg_list.append(train_msg_buffer[client_id])
                else:
                    train_data_size, model_para_multiple = \
                        train_msg_buffer[client_id]
                    msg_list.append(
                        (train_data_size, model_para_multiple[model_idx]))

                # The staleness of the messages in train_msg_buffer
                # should be 0
                staleness.append((client_id, 0))

            for staled_message in self.staled_msg_buffer:
                state, client_id, content = staled_message
                if self.model_num == 1:
                    msg_list.append(content)
                else:
                    train_data_size, model_para_multiple = content
                    msg_list.append(
                        (train_data_size, model_para_multiple[model_idx]))

                staleness.append((client_id, self.state - state))

            # if self.state % 50 == 0:
            #     list_feat = list()
            #     data = self.data['test'].dataset[0]
            #     labels = data.y
            #     for tuple in msg_list:
            #         size, para = tuple
            #         # edge_index = data_m.dataset[0].edge_index
            #         # adj_orig = to_dense_adj(edge_index, max_num_nodes=data.x.shape[0]).squeeze(0)
            #         # awe = anonymous_walk_embedding(adj_orig.fill_diagonal_(True))
            #         model_for_e.load_state_dict(para)
            #         model_for_e.eval()
            #         awe = 1
            #         x, f = model_for_e(data)
            #         f = f
            #         x = x
            #         list_feat.append(dict({"x": x, "f": f, "awe": awe, "id": len(list_feat)}))
            #     visual_cka(list_feat, self.state, labels, 6, "YlOrRd")  # GnBu magma "YlOrRd"

            aggregated_num = len(msg_list)
            agg_info = {
                'client_feedback': msg_list,
                'recover_fun': self.recover_fun,
                'staleness': staleness,
            }
            # logger.info(f'The staleness is {staleness}')
            result = aggregator.aggregate(agg_info)
            # Due to lazy load, we merge two state dict
            merged_param = merge_param_dict(model.state_dict().copy(), result)
            model.load_state_dict(merged_param, strict=False)
            # if self.state == 196:
            #     visual_tsne(model, self.data['val'].dataset[0], self.__class__.__name__)

        return aggregated_num


def merge_param_dict(raw_param, filtered_param):
    for key in filtered_param.keys():
        raw_param[key] = filtered_param[key]
    return raw_param


def call_my_worker(method):
    if method == 'fgcl':
        worker_builder = {'client': FGCLClient, 'server': FGCLServer}
        return worker_builder


register_worker('fgcl', call_my_worker)
def plot_features(features, labels, num_classes):
    colors = ['C' + str(i) for i in range(num_classes)]
    plt.figure(figsize=(6, 6))
    for l in range(num_classes):
        plt.scatter(
            features[labels == l, 0],
            features[labels == l, 1],
            c=colors[l], s=1, alpha=0.4)
    plt.xticks([])
    plt.yticks([])
    plt.show()





def visual_tsne(model, data,name):
    labels = data.y
    model.eval()
    z,h = model(data)
    num_class = labels.max().item() + 1
    z = z.detach().cpu().numpy()
    tsne = manifold.TSNE(n_components=2, perplexity=35, init='pca')
    plt.figure(figsize=(8, 8))
    x_tsne_data = list()
    f = tsne.fit_transform(z)
    for clazz in range(num_class):
        fp = f[labels == clazz]
        clazz = np.full(fp.shape[0], clazz)
        clazz = np.expand_dims(clazz, axis=1)
        fe = np.concatenate([fp, clazz], axis=1)
        x_tsne_data.append(fe)

    x_tsne_data = np.concatenate(x_tsne_data, axis=0)
    df_tsne = pd.DataFrame(x_tsne_data, columns=["dim1", "dim2", "class"])

    sns.scatterplot(data=df_tsne, palette="bright", hue='class', x='dim1', y='dim2')
    plt.legend([],[], frameon=False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    import os
    if not os.path.exists('data/output/tsne/'):
        os.mkdir("data/output/tsne/")
    plt.savefig("data/output/tsne/result_"+ name + ".png", format='png', dpi=800,
                pad_inches=0.1, bbox_inches='tight')
    plt.show()


import numpy as np


def anonymous_walk_embedding(adj_matrix, dimensions=128, walk_length=5, num_walks=5):
    import numpy as np
    from sklearn.decomposition import TruncatedSVD
    adj_matrix= adj_matrix.cpu().numpy()
    rows, cols = adj_matrix.shape
    walk_matrix = np.linalg.matrix_power(adj_matrix, walk_length)
    row_sums = np.array(np.sum(walk_matrix, axis=1)).flatten()
    walk_matrix = np.divide(walk_matrix, row_sums[:, np.newaxis])
    svd = TruncatedSVD(n_components=dimensions)
    awe_representation = svd.fit_transform(walk_matrix)
    return awe_representation


import logging

import torch
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from tqdm import tqdm

from federatedscope.core.message import Message
from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client

logger = logging.getLogger(__name__)
from sklearn import manifold
import numpy as np
import pandas as pd
# import cca_core
# from CKA import linear_CKA, kernel_CKA
import matplotlib as mpl
import matplotlib.pyplot as plt
import gensim
from torch_geometric.utils import to_dense_adj
import seaborn as sns
import seaborn.objects as so


def call_my_worker(method):
    if method == 'fgcl':
        worker_builder = {'client': FGCLClient, 'server': FGCLServer}
        return worker_builder


register_worker('fgcl', call_my_worker)
def plot_features(features, labels, num_classes):
    colors = ['C' + str(i) for i in range(num_classes)]
    plt.figure(figsize=(6, 6))
    for l in range(num_classes):
        plt.scatter(
            features[labels == l, 0],
            features[labels == l, 1],
            c=colors[l], s=1, alpha=0.4)
    plt.xticks([])
    plt.yticks([])
    plt.show()

import numpy as np



def visual_cka(list_feat, state, labels, num_class, theme):
    plt.figure()
    client_number = len(list_feat)
    client_list = [i for i in range(client_number)]
    random.shuffle(client_list)
    result = np.zeros(shape=(client_number, client_number))
    for i in range(client_number):
        for j in range(client_number):
            result[i][j] = CKA(list_feat[i]["f"].detach().numpy(), list_feat[j]["f"].detach().numpy())

    np.save("data/output/cka/result_node.npy",arr=result)
    sns.heatmap(data=result, vmin=0.0, vmax=1.0, cmap=theme).invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.title("")
    plt.savefig("data/output/cka/node_state_" + str(state) + ".png", dpi=600)
    plt.show()

    plt.figure()
    client_number = len(list_feat)
    result = np.zeros(shape=(client_number, client_number))
    for i in range(client_number):
        for j in range(client_number):
            result[i][j] = CKA(list_feat[client_list[i]]["x"].detach().numpy(), list_feat[client_list[j]]["x"].detach().numpy())
    np.save("data/output/cka/result_node2.npy", arr=result)
    sns.heatmap(data=result, vmin=0.0, vmax=1.0, cmap=theme).invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.title("")
    plt.savefig("data/output/cka/structure_state_" + str(state) + ".png", dpi=600)
    plt.show()





def CKA(X, Y):
    '''Computes the CKA of two matrices. This is equation (1) from the paper'''

    nominator = unbiased_HSIC(np.dot(X, X.T), np.dot(Y, Y.T))
    denominator1 = unbiased_HSIC(np.dot(X, X.T), np.dot(X, X.T))
    denominator2 = unbiased_HSIC(np.dot(Y, Y.T), np.dot(Y, Y.T))

    cka = nominator / np.sqrt(denominator1 * denominator2)

    return cka


def unbiased_HSIC(K, L):
    '''Computes an unbiased estimator of HISC. This is equation (2) from the paper'''

    # create the unit **vector** filled with ones
    n = K.shape[0]
    ones = np.ones(shape=(n))

    # fill the diagonal entries with zeros
    np.fill_diagonal(K, val=0)  # this is now K_tilde
    np.fill_diagonal(L, val=0)  # this is now L_tilde

    # first part in the square brackets
    trace = np.trace(np.dot(K, L))

    # middle part in the square brackets
    nominator1 = np.dot(np.dot(ones.T, K), ones)
    nominator2 = np.dot(np.dot(ones.T, L), ones)
    denominator = (n - 1) * (n - 2)
    middle = np.dot(nominator1, nominator2) / denominator

    # third part in the square brackets
    multiplier1 = 2 / (n - 2)
    multiplier2 = np.dot(np.dot(ones.T, K), np.dot(L, ones))
    last = multiplier1 * multiplier2

    # complete equation
    unbiased_hsic = 1 / (n * (n - 3)) * (trace + middle - last)

    return unbiased_hsic

