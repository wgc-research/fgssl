import copy

import torch
from copy import deepcopy
from federatedscope.core.auxiliaries.enums import MODE
from federatedscope.core.auxiliaries.enums import LIFECYCLE
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.gfl.trainer.nodetrainer import NodeFullBatchTrainer
from federatedscope.core.auxiliaries.enums import LIFECYCLE
from federatedscope.core.trainers.context import CtxVar
from federatedscope.gfl.loss.vat import VATLoss
from federatedscope.core.trainers import GeneralTorchTrainer
from GCL.models import DualBranchContrast
import GCL.losses as L
import GCL.augmentors as A
import pyro
from GCL.models.contrast_model import WithinEmbedContrast
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
import torch.nn as nn


class FGCLTrainer2(GeneralTorchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FGCLTrainer2, self).__init__(model, data, device, config,
                                                 only_for_eval, monitor)

        self.global_model = copy.deepcopy(model)

    def register_default_hooks_train(self):
        super(FGCLTrainer2, self).register_default_hooks_train()
        self.register_hook_in_train(new_hook=begin,
                                    trigger='on_fit_start',
                                    insert_pos=-1)
        self.register_hook_in_train(new_hook=leave,
                                    trigger='on_fit_end',
                                    insert_pos=-1)

    def register_default_hooks_eval(self):
        super(FGCLTrainer2, self).register_default_hooks_eval()
        self.register_hook_in_eval(new_hook=begin,
                                   trigger='on_fit_start',
                                   insert_pos=-1)
        self.register_hook_in_eval(new_hook=leave,
                                   trigger='on_fit_end',
                                   insert_pos=-1)

    def _hook_on_batch_forward(self, ctx):

        batch = ctx.data_batch.to(ctx.device)
        mask = batch['{}_mask'.format(ctx.cur_split)].detach()

        label = batch.y[batch['{}_mask'.format(ctx.cur_split)]]

        self.global_model.to(ctx.device).eval()

        pred, raw_feature_local, adj_sampled , adj_logits, adj_orig = ctx.model(batch)

        pred_global, raw_feature_global, adj_sampled , adj_logits, adj_orig = self.global_model(batch)

        pred = pred[mask]

        loss1 = ctx.criterion(pred, label)


        kd_loss = com_distillation_loss(raw_feature_global,raw_feature_local,adj_orig,adj_sampled,2)
        norm_w = adj_orig.shape[0] ** 2 / float((adj_orig.shape[0] ** 2 - adj_orig.sum()) * 2)
        pos_weight = torch.FloatTensor([float(adj_orig.shape[0] ** 2 - adj_orig.sum()) / adj_orig.sum()]).to(ctx.device)
        ga_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=pos_weight)

        # ctx.loss_batch = loss1  + (loss2 + loss3) * 0.5
        ctx.loss_batch = loss1
        ctx.batch_size = torch.sum(mask).item()
        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)


def begin(ctx):
    if 'lastModel' not in ctx.keys():
        ctx.lastModel = copy.deepcopy(ctx.model).to(ctx.device)


def leave(ctx):
    ctx.lastModel = copy.deepcopy(ctx.model).to(ctx.device)






def com_distillation_loss(t_logits, s_logits, adj_orig, adj_sampled, temp):

    s_dist = F.log_softmax(s_logits / temp, dim=-1)
    t_dist = F.softmax(t_logits / temp, dim=-1)
    kd_loss = temp * temp * F.kl_div(s_dist, t_dist.detach())


    adj = torch.triu(adj_orig * adj_sampled).detach()
    edge_list = (adj + adj.T).nonzero().t()

    s_dist_neigh = F.log_softmax(s_logits[edge_list[0]] / temp, dim=-1)
    t_dist_neigh = F.softmax(t_logits[edge_list[1]] / temp, dim=-1)

    kd_loss += temp * temp * F.kl_div(s_dist_neigh, t_dist_neigh.detach())

    return kd_loss