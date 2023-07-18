import copy
from torch_geometric.nn import GCNConv
import torch
from copy import deepcopy
from federatedscope.core.auxiliaries.enums import MODE
from federatedscope.core.auxiliaries.enums import LIFECYCLE
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.gfl.trainer import LinkFullBatchTrainer
from federatedscope.gfl.trainer import GraphMiniBatchTrainer
from federatedscope.gfl.trainer.nodetrainer import NodeFullBatchTrainer
from federatedscope.core.auxiliaries.enums import LIFECYCLE
from federatedscope.core.trainers.context import CtxVar
from torch_geometric.utils import remove_self_loops, add_self_loops, degree

from federatedscope.gfl.loss.vat import VATLoss
from federatedscope.gfl.loss.suploss import SupConLoss
from federatedscope.core.trainers import GeneralTorchTrainer
from GCL.models import DualBranchContrast,SingleBranchContrast
import GCL.losses as L
import GCL.augmentors as A
from GCL.models.contrast_model import WithinEmbedContrast
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
import torch.nn as nn
from torch_geometric.utils import to_dense_adj


MODE2MASK = {
    'train': 'train_edge_mask',
    'val': 'valid_edge_mask',
    'test': 'test_edge_mask'
}

class FGCLTrainer1(NodeFullBatchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FGCLTrainer1, self).__init__(model, data, device, config,
                                           only_for_eval, monitor)
        self.global_model = copy.deepcopy(model)
        self.state = 0
        # self.aug = GCNConv(config.)
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.1), mode='L2L').to(device)
        self.withcontrast_model = WithinEmbedContrast(loss=L.BarlowTwins()).to(device)
        self.augWeak = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
        self.augStrongF = A.Compose([A.EdgeRemoving(pe=0.8), A.FeatureMasking(pf=0.5)])
        self.augNone = A.Identity()
        self.ccKD = Correlation()
        self.yn = 10
        self.mu = 50

    def register_default_hooks_train(self):
        super(FGCLTrainer1, self).register_default_hooks_train()
        self.register_hook_in_train(new_hook=begin,
                                    trigger='on_fit_start',
                                    insert_pos=-1)
        self.register_hook_in_train(new_hook=del_initialization_local,
                                    trigger='on_fit_end',
                                    insert_pos=-1)
        self.register_hook_in_train(new_hook=record_initialization_global,
                                    trigger='on_fit_start',
                                    insert_pos=-1)
        self.register_hook_in_train(new_hook=leave,
                                    trigger='on_fit_end',
                                    insert_pos=-1)

    def register_default_hooks_eval(self):
        super(FGCLTrainer1, self).register_default_hooks_eval()
        self.register_hook_in_eval(new_hook=begin,
                                   trigger='on_fit_start',
                                   insert_pos=-1)
        self.register_hook_in_eval(new_hook=del_initialization_local,
                                   trigger='on_fit_end',
                                   insert_pos=-1)
        self.register_hook_in_eval(new_hook=record_initialization_global,
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

        pred, raw_feature_local = ctx.model(batch)

        pred_global, raw_feature_global = self.global_model(batch)


        pred = pred[mask]
        pred_global = pred_global[mask].detach()

        loss1 = ctx.criterion(pred, label)

        batch1 = copy.deepcopy(batch)
        batch2 = copy.deepcopy(batch)

        g1, edge_index1, edge_weight1 = self.augWeak(batch.x, batch.edge_index)
        g2, edge_index2, edge_weight2 = self.augStrongF(batch.x, batch.edge_index)

        batch1.x = g1
        batch1.edge_index = edge_index1

        batch2.x = g2
        batch2.edge_index = edge_index2

        with torch.no_grad():
             pred_aug_global, globalOne = self.global_model(batch1)

        _, now2 = ctx.model(batch1)

        pred_aug_local, now = ctx.model(batch2)

        adj_orig = to_dense_adj(batch.edge_index, max_num_nodes=batch.x.shape[0]).squeeze(0).to(ctx.device)

        struct_kd = com_distillation_loss(pred_aug_global, pred_aug_local, adj_orig, adj_orig, 3)
        simi_kd_loss = simi_kd(pred_aug_global, pred_aug_local, batch.edge_index, 4)

        # rkd_Loss = rkd_loss(pred_aug_local , pred_aug_global)
        # "tag"
        cc_loss = self.ccKD(pred_aug_local, pred_aug_global)
        loss_ds = simi_kd_2(batch.edge_index,pred_aug_local,pred_aug_global,self.yn) * self.mu
        loss_ff = edge_distribution_high(batch.edge_index,pred_aug_local,pred_aug_global)
        globalOne = globalOne[mask]
        now = now[mask]
        now2 = now2[mask]
        extra_pos_mask = torch.eq(label, label.unsqueeze(dim=1)).to(ctx.device)
        extra_pos_mask.fill_diagonal_(True)

        extra_neg_mask = torch.ne(label, label.unsqueeze(dim=1)).to(ctx.device)
        extra_neg_mask.fill_diagonal_(False)

        loss3 = self.contrast_model(globalOne, now, extra_pos_mask=extra_pos_mask,extra_neg_mask=extra_neg_mask)
        loss3 = self.contrast_model(now2,now,extra_pos_mask=extra_pos_mask,extra_neg_mask=extra_neg_mask)

        # raw_feature_local_list = list()
        # for clazz in range(int(ctx.cfg.model.out_channels)):
        #     temp = raw_feature_local[ clazz == label ] / (raw_feature_local[ clazz == label ].norm(dim=-1,keepdim=True) + 1e-6)
        #     mean_result = temp.sum(dim=0, keepdim=True)
        #     raw_feature_local_list.append(mean_result)
        #
        # graph_level_local = torch.concat(raw_feature_local_list,dim=0)
        #
        #
        #
        # raw_feature_global_list = list()
        # for clazz in range(int(ctx.cfg.model.out_channels)):
        #     temp = raw_feature_global[clazz == label] / (raw_feature_global[clazz == label].norm(dim=-1,keepdim=True) + 1e-6)
        #     mean_result = temp.sum(dim=0,keepdim=True)
        #     raw_feature_global_list.append(mean_result)
        #
        # graph_level_global = torch.concat(raw_feature_global_list, dim=0)
        #
        # cos = torch.nn.CosineEmbeddingLoss()
        #
        # N_tensor = torch.ones(graph_level_global.shape[0]).to(ctx.device)
        # cos_loss = cos(graph_level_global, graph_level_local, N_tensor)
        ctx.loss_batch = loss1 + loss3 * 3



        ctx.batch_size = torch.sum(mask).item()
        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)

    def _hook_on_fit_start_init(self, ctx):
        # prepare model and optimizer
        ctx.model.to(ctx.device)

        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            # Initialize optimizer here to avoid the reuse of optimizers
            # across different routines
            ctx.optimizer = get_optimizer(ctx.model,
                                          **ctx.cfg[ctx.cur_mode].optimizer)
            ctx.scheduler = get_scheduler(ctx.optimizer,
                                          **ctx.cfg[ctx.cur_mode].scheduler)

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)


def record_initialization_local(ctx):
    """Record weight denomaitor to cpu

    """
    ctx.weight_denomaitor = None


def del_initialization_local(ctx):
    """Clear the variable to avoid memory leakage

    """
    ctx.weight_denomaitor = None


def record_initialization_global(ctx):
    """Record the shared global model to cpu

    """

    pass


def begin(ctx):
    if 'lastModel' not in ctx.keys():
        ctx.lastModel = copy.deepcopy(ctx.model).to(ctx.device)


def leave(ctx):
    ctx.lastModel = copy.deepcopy(ctx.model).to(ctx.device)


from federatedscope.register import register_trainer






class FGCLTrainer2(LinkFullBatchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FGCLTrainer2, self).__init__(model, data, device, config,
                                           only_for_eval, monitor)
        self.global_model = copy.deepcopy(model).to(device)
        self.state = 0
        # self.aug = GCNConv(config.)
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L').to(device)
        self.withcontrast_model = WithinEmbedContrast(loss=L.BarlowTwins()).to(device)
        self.augWeak = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
        self.augStrongF = A.Compose([A.EdgeRemoving(pe=0.8), A.FeatureMasking(pf=0.3)])


    def _hook_on_batch_forward(self, ctx):
        data = ctx.data
        perm = ctx.data_batch
        batch = ctx.data.to(ctx.device)
        mask = ctx.data[MODE2MASK[ctx.cur_split]]
        edges = data.edge_index.T[mask]
        if data.x.shape[0] < 10000:
            print("j")
            if ctx.cur_split in ['train', 'val']:
                z, h = ctx.model((data.x, ctx.input_edge_index))
            else:
                z, h = ctx.model((data.x, data.edge_index))
            pred = ctx.model.link_predictor(h, edges[perm].T)
            label = data.edge_type[mask][perm]
            loss_ce = ctx.criterion(pred, label)
            batch1 = copy.deepcopy(batch).to(ctx.device)
            batch2 = copy.deepcopy(batch).to(ctx.device)

            g1, edge_index1, edge_weight1 = self.augWeak(batch.x, batch.edge_index)
            g2, edge_index2, edge_weight2 = self.augStrongF(batch.x, batch.edge_index)

            batch1.x = g1
            batch1.edge_index = edge_index1

            batch2.x = g2
            batch2.edge_index = edge_index2

            with torch.no_grad():
                 pred_aug_global, globalOne = self.global_model(batch1)


            pred_aug_local, now = ctx.model(batch2)

            adj_orig = to_dense_adj(batch.edge_index, max_num_nodes=batch.x.shape[0]).squeeze(0).to(ctx.device)

            struct_kd = com_distillation_loss(pred_aug_global, pred_aug_local, adj_orig, adj_orig, 3)
            simi_kd_loss = simi_kd(pred_aug_global, pred_aug_local, batch.edge_index, 4)

            globalOne = globalOne[mask]
            now = now[mask]

            extra_pos_mask = torch.eq(label, label.unsqueeze(dim=1)).to(ctx.device)
            extra_pos_mask.fill_diagonal_(True)

            extra_neg_mask = torch.ne(label, label.unsqueeze(dim=1)).to(ctx.device)
            extra_neg_mask.fill_diagonal_(False)

            loss3 = self.contrast_model(globalOne, now, extra_pos_mask=extra_pos_mask,extra_neg_mask=extra_neg_mask)

            ctx.loss_batch = loss_ce + loss3 * 3

            ctx.batch_size = len(label)
            ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
            ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        else:
            if ctx.cur_split in ['train', 'val']:
                z, h = ctx.model((data.x, ctx.input_edge_index))
            else:
                z, h = ctx.model((data.x, data.edge_index))
            pred = ctx.model.link_predictor(h, edges[perm].T)
            label = data.edge_type[mask][perm]
            loss_ce = ctx.criterion(pred, label)

            ctx.loss_batch = loss_ce
            ctx.batch_size = len(label)
            ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
            ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)





class FGCLTrainer3(GraphMiniBatchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FGCLTrainer3, self).__init__(model, data, device, config,
                                           only_for_eval, monitor)
        self.global_model = copy.deepcopy(model).to(device)
        self.state = 0
        # self.aug = GCNConv(config.)
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L').to(device)
        self.withcontrast_model = WithinEmbedContrast(loss=L.BarlowTwins()).to(device)
        self.augWeak = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
        self.augStrongF = A.Compose([A.EdgeRemoving(pe=0.8), A.FeatureMasking(pf=0.3)])


    def _hook_on_batch_forward(self, ctx):
        data = ctx.data
        perm = ctx.data_batch
        batch = ctx.data.to(ctx.device)
        mask = ctx.data[MODE2MASK[ctx.cur_split]]
        edges = data.edge_index.T[mask]
        if data.x.shape[0] < 10000:
            print("j")
            if ctx.cur_split in ['train', 'val']:
                z, h = ctx.model((data.x, ctx.input_edge_index))
            else:
                z, h = ctx.model((data.x, data.edge_index))
            pred = ctx.model.link_predictor(h, edges[perm].T)
            label = data.edge_type[mask][perm]
            loss_ce = ctx.criterion(pred, label)
            batch1 = copy.deepcopy(batch).to(ctx.device)
            batch2 = copy.deepcopy(batch).to(ctx.device)

            g1, edge_index1, edge_weight1 = self.augWeak(batch.x, batch.edge_index)
            g2, edge_index2, edge_weight2 = self.augStrongF(batch.x, batch.edge_index)

            batch1.x = g1
            batch1.edge_index = edge_index1

            batch2.x = g2
            batch2.edge_index = edge_index2

            with torch.no_grad():
                 pred_aug_global, globalOne = self.global_model(batch1)


            pred_aug_local, now = ctx.model(batch2)

            adj_orig = to_dense_adj(batch.edge_index, max_num_nodes=batch.x.shape[0]).squeeze(0).to(ctx.device)

            struct_kd = com_distillation_loss(pred_aug_global, pred_aug_local, adj_orig, adj_orig, 3)
            simi_kd_loss = simi_kd(pred_aug_global, pred_aug_local, batch.edge_index, 4)

            globalOne = globalOne[mask]
            now = now[mask]

            extra_pos_mask = torch.eq(label, label.unsqueeze(dim=1)).to(ctx.device)
            extra_pos_mask.fill_diagonal_(True)

            extra_neg_mask = torch.ne(label, label.unsqueeze(dim=1)).to(ctx.device)
            extra_neg_mask.fill_diagonal_(False)

            loss3 = self.contrast_model(globalOne, now, extra_pos_mask=extra_pos_mask,extra_neg_mask=extra_neg_mask)

            ctx.loss_batch = loss_ce + loss3 * 3

            ctx.batch_size = len(label)
            ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
            ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        else:
            if ctx.cur_split in ['train', 'val']:
                z, h = ctx.model((data.x, ctx.input_edge_index))
            else:
                z, h = ctx.model((data.x, data.edge_index))
            pred = ctx.model.link_predictor(h, edges[perm].T)
            label = data.edge_type[mask][perm]
            loss_ce = ctx.criterion(pred, label)

            ctx.loss_batch = loss_ce
            ctx.batch_size = len(label)
            ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
            ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)


def call_my_trainer(trainer_type):
    if trainer_type == 'fgcl1':
        trainer_builder = FGCLTrainer1
        return trainer_builder
    elif trainer_type == 'fgcl2':
        trainer_builder = FGCLTrainer2
        return trainer_builder
    elif trainer_type == 'fgcl3':
        trainer_builder = FGCLTrainer3
        return trainer_builder


register_trainer('fgcl1', call_my_trainer)
register_trainer('fgcl2', call_my_trainer)
register_trainer('fgcl3', call_my_trainer)





def GSP(student_feat, teacher_feat):
    student_feat = F.normalize( student_feat, p = 2, dim = -1)
    teacher_feat = F.normalize(teacher_feat, p = 2, dim = -1)
    student_pw_sim = torch.mm(student_feat, student_feat.transpose(0, 1))
    teacher_pw_sim = torch.mm(teacher_feat, teacher_feat.transpose(0, 1))

    loss_gsp = F.mse_loss(student_pw_sim, teacher_pw_sim)

    return loss_gsp




def _pdist(e, squared, eps):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


def rkd_loss(f_s, f_t, squared=False, eps=1e-12, distance_weight=25, angle_weight=50):
    stu = f_s.view(f_s.shape[0], -1)
    tea = f_t.view(f_t.shape[0], -1)

    # RKD distance loss
    with torch.no_grad():
        t_d = _pdist(tea, squared, eps)
        mean_td = t_d[t_d > 0].mean()
        t_d = t_d / (mean_td + 1e-6)

    d = _pdist(stu, squared, eps)
    mean_d = d[d > 0].mean()
    d = d / (mean_d+1e-6)

    loss_d = F.smooth_l1_loss(d, t_d)

    # RKD Angle loss
    with torch.no_grad():
        td = tea.unsqueeze(0) - tea.unsqueeze(1)
        norm_td = F.normalize(td, p=2, dim=2)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

    sd = stu.unsqueeze(0) - stu.unsqueeze(1)
    norm_sd = F.normalize(sd, p=2, dim=2)
    s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

    loss_a = F.smooth_l1_loss(s_angle, t_angle)

    loss = distance_weight * loss_d + angle_weight * loss_a
    return loss

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def kl(pred, pred_global, tau=2):
    p_loss = F.kl_div(F.log_softmax(pred / tau, dim=-1), F.softmax(pred_global / tau, dim=-1), reduction='none') * (tau ** 2) / pred.shape[0]
    q_loss = F.kl_div(F.log_softmax(pred_global / tau, dim=-1), F.softmax(pred / tau, dim=-1), reduction='none') * (tau ** 2) / pred.shape[0]
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

def kd(pred, pred_global, tau):
    p_s = F.log_softmax(pred / tau, dim=1)
    p_t = F.softmax(pred_global / tau, dim=1)
    loss = F.kl_div(p_s, p_t, reduction='none').sum(1).mean()
    loss *= tau ** 2
    return loss

def sp_loss(g_s, g_t):
    return sum([similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)])


def similarity_loss(f_s, f_t):
    bsz = f_s.shape[0]
    f_s = f_s.view(bsz, -1)
    f_t = f_t.view(bsz, -1)

    G_s = torch.mm(f_s, torch.t(f_s))
    G_s = torch.nn.functional.normalize(G_s)
    G_t = torch.mm(f_t, torch.t(f_t))
    G_t = torch.nn.functional.normalize(G_t)

    G_diff = G_t - G_s
    loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
    return loss


def com_distillation_loss(t_logits, s_logits, adj_orig, adj_sampled, temp):

    s_dist = F.log_softmax(s_logits / temp, dim=-1)
    t_dist = F.softmax(t_logits / temp, dim=-1)
    kd_loss = temp * temp * F.kl_div(s_dist, t_dist.detach())


    adj = torch.triu(adj_orig).detach()
    edge_list = (adj + adj.T).nonzero().t()

    s_dist_neigh = F.log_softmax(s_logits[edge_list[0]] / temp, dim=-1)
    t_dist_neigh = F.softmax(t_logits[edge_list[1]] / temp, dim=-1)

    kd_loss += temp * temp * F.kl_div(s_dist_neigh, t_dist_neigh.detach())

    return kd_loss


def simi_kd_2(edge_index, feats, out,yn):
    tau = 0.1
    kl = nn.KLDivLoss()
    N = feats.shape[0]
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    src = edge_index[0]
    dst = edge_index[1]
    deg = degree(dst,num_nodes=N)
    values, max_degree_nodes = torch.topk(deg, yn)
    nodes_index = max_degree_nodes
    loss = 0
    feats = F.softmax(feats / tau, dim=-1)
    out = F.softmax(out / tau, dim=-1)
    _1 = torch.cosine_similarity(feats[src], feats[dst], dim=-1)
    _2 = torch.cosine_similarity(out[src], out[dst], dim=-1)
    for index in nodes_index:
        index_n = edge_index[:,torch.nonzero(edge_index[1] == index.item()).squeeze()]
        _1 = torch.cosine_similarity(feats[index_n[0]], feats[index_n[1]], dim=-1)
        _2 = torch.cosine_similarity(out[index_n[0]], out[index_n[1]], dim=-1)
        _1 = F.log_softmax(_1,dim=0)
        _2 = F.softmax(_2,dim=0)
        loss += kl(_1, _2)
    return loss

def edge_distribution_high(edge_idx, feats, out):

    tau =0.1
    src = edge_idx[0]
    dst = edge_idx[1]
    criterion_t = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    feats_abs = torch.abs(feats[src] - feats[dst])
    e_softmax = F.log_softmax(feats_abs / tau, dim=-1)

    out_1 = torch.abs(out[src] - out[dst])
    e_softmax_2 = F.log_softmax(out_1 / tau, dim=-1)

    loss_s = criterion_t(e_softmax, e_softmax_2)
    return loss_s
def simi_kd(global_nodes, local_nodes, edge_index, temp):
    adj_orig = to_dense_adj(edge_index).squeeze(0)
    adj_orig.fill_diagonal_(True)
    s_dist = F.log_softmax(local_nodes / temp, dim=-1)
    t_dist = F.softmax(global_nodes / temp, dim=-1)
    # kd_loss = temp * temp * F.kl_div(s_dist, t_dist.detach())
    local_simi = torch.cosine_similarity(local_nodes.unsqueeze(1), local_nodes.unsqueeze(0), dim=-1)
    global_simi = torch.cosine_similarity(global_nodes.unsqueeze(1), global_nodes.unsqueeze(0), dim=-1)

    local_simi = torch.where(adj_orig > 0, local_simi, torch.zeros_like(local_simi))
    global_simi = torch.where(adj_orig > 0, global_simi, torch.zeros_like(global_simi))

    s_dist_neigh = F.log_softmax(local_simi / temp, dim=-1)
    t_dist_neigh = F.softmax(global_simi / temp, dim=-1)

    kd_loss = temp * temp * F.kl_div(s_dist_neigh, t_dist_neigh.detach())

    return kd_loss


class Correlation(nn.Module):
    """Similarity-preserving loss. My origianl own reimplementation
    based on the paper before emailing the original authors."""
    def __init__(self):
        super(Correlation, self).__init__()

    def forward(self, f_s, f_t):
        return self.similarity_loss(f_s, f_t)

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        G_s = G_s / G_s.norm(2)
        G_t = torch.mm(f_t, torch.t(f_t))
        G_t = G_t / G_t.norm(2)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss
