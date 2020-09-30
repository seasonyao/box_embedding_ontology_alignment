from .box_operations import *
from .utils import log1mexp
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module
from typing import *


ModelOutput = Dict[str, Union[Tensor, Module]]


def mean_unit_cube_loss(model_out: ModelOutput, _) -> Tensor:
    return ((model_out["box_embeddings_orig"] - 1).clamp(0) + (-model_out["box_embeddings_orig"]).clamp(0)).sum(dim=[-2, -1]).mean()


def mean_unary_kl_loss(unary: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Callable:
    """ Factory Function to create the actual mean_unary_kl_loss function with the given set of unary probabilities. """
    def mean_unary_kl_loss(model_out: ModelOutput, _) -> Tensor:
        return kl_div_sym(model_out["unary_probs"], unary, eps).mean()
    return mean_unary_kl_loss


def mean_unary_kl_loss_log(unary: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Callable:
    """ Factory Function to create the actual mean_unary_kl_loss function with the given set of unary probabilities. """
    def mean_unary_kl_loss_log(model_out: ModelOutput, _) -> Tensor:
        return kl_div_sym_log(model_out["log_unary_probs"], unary, eps).mean()
    return mean_unary_kl_loss_log


def mean_cond_kl_loss(model_out: ModelOutput, target: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    return kl_div_sym(model_out["P_A_B"] / model_out["P_B"], target[:, 4], eps).mean() + kl_div_sym(model_out["P_A_B"] / model_out["P_A"], target[:,5], eps).mean()


def mean_cond_kl_loss_log(model_out: ModelOutput, target: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    return kl_div_sym_log(model_out["log P(A|B)"], target, eps).mean()


def mean_joint_kl_loss(model_out: ModelOutput, target: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    # P(A,B)
    loss = kl_div_term(model_out["P_A_B"], target[:,0], eps)
    # P(A,-B)
    loss += kl_div_term(model_out["P_A"] - model_out["P_A_B"], target[:,1], eps)
    # P(-A,B)
    loss += kl_div_term(model_out["P_B"] - model_out["P_A_B"], target[:,2], eps)
    # P(-A,-B)
    loss += kl_div_term(1 - model_out["P_A"] - model_out["P_B"] + model_out["P_A_B"], target[:,3], eps)
    return loss.mean()


def mean_joint_kl_loss(model_out: ModelOutput, target: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    # P(A,B)
    loss = kl_div_term(model_out["P_A_B"], target[:,0], eps)
    # P(A,-B)
    loss += kl_div_term(model_out["P_A"] - model_out["P_A_B"], target[:,1], eps)
    # P(-A,B)
    loss += kl_div_term(model_out["P_B"] - model_out["P_A_B"], target[:,2], eps)
    # P(-A,-B)
    loss += kl_div_term(1 - model_out["P_A"] - model_out["P_B"] + model_out["P_A_B"], target[:,3], eps)
    return loss.mean()


def mean_joint_jensen_shannon_loss(model_out: ModelOutput, target: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    # P(A,B)
    loss = jensen_shannon(model_out["P_A_B"], target[:,0], eps)
    # P(A,-B)
    loss += jensen_shannon(model_out["P_A"] - model_out["P_A_B"], target[:,1], eps)
    # P(-A,B)
    loss += jensen_shannon(model_out["P_B"] - model_out["P_A_B"], target[:,2], eps)
    # P(-A,-B)
    loss += jensen_shannon(1 - model_out["P_A"] - model_out["P_B"] + model_out["P_A_B"], target[:,3], eps)
    return loss.mean()


def mean_joint_triple_jensen_shannon_loss(model_out: ModelOutput, target: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    # P(A,B,C)
    loss = jensen_shannon(model_out["P_A_B_C"], target[:,0], eps)
    # P(A,B,-C)
    loss += jensen_shannon(model_out["P_A_B"] - model_out["P_A_B_C"], target[:,1], eps)
    # P(A,-B,C)
    loss += jensen_shannon(model_out["P_A_C"] - model_out["P_A_B_C"], target[:,2], eps)
    # P(A,-B,-C) = P(A, -B) - P(A, -B, C) = P(A) - P(A, B) - (P(A, C) - P(A, B, C)) = P(A) - P(A, B) - P(A, C) + P(A, B, C)
    loss += jensen_shannon(model_out["P_A"] - model_out["P_A_B"] - model_out["P_A_C"] + model_out["P_A_B_C"], target[:,3], eps)
    # P(-A, B, C)
    loss += jensen_shannon(model_out["P_B_C"] - model_out["P_A_B_C"], target[:,4], eps)
    # P(-A, B, -C) = P(B) - P(A, B) - P(B, C) + P(A, B, C)
    loss += jensen_shannon(model_out["P_B"] - model_out["P_A_B"] - model_out["P_B_C"] + model_out["P_A_B_C"], target[:,5], eps)
    # P(-A, -B, C) = P(C) - P(A, C) - P(B, C) + P(A, B, C)
    loss += jensen_shannon(model_out["P_C"] - model_out["P_A_C"] - model_out["P_B_C"] + model_out["P_A_B_C"], target[:,6], eps)
    # P(-A, -B, -C)
    loss += jensen_shannon(1 - model_out["P_A"] - model_out["P_B"] - model_out["P_C"]  + model_out["P_A_B"] + model_out["P_A_C"] + model_out["P_B_C"] - model_out["P_A_B_C"], target[:,7], eps)
    return loss.mean()


def mean_cond_jensen_shannon_loss(model_out: ModelOutput, target: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    return jensen_shannon(model_out["P_A_B"] / model_out["P_B"], target[:, 4], eps).mean() + jensen_shannon(model_out["P_A_B"] / model_out["P_A"], target[:,5], eps).mean()


def jensen_shannon(p, q, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    m = (p + q) / 2
    return (kl_div_term(p,m,eps) + kl_div_term(q,m,eps))/2


def kl_div_sym(p: Tensor, q: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    return kl_div_term(p, q, eps) + kl_div_term(1-p, 1-q, eps)


def kl_div_sym_log(log_p: Tensor, q: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    return kl_div_term_log(log_p, q, eps) + kl_div_term_log(log1mexp(-(log_p.clamp_max(-1e-7))), 1-q, eps)

def kl_div_term_log(log_p: Tensor, q: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    return F.kl_div(log_p, q, reduction="none")

def kl_div_term(p: Tensor, q: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    return F.kl_div(torch.log(p.clamp_min(eps)), q.clamp_min(eps), reduction="none")


def mean_pull_loss(model_out: ModelOutput, target: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Pulls together boxes which are disjoint but should overlap.
    """
    A, B = model_out["A"], model_out["B"]
    _needing_pull_mask = needing_pull_mask(A, B, target[:, 0])
    num_needing_pull = _needing_pull_mask.sum()
    if num_needing_pull == 0:
        return 0
    else:
        penalty = ((A[:,:,0] - B[:,:,1] + eps).clamp(0) + (B[:,:,0] - A[:,:,1] + eps).clamp(0)).sum(dim=-1)
        return penalty[_needing_pull_mask].sum() / num_needing_pull


def mean_push_loss(model_out: ModelOutput, target: Tensor, eps: float = 1e-6) -> Tensor:
    A, B = model_out["A"], model_out["B"]
    _needing_push_mask = needing_push_mask(A, B, target[:, 0])
    num_needing_push = _needing_push_mask.sum()
    if num_needing_push == 0:
        return 0
    else:
        penalty = torch.min((A[:,:,1] - B[:,:,0] + eps).clamp(0).min(dim=-1)[0], (B[:,:,1] - A[:,:,0] + eps).clamp(0).min(dim=-1)[0])
        return penalty[_needing_push_mask].sum() / num_needing_push


def mean_surrogate_loss(model_out: ModelOutput, target: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    """
    Note: surrogate loss is only used with clamp_volume, so it is hardcoded here.
    """
    A, B = model_out["A"], model_out["B"]
    A_join_B = join(A, B)
    _needing_pull_mask = needing_pull_mask(A, B, target)
    num_needing_pull = _needing_pull_mask.sum()
    if num_needing_pull == 0:
        return 0
    else:
        surrogate = clamp_volume(A_join_B) - clamp_volume(A) - clamp_volume(B)
        penalty = torch.log(surrogate.clamp_min(eps)) - torch.log(clamp_volume(A).clamp_min(eps))
        return (target[_needing_pull_mask[0]] * penalty[_needing_pull_mask]).sum() / num_needing_pull


def mean_weighted_pull_loss(model_out: ModelOutput, target: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    ids = model_out["ids"]
    two_boxes_mask = model_out["two_boxes_mask"]
    box_embeddings = model_out["box_embeddings"]
    two_vol = model_out["two_vol"]
    A = box_embeddings[:, ids[two_boxes_mask, 0]]
    B = box_embeddings[:, ids[two_boxes_mask, 1]]
    target_prob_A_given_B = target[two_boxes_mask]

    _needing_pull_mask = (two_vol <= eps) & (target_prob_A_given_B > eps)
    num_needing_pull = _needing_pull_mask.sum()
    if num_needing_pull == 0:
        return 0
    else:
        penalty = ((A[:,:,0] - B[:,:,1] + eps).clamp(0) + (B[:,:,0] - A[:,:,1] + eps).clamp(0)).sum(dim=-1)
        return model_out["weights_layer"](penalty[:,_needing_pull_mask].sum(dim=1)) / num_needing_pull
