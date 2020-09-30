from .box_operations import *
import torch
from torch import Tensor
import scipy.stats as spstats # For Spearman r
from sklearn.metrics import roc_curve, precision_recall_curve  # for roc_curve

def pearson_r(p: Tensor, q: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    """Pearson r statistic
    Implementation is translated from scipy.stats.pearsonr
    """
    mp = p.mean()
    mq = q.mean()
    pm, qm = p-mp, q-mq
    r_num = torch.sum(pm * qm)
    r_den = torch.sqrt(torch.sum(pm**2) * torch.sum(qm**2))
    r = r_num / (r_den + 1e-38)

    # Presumably, if abs(r) > 1, then it is only some small artifact of floating
    # point arithmetic.
    r = max(min(r, 1.0), -1.0)

    # The rest is leftover from the SciPy function, but we don't need it
    # n = p.shape[0]
    # df = n-2
    # if abs(r) == 1.0:
    #     prob = 0.0
    # else:
    #     t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
    #     prob = betai(0.5*df, 0.5, df / (df + t_squared))
    # return r, prob
    return r


def spearman_r(p: Tensor, q: Tensor) -> float:
    """Spearman r statistic"""
    # TODO: Make a pytorch tensor version of this
    p = p.cpu().detach().numpy()
    q = q.cpu().detach().numpy()
    sr, _ = spstats.spearmanr(p, q)
    return sr


def metric_pearson_r(model, data_in, data_out):
    return pearson_r(model(data_in)["P(A|B)"], data_out)


def metric_spearman_r(model, data_in, data_out):
    return spearman_r(model(data_in)["P(A|B)"], data_out)


def metric_num_needing_push(model, data_in, data_out):
    model_out = model(data_in)
    return needing_push_mask(model_out["A"], model_out["B"], data_out).sum()


def metric_num_needing_pull(model, data_in, data_out):
    model_out = model(data_in)
    return needing_pull_mask(model_out["A"], model_out["B"], data_out).sum()


def metric_hard_accuracy(model, data_in, data_out):
    hard_pred = model(data_in)["P(A|B)"] > 0.5
    return (data_out == hard_pred.float()).float().mean()


def metric_hard_f1(model, data_in, data_out):
    hard_pred = model(data_in)["P(A|B)"] > 0.5
    true_pos = data_out[hard_pred==1].sum()
    total_pred_pos = (hard_pred==1).sum().float()
    total_actual_pos = data_out.sum().float()
    precision = true_pos / total_pred_pos
    recall = true_pos / total_actual_pos
    return 2 * (precision*recall) / (precision + recall)

def metric_hard_accuracy_align(model, data_in, data_out, threshold:float):
    A_given_B = data_in[::2]
    B_given_A = data_in[1::2,:]
    data_out = data_out[::2]

    align_probs = torch.stack((model(A_given_B)["P(A|B)"], model(B_given_A)["P(A|B)"]), dim=1)
    p = torch.min(align_probs, dim=1).values
    hard_pred = p > threshold

    return (data_out == hard_pred).float().mean()

def metric_hard_f1_align(model, data_in, data_out, threshold:float):
    A_given_B = data_in[::2]
    B_given_A = data_in[1::2,:]
    data_out = data_out[::2]

    align_probs = torch.stack((model(A_given_B)["P(A|B)"], model(B_given_A)["P(A|B)"]), dim=1)
    p = torch.min(align_probs, dim=1).values
    hard_pred = p > threshold

    true_pos = data_out[hard_pred==1].sum()
    total_pred_pos = (hard_pred==1).sum().float()
    total_actual_pos = data_out.sum().float()

    precision = true_pos / total_pred_pos
    recall = true_pos / total_actual_pos

    return 2 * (precision*recall) / (precision + recall)

def metric_hard_accuracy_align_mean(model, data_in, data_out, threshold):
    A_given_B = data_in[::2]
    B_given_A = data_in[1::2,:]
    data_out = data_out[::2]

    align_probs = torch.stack((model(A_given_B)["P(A|B)"], model(B_given_A)["P(A|B)"]), dim=1)
    p = torch.mean(align_probs, dim=1)
    hard_pred = p > threshold

    return (data_out == hard_pred).float().mean()

def metric_hard_f1_align_mean(model, data_in, data_out, threshold):
    A_given_B = data_in[::2]
    B_given_A = data_in[1::2,:]
    data_out = data_out[::2]

    align_probs = torch.stack((model(A_given_B)["P(A|B)"], model(B_given_A)["P(A|B)"]), dim=1)
    p = torch.mean(align_probs, dim=1)
    hard_pred = p > threshold

    true_pos = data_out[hard_pred==1].sum()
    total_pred_pos = (hard_pred==1).sum().float()
    total_actual_pos = data_out.sum().float()

    precision = true_pos / total_pred_pos
    recall = true_pos / total_actual_pos

    return 2 * (precision*recall) / (precision + recall)

def roc_plot(model, data_in, data_out):
    A_given_B = data_in[::2]
    B_given_A = data_in[1::2,:]
    data_out = data_out[::2]

    align_probs = torch.stack((model(A_given_B)["P(A|B)"], model(B_given_A)["P(A|B)"]), dim=1)
    p = torch.min(align_probs, dim=1).values

    fpr, tpr, thresholds = roc_curve(y_true=data_out.cpu(), y_score=p.cpu())

    # print(fpr.shape, tpr.shape, thresholds.shape)

    return fpr, tpr, thresholds

def pr_plot(model, data_in, data_out):
    A_given_B = data_in[::2]
    B_given_A = data_in[1::2,:]
    data_out = data_out[::2]

    align_probs = torch.stack((model(A_given_B)["P(A|B)"], model(B_given_A)["P(A|B)"]), dim=1)
    p = torch.min(align_probs, dim=1).values

    precision, recall, thresholds = precision_recall_curve(y_true=data_out.cpu(), probas_pred=p.cpu())

    return precision[0:-1], recall[0:-1], thresholds

def pct_of_align_cond_on_human_as_min(model, data_in, data_out):
    """Get the percentage of minimum alignment pairs that are conditioned on humans"""
    A_given_B = data_in[::2]
    B_given_A = data_in[1::2,:]
    data_out = data_out[::2]

    align_pair_in = torch.stack((A_given_B, B_given_A), dim=0)
    align_probs = torch.stack((model(A_given_B)["P(A|B)"], model(B_given_A)["P(A|B)"]), dim=1)

    p = torch.min(align_probs, dim=1)
    min_indices = p.indices.repeat_interleave(data_in.shape[1]).reshape(-1,data_in.shape[1])

    min_aligns = torch.gather(align_pair_in, dim=0, index=min_indices.view(1, align_pair_in.shape[1] ,-1)).squeeze(0)

    max_mouse_idx = 2737
    num_cond_on_human_as_min = (min_aligns[:,1] > max_mouse_idx).sum().float()
    
    return num_cond_on_human_as_min / align_pair_in.shape[1]


def get_probabilities(model, data_in, data_out):
    A_given_B = data_in[::2]
    B_given_A = data_in[1::2,:]
    data_out = data_out[::2]

    align_probs = torch.stack((model(A_given_B)["P(A|B)"], model(B_given_A)["P(A|B)"]), dim=1)

    return align_probs[:,0].cpu().numpy(), align_probs[:,1].cpu().numpy(), torch.min(align_probs, dim=1).values.cpu().numpy()