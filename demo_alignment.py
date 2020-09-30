#!/bin/env python

from boxes import *
from learner import *
import math
import matplotlib.pyplot as plt
import os

use_wandb = True

PATH = '/mnt/nfs/work1/llcao/zonghaiyao/ontology_alignment/box_for_ontology_alignment/data/ontologies/anatomy/'

nEpochs = 50
ats = 0.8                       # aligment training split
Transitive_Closure = False      # Transitive closure
box_type_dict = {'MinMaxSigmoidBoxes': MinMaxSigmoidBoxes,
                 'SigmoidBoxes': SigmoidBoxes,
                 'DeltaBoxes': DeltaBoxes,
                 'MinMaxBoxes': MinMaxBoxes}

# make sure codes work whether or not using wandb
if use_wandb:
    import wandb

    # Default values for hyper-parameters we're going to sweep over
    config_defaults = {
        'dims': 10,
        'lr': 5e-1,
        'rns_ratio': 1,
        'box_type': 'MinMaxSigmoidBoxes',
        'mouse_human_eps': 0.05,
        'alignment_eps': 0.05
    }


    # Initialize a new wandb run
    wandb.init(project="ontology-alignment", config=config_defaults, dir="/mnt/nfs/work1/llcao/zonghaiyao/ontology_alignment/wandb")

    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config
else:
    class Config():
        def __init__(self):
            self.dims = 10
            self.lr = 5e-1
            self.rns_ratio = 1
            self.box_type = 'MinMaxSigmoidBoxes'
            self.mouse_human_eps = 0.05
            self.alignment_eps = 0.05
    config = Config()


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

#print("is use cuda?", use_cuda)


def mean_cond_kl_loss(model_out: ModelOutput, target: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    return kl_div_sym(model_out["P(A|B)"], target, eps).mean()

def human_cond_kl_loss(model_out: ModelOutput, target: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    return kl_div_sym(model_out["P(A|B)"], target, eps).mean()

def mouse_cond_kl_loss(model_out: ModelOutput, target: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    return kl_div_sym(model_out["P(A|B)"], target, eps).mean()

def align_cond_kl_loss(model_out: ModelOutput, target: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    return kl_div_sym(model_out["P(A|B)"], target, eps).mean()

# See boxes/loss_functions.py file for more options. Note that you may have to changed them to fit your use case.
# Also note that "kl_div_sym" is just binary cross-entropy.


if Transitive_Closure:
    tc = "tc_"
else:
    tc = ""

# Data in unary.tsv are probabilites separated by newlines. The probability on line n is P(n), where n is the id assigned to the nth element.
unary_prob = torch.from_numpy(np.loadtxt(f'{PATH}unary/unary.tsv')).float().to(device)
num_boxes = unary_prob.shape[0]

# We're going to use random negative sampling during training, so no need to include negatives in our training data itself
train = Probs.load_from_julia(PATH, f'tr_pos_{tc}{ats}.tsv', f'tr_neg_{ats}.tsv', ratio_neg = 0).to(device)

# The dev set will have a fixed set of negatives, however.
dev = Probs.load_from_julia(PATH, f'dev_align_pos_{ats}.tsv', f'dev_align_neg_{ats}.tsv', ratio_neg = 1).to(device)

# This set is used just for evaluation purposes after training
tr_align = Probs.load_from_julia(PATH, f'tr_align_pos_{ats}.tsv', f'tr_align_neg_{ats}.tsv', ratio_neg = 1).to(device)

mouse_eval = Probs.load_from_julia(PATH, 'human_dev_pos.tsv', 'human_dev_neg.tsv', ratio_neg = 1).to(device)
human_eval = Probs.load_from_julia(PATH, 'mouse_dev_pos.tsv', 'mouse_dev_neg.tsv', ratio_neg = 1).to(device)



box_model = BoxModel(
    BoxParamType=box_type_dict[config.box_type],
    vol_func=soft_volume,
    num_models=1,
    num_boxes=num_boxes,
    dims=config.dims,
    method="orig").to(device)

#### IF YOU ARE LOADING FROM JULIA WITH ratio_neg=0, train_dl WILL ONLY CONTAIN POSITIVE EXAMPLES
#### THIS MEANS YOUR MODEL SHOULD USE NEGATIVE SAMPLING DURING TRAINING
train_dl = TensorDataLoader(train, batch_size=2**6, shuffle=True)

mouse_dl = TensorDataLoader(mouse_eval, batch_size=2**6)
human_dl = TensorDataLoader(human_eval, batch_size=2**6)

eval_dl = [mouse_dl, human_dl]

opt = torch.optim.Adam(box_model.parameters(), lr=config.lr)


loss_func = LossPieces( (config.mouse_human_eps,mouse_cond_kl_loss), (config.mouse_human_eps,human_cond_kl_loss), (config.alignment_eps,align_cond_kl_loss))

metrics = [metric_hard_accuracy, metric_hard_f1]
align_metrics = [metric_hard_accuracy_align, metric_hard_f1_align, metric_hard_accuracy_align_mean, metric_hard_f1_align_mean]

rec_col = RecorderCollection()

callbacks = CallbackCollection(
    LossCallback(rec_col.train, train),
    LossCallback(rec_col.dev, dev),
    *(MetricCallback(rec_col.dev, dev, "dev", m, use_wandb) for m in metrics),
    *(MetricCallback(rec_col.train, train, "train", m, use_wandb) for m in metrics),
    *(MetricCallback(rec_col.onto, human_eval, "human", m, use_wandb) for m in metrics),
    *(MetricCallback(rec_col.onto, mouse_eval, "mouse", m, use_wandb) for m in metrics),
    *(EvalAlignment(rec_col.train_align, tr_align, "train_align", m, use_wandb) for m in align_metrics),
    *(EvalAlignment(rec_col.dev_align, dev, "dev_align", m, use_wandb) for m in align_metrics),
    JustGiveMeTheData(rec_col.probs, dev, get_probabilities),
    BiasMetric(rec_col.bias, dev, pct_of_align_cond_on_human_as_min),
    PlotMetrics(rec_col.dev_roc_plot, dev, roc_plot),
    PlotMetrics(rec_col.dev_pr_plot, dev, pr_plot),
    PlotMetrics(rec_col.tr_roc_plot, tr_align, roc_plot),
    PlotMetrics(rec_col.tr_pr_plot, tr_align, pr_plot),
    MetricCallback(rec_col.train, train, 'train', metric_pearson_r, use_wandb),
    MetricCallback(rec_col.train, train, 'train', metric_spearman_r, use_wandb),
    MetricCallback(rec_col.dev, dev, 'dev', metric_pearson_r, use_wandb),
    MetricCallback(rec_col.dev, dev, 'dev', metric_spearman_r, use_wandb),
#     PercentIncreaseEarlyStopping(rec_col.dev, "mean_cond_kl_loss", 0.25, 10),
#     PercentIncreaseEarlyStopping(rec_col.dev, "mean_cond_kl_loss", 0.5),
#     PercentIncreaseEarlyStopping(rec_col.dev, "mouse_cond_kl_loss", 0.25, 10),
#     PercentIncreaseEarlyStopping(rec_col.dev, "mouse_cond_kl_loss", 0.5),
#     GradientClipping(-1000,1000),
    RandomNegativeSampling(num_boxes, config.rns_ratio),
    StopIfNaN(),
)

# l = Learner(train_dl, box_model, loss_func, opt, callbacks, recorder = rec_col.learn)
l = Learner(train_dl, box_model, loss_func, opt, callbacks, recorder = rec_col.learn, categories=True, use_wandb = use_wandb)

l.train(nEpochs)
#--------------------
if use_wandb:
    wandb.watch(box_model)
#--------------------


print('Training complete!')
# print('==================')
# print('Train:')
# pprint.pprint(rec_col.train.last_update(), indent=4)
# print('==================')
# print('Dev:')
# pprint.pprint(rec_col.dev.last_update(), indent=4)