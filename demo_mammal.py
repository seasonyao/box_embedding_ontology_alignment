#!/bin/env python

from boxes import *
from learner import *
import pprint
import pickle
import argparse

PATH = 'data/wordnet/mjb/rigorous_sampling/mammal_pos_0.5_neg_10_samp_uniform_num_1182/0/'

#----------------
import wandb

hyperparameter_defaults = dict(
    dims = 1,
    log_batch_size = 8, # batch size for training will be 2**LOG_BATCH_SIZE (default: 8)
    learning_rate = 1, # MinMaxSigmoidBoxes may benefit from relatively high learning rates
    unary_weight = 0.0, # weight for unary loss during training (default: 0.01)
    epochs = 10,
    )

wandb.init(config=hyperparameter_defaults)
config = wandb.config
#----------------

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# Data in unary.tsv are probabilites separated by newlines. The probability on line n is P(n), where n is the id assigned to the nth element.
unary_prob = torch.from_numpy(np.loadtxt(f'{PATH}train_tc_unary.tsv')).float().to(device)
num_boxes = unary_prob.shape[0]

# We're going to use random negative sampling during training, so no need to include negatives in our training data itself
train = Probs.load_from_julia(PATH, 'train_tc_pos.tsv', 'train_neg.tsv', ratio_neg = 0).to(device)

# The dev set will have a fixed set of negatives, however.
dev = Probs.load_from_julia(PATH, 'dev_pos.tsv', 'dev_neg.tsv', ratio_neg = 1).to(device)

box_model = BoxModel(
    BoxParamType=MinMaxSigmoidBoxes,
    vol_func=soft_volume,
    num_models=1,
    num_boxes=num_boxes,
    dims=config.dims,
    method="orig").to(device)

train_dl = TensorDataLoader(train, batch_size=2**config.log_batch_size, shuffle=True)

opt = torch.optim.Adam(box_model.parameters(), lr=config.learning_rate)

# Here we define our loss function:
def mean_cond_kl_loss(model_out: ModelOutput, target: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    return kl_div_sym(model_out["P(A|B)"], target, eps).mean()

# See boxes/loss_functions.py file for more options. Note that you may have to changed them to fit your use case.
# Also note that "kl_div_sym" is just binary cross-entropy.

# For this dataset we had unary probabilities as well as conditional probabilities. Our loss function will be a sum of these, which is provided by the following loss function wrapper:
if config.unary_weight == 0:
    loss_func = LossPieces(mean_cond_kl_loss)
else:
    loss_func = LossPieces(mean_cond_kl_loss, (config.unary_weight, mean_unary_kl_loss(unary_prob)))

metrics = [metric_hard_accuracy, metric_hard_f1]


rec_col = RecorderCollection()

callbacks = CallbackCollection(
    LossCallback(rec_col.train, train),
    LossCallback(rec_col.dev, dev),
    *(MetricCallback(rec_col.dev, dev, m) for m in metrics),
    *(MetricCallback(rec_col.train, train, m) for m in metrics),
    PercentIncreaseEarlyStopping(rec_col.dev, "mean_cond_kl_loss", 0.25, 10),
    PercentIncreaseEarlyStopping(rec_col.dev, "mean_cond_kl_loss", 0.5),
#     GradientClipping(-1000,1000),
    RandomNegativeSampling(num_boxes, 1),
    StopIfNaN(),
)

l = Learner(train_dl, box_model, loss_func, opt, callbacks, recorder = rec_col.learn)


l.train(config.epochs)

#--------------------
wandb.watch(box_model)
#--------------------


print('Training complete!')
print('==================')
print('Train:')
pprint.pprint(rec_col.train.last_update(), indent=4)
print('==================')
print('Dev:')
pprint.pprint(rec_col.dev.last_update(), indent=4)

print('Saving model as \'learned_model.pt\'...')
torch.save(box_model.state_dict(), 'learned_model.pt')
print('Saving metrics as \'recorder_collection.pkl\'...')
pickle.dump(rec_col, open('recorder_collection.pkl', 'wb'))



