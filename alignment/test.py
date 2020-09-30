from boxes import *
from learner import *
import math

PATH = 'data/ontologies/'
# Data in unary.tsv are probabilites separated by newlines. The probability on line n is P(n), where n is the id assigned to the nth element.
unary_prob = torch.from_numpy(np.loadtxt(f'{PATH}unary.tsv')).float().to("cuda")
num_boxes = unary_prob.shape[0]

# We're going to use random negative sampling during training, so no need to include negatives in our training data itself
train = Probs.load_from_julia(PATH, 'tr_pos.tsv', 'tr_neg.tsv', ratio_neg = 0).to("cuda")

# The dev set will have a fixed set of negatives, however.
dev = Probs.load_from_julia(PATH, 'dev_pos.tsv', 'dev_neg.tsv', ratio_neg = 1).to("cuda")

mouse_eval = Probs.load_from_julia(PATH, 'human_dev_pos.tsv', 'human_dev_neg.tsv', ratio_neg = 1).to("cuda")
human_eval = Probs.load_from_julia(PATH, 'mouse_dev_pos.tsv', 'mouse_dev_neg.tsv', ratio_neg = 1).to("cuda")

box_model = BoxModel(
    BoxParamType=MinMaxSigmoidBoxes,
    vol_func=soft_volume,
    num_models=1,
    num_boxes=num_boxes,
    dims=50,
    method="orig").to("cuda")

train_dl = TensorDataLoader(train, batch_size=2**6, shuffle=True)

opt = torch.optim.Adam(box_model.parameters(), lr=1e-2)

def mean_cond_kl_loss(model_out: ModelOutput, target: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    return kl_div_sym(model_out["P(A|B)"], target, eps).mean()

# See boxes/loss_functions.py file for more options. Note that you may have to changed them to fit your use case.
# Also note that "kl_div_sym" is just binary cross-entropy.

# For this dataset we had unary probabilities as well as conditional probabilities. Our loss function will be a sum of these, which is provided by the following loss function wrapper:
loss_func = LossPieces(mean_cond_kl_loss, (1e-2, mean_unary_kl_loss(unary_prob)))

metrics = [metric_hard_accuracy, metric_hard_f1]

rec_col = RecorderCollection()

callbacks = CallbackCollection(
    LossCallback(rec_col.train, train),
    LossCallback(rec_col.dev, dev),
    *(MetricCallback(rec_col.dev, dev, m) for m in metrics),
    *(MetricCallback(rec_col.train, train, m) for m in metrics),
    *(TestMetricCallback(rec_col.onto, human_eval, m) for m in metrics),
    *(TestMetricCallback(rec_col.onto, mouse_eval, m) for m in metrics),
    MetricCallback(rec_col.dev, dev, metric_pearson_r),
    MetricCallback(rec_col.train, dev, metric_spearman_r),
    PercentIncreaseEarlyStopping(rec_col.dev, "mean_cond_kl_loss", 0.25, 10),
    PercentIncreaseEarlyStopping(rec_col.dev, "mean_cond_kl_loss", 0.5),
#     GradientClipping(-1000,1000),
    RandomNegativeSampling(num_boxes, 1),
    StopIfNaN(),
)

l = Learner(train_dl, box_model, loss_func, opt, callbacks, recorder = rec_col.learn)

l.train(2)

l.evaluate()

print(rec_col)
rec_col.onto

# box_model.forward(torch.LongTensor([[1,1]]))
