from __future__ import annotations
from .exceptions import *
import torch
from torch.utils.data import Dataset
from typing import *
from dataclasses import dataclass, field
import wandb

try:
    from IPython import get_ipython
    if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
        raise ImportError("console")
except:
    pass
else:
    import ipywidgets as widgets
    from IPython.core.display import HTML, display

if TYPE_CHECKING:
    from learner import Learner, Recorder


class Callback:
    def learner_post_init(self, learner: Learner):
        pass

    def train_begin(self, learner: Learner):
        pass

    def epoch_begin(self, learner: Learner):
        pass

    def batch_begin(self, learner: Learner):
        pass

    def backward_end(self, learner: Learner):
        pass

    def batch_end(self, learner: Learner):
        pass

    def epoch_end(self, learner: Learner):
        pass

    def train_end(self, learner: Learner):
        pass

    def eval_align(self, learner: Learner, threshold:float):
        pass

    def metric_plots(self, l: Learner):
        pass

    def eval_end(self, l: Learner):
        pass

    def bias_metric(self, l: Learner):
        pass

class CallbackCollection:

    def __init__(self, *callbacks: Callback):
        self._callbacks = callbacks

    def __call__(self, action: str, *args, **kwargs):
        for c in self._callbacks:
            getattr(c, action)(*args, **kwargs)

    def __getattr__(self, action: str):
        return lambda *args, **kwargs: self.__call__(action, *args, **kwargs)


@dataclass
class GradientClipping(Callback):
    min: float = None
    max: float = None

    def backward_end(self, learner: Learner):
        for param in learner.model.parameters():
            if param.grad is not None:
                param.grad = param.grad.clamp(self.min, self.max)


@dataclass
class LossCallback(Callback):
    recorder: Recorder
    ds: Dataset
    weighted: bool = True

    @torch.no_grad()
    def train_begin(self, learner: Learner):
        self.epoch_end(learner)

    @torch.no_grad()
    def epoch_end(self, l: Learner):
        data_in, data_out = self.ds[:]
        if l.categories:
            split_in, split_out = l.split_data(data_in, data_out, split=2737)
            model_pred = [l.model(item) if len(item)>0 else {'P(A|B)':l.TensorNaN(device=data_in.device)} for item in split_in]
            l.loss_fn(model_pred, split_out, l, self.recorder, weighted=self.weighted, categories=True)  
        else:
            output = l.model(data_in)
            l.loss_fn(output, data_out, l, self.recorder, weighted=self.weighted) # this logs the data to the recorder


@dataclass
class MetricCallback(Callback):
    recorder: Recorder
    ds: Dataset
    data_categories: str
    metric: Callable
    use_wandb: bool = False
    name: Union[str, None] = None

    def __post_init__(self):
        if self.name is None:
            self.name = self.metric.__name__
        self.name = self.recorder.get_unique_name(self.name)

    @torch.no_grad()
    def train_begin(self, learner: Learner):
        self.epoch_end(learner)

    @torch.no_grad()
    def epoch_end(self, l: Learner):
        data_in, data_out = self.ds[:]
        metric_val = self.metric(l.model, data_in, data_out)
        self.recorder.update_({self.name: metric_val}, l.progress.current_epoch_iter)
        
        print("evaluation_" + self.data_categories + "_" + self.name, str(metric_val))
        
        if self.use_wandb:
            metric_name = "evaluation_" + self.data_categories + "_" + self.name
            wandb.log({metric_name: metric_val})

@dataclass
class EvalAlignment(Callback):
    recorder: Recorder
    ds: Dataset
    data_categories: str
    metric: callable
    use_wandb: bool = False
    name: Union[str, None] = None
        

    def __post_init__(self):
        if self.name is None:
            self.name = self.metric.__name__
        self.name = self.recorder.get_unique_name(self.name)

    @torch.no_grad()
    def eval_align(self, l: Learner, threshold: float):
        data_in, data_out = self.ds[:]
        metric_val = self.metric(l.model, data_in, data_out, threshold)
        self.recorder.update_({self.name: metric_val}, threshold)
        
        print("align_evaluation_" + self.data_categories + "_" + str(threshold) + "_" + self.name, str(metric_val))
        
        if self.use_wandb:
            metric_name = "align_evaluation_" + self.data_categories + "_" + str(threshold) + "_" + self.name
            wandb.log({metric_name: metric_val})
        


@dataclass
class JustGiveMeTheData(Callback):
    recorder: Recorder
    ds: Dataset
    metric: Callable
    
    @torch.no_grad()
    def eval_end(self, l: Learner):
        data_in, data_out = self.ds[:]
        align1, align2, min_prob = self.metric(l.model, data_in, data_out)
        self.recorder.update_({"Alignment 1 Probablity": align1}, [i for i in range(align1.shape[0])] )
        self.recorder.update_({"Alignment 2 Probablity": align2}, [i for i in range(align2.shape[0])] )
        self.recorder.update_({"Minimum Probablity": min_prob}, [i for i in range(min_prob.shape[0])] )
        

@dataclass
class BiasMetric(Callback):
    recorder: Recorder
    ds: Dataset
    metric: Callable
    name: Union[str, None] = None

    def __post_init__(self):
        if self.name is None:
            self.name = self.metric.__name__
        self.name = self.recorder.get_unique_name(self.name)

    @torch.no_grad()
    def bias_metric(self, l: Learner):
        data_in, data_out = self.ds[:]
        metric_val = self.metric(l.model, data_in, data_out)
        self.recorder.update_({self.name: metric_val}, 0)

@dataclass
class PlotMetrics(Callback):
    recorder: Recorder
    ds: Dataset
    metric: callable

    @torch.no_grad()
    def metric_plots(self, l: Learner):
        data_in, data_out = self.ds[:]
        metric_1, metric_2, thresholds = self.metric(l.model, data_in, data_out)
        
        if self.metric.__name__ == 'roc_plot':
            metric_1_name = "fpr"
            metric_2_name = "tpr"
        else:
            metric_1_name = "precision"
            metric_2_name = "recall"

        self.recorder.update_({metric_1_name: metric_1}, thresholds)
        self.recorder.update_({metric_2_name: metric_2}, thresholds)


@dataclass
class DisplayTable(Callback):
    recorder: Union[Recorder, None] = None

    def learner_post_init(self, learner: Learner):
        if self.recorder is None:
            self.recorder = learner.recorder

    @torch.no_grad()
    def train_begin(self, learner: Learner):
        self.out = widgets.Output()

    @torch.no_grad()
    def epoch_end(self, learner: Learner):
        self.out.clear_output()
        with self.out:
            display(self.recorder)


@dataclass
class StopAtMaxLoss(Callback):
    max_loss: float = 100.

    @torch.no_grad()
    def batch_end(self, learner: Learner):
        if learner.loss > self.max_loss:
            raise MaxLoss(learner.loss, self.max_loss)


@dataclass
class PercentIncreaseEarlyStopping(Callback):
    rec: Recorder
    metric_name: str
    percent_inc: float
    epoch_count: int = 0
    flag: Optional[str] = None

    def __post_init__(self):
        if self.epoch_count == 0:
            self.epoch_end = self._epoch_end_percent_only
        else:
            self.epoch_end = self._epoch_end_both

    @torch.no_grad()
    def _epoch_end_percent_only(self, learner: Learner):
        vals = self.rec[self.metric_name]
        min_val = vals.min()
        cur_val = next(iter(vals.tail(1)), min_val)
        if  cur_val > (1 + self.percent_inc) * min_val:
            if self.flag is not None:
                self.rec.update_({self.flag: True}, vals.tail(1).index.item())
            else:
                raise EarlyStopping(f"{self.metric_name} is now {cur_val}, which is more than {1 + self.percent_inc} times it's minimum of {min_val}.")

    @torch.no_grad()
    def _epoch_end_both(self, learner: Learner):
        vals = self.rec[self.metric_name]
        min_idx = vals.idxmin()
        cur_idx = next(iter(vals.tail(1).index), min_idx)
        if cur_idx >= min_idx + self.epoch_count and vals[cur_idx] > (1 + self.percent_inc) * vals[min_idx]:
            if self.flag is not None:
                self.rec.update_({self.flag: True}, cur_idx)
            else:
                raise EarlyStopping(f"{self.metric_name} is now {vals[cur_idx]}, which is more than {1 + self.percent_inc} times it's minimum of {vals[min_idx]}, which occurred {cur_idx - min_idx} >= {self.epoch_count} epochs ago.")


@dataclass
class ModelHistory(Callback):
    state_dict: List[dict] = field(default_factory=list)

    @torch.no_grad()
    def batch_end(self, learner: Learner):
        self.state_dict.append({k: v.detach().cpu().clone() for k, v in learner.model.state_dict().items()})

    def __getitem__(self, item):
        return self.state_dict[item]


class StopIfNaN(Callback):

    @torch.no_grad()
    def backward_end(self, learner: Learner):
        for param in learner.model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                raise NaNsException("gradients")

    @torch.no_grad()
    def batch_end(self, learner: Learner):
        for param in learner.model.parameters():
            if torch.isnan(param).any():
                raise NaNsException("parameters")
