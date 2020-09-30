from __future__ import annotations
from dataclasses import dataclass, field
from typing import *
import torch
from torch.nn import Module
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from .exceptions import *
from collections import defaultdict
import pandas as pd
import wandb
from .callbacks import CallbackCollection

@dataclass
class Recorder:
    _data: pd.DataFrame = field(default_factory=pd.DataFrame)
    _names: Collection[(float, str)] = field(default_factory=list)

    def update_(self, data_in: Dict[str, Any], index: Union[int, float]):
        data_no_tensors = {k: v if type(v) is not torch.Tensor else v.detach().cpu().item() for k,v in data_in.items()}
        self._data = self._data.combine_first(
            pd.DataFrame(data_no_tensors, [index])
        )

    def get_unique_name(self, name:str):
        i = 1
        while name in self._data.columns:
            name = f"{name}_{i}"
            i += 1
        self._data[name] = [] # adds this column to DataFrame
        return name

    @property
    def dataframe(self):
        return self._data

    def last_update(self):
        last_row = self._data.iloc[[-1]].to_dict('split')
        dict_to_return = {column:data for column, data in zip(last_row['columns'], last_row['data'][0])}
        dict_to_return['index'] = last_row['index'][0]
        return dict_to_return

    def __getitem__(self, item):
        return self._data.__getitem__(item)

    def _repr_html_(self):
        return self._data._repr_html_()

    def __str__(self):
        return self._data.__str__()

    def __repr__(self):
        return self._data.__repr__()


@dataclass
class RecorderCollection:
    learn:Recorder = field(default_factory=Recorder)
    train:Recorder = field(default_factory=Recorder)
    dev:Recorder = field(default_factory=Recorder)
    onto:Recorder = field(default_factory=Recorder)
    train_align:Recorder = field(default_factory=Recorder)
    dev_align:Recorder = field(default_factory=Recorder)
    dev_roc_plot:Recorder = field(default_factory=Recorder)
    dev_pr_plot:Recorder = field(default_factory=Recorder)
    tr_roc_plot:Recorder = field(default_factory=Recorder)
    tr_pr_plot:Recorder = field(default_factory=Recorder)
    probs:Recorder = field(default_factory=Recorder)
    bias:Recorder = field(default_factory=Recorder)


@dataclass
class Progress:
    current_epoch_iter: int = 0
    current_batch_iter: int = 0
    num_batches: int = 0

    def increment(self):
        self.current_batch_iter += 1
        if self.current_batch_iter == self.num_batches:
            self.current_batch_iter = 0
            self.current_epoch_iter += 1

    def percent_epoch_complete(self):
        return self.current_batch_iter / self.num_batches

    def partial_epoch_progress(self):
        return self.current_epoch_iter + self.percent_epoch_complete()


@dataclass
class Learner:
    train_dl: DataLoader
    model: Module
    loss_fn: Callable
    opt: optim.Optimizer
    callbacks: CallbackCollection = field(default_factory=CallbackCollection)
    recorder: Recorder = field(default_factory=Recorder)
    categories: bool = False
    use_wandb: bool = False
    reraise_keyboard_interrupt: bool = False
    reraise_stop_training_exceptions: bool = False

    def __post_init__(self):
        self.progress = Progress(0,0,len(self.train_dl))
        self.callbacks.learner_post_init(self)

    #the split parameter will be used to find human/mouse/align data, so you need change it when using diff dataset(index)
    def split_data(self, batch_in, batch_out, split):
        category = torch.zeros(size=(batch_in.shape[0],), dtype=int)

        batch_class = batch_in > split

        for i, (a,b) in enumerate(batch_class):
            if not a and not b:
                category[i] = 0
            elif a and b:
                category[i] = 1
            else:
                category[i] = 2

        self.mouse_in = batch_in[category == 0]
        self.human_in = batch_in[category == 1]
        self.align_in = batch_in[category == 2]

        self.mouse_out = batch_out[category == 0]
        self.human_out = batch_out[category == 1]
        self.align_out = batch_out[category == 2]

        # INPUT TO THE MODEL:
        data_in = (self.mouse_in, self.human_in, self.align_in)
        # TARGET/LABEL:
        data_out = (self.mouse_out, self.human_out, self.align_out)

        return data_in, data_out

    def TensorNaN(self, size:Union[None,List[int], Tuple[int]]=None, device=None, requires_grad:bool=True):
        if size is None:    
            return torch.tensor(float('nan'), device=device, requires_grad=requires_grad)
        else:
            return float('nan') * torch.zeros(size=size, device=device, requires_grad=requires_grad)


    def train(self, epochs, progress_bar = True):
        try:
            self.callbacks.train_begin(self)
            for epoch in trange(epochs, desc="Overall Training:", disable=not progress_bar):
                self.callbacks.epoch_begin(self)
                for iteration, batch in enumerate(tqdm(self.train_dl, desc="Current Batch:", leave=False, disable=not progress_bar)):
                    if len(batch) == 2: # KLUDGE
                        self.batch_in, self.batch_out = batch
                    else:
                        self.batch_in = batch[0]
                        self.batch_out = None
                    self.progress.increment()
                    self.callbacks.batch_begin(self)
                    self.opt.zero_grad()
                    
                    if self.categories:
                        self.data_in, self.data_out = self.split_data(self.batch_in, self.batch_out, split=2737)
                        #2737 is max mouse index
                        self.model_pred = [self.model(item) if len(item)>0 else {'P(A|B)':self.TensorNaN(device=self.batch_in.device)} for item in self.data_in]
                        self.loss = self.loss_fn(self.model_pred, self.data_out, self, self.recorder, categories=True, use_wandb=self.use_wandb)                        
                    else:
                        self.model_out = self.model(self.batch_in)
                        if self.batch_out is None:
                            self.loss = self.loss_fn(self.model_out, self, self.recorder, categories=True, use_wandb=self.use_wandb)
                        else:
                            self.loss = self.loss_fn(self.model_out, self.batch_out, self, self.recorder, categories=True, use_wandb=self.use_wandb)
                        
                    # Log metrics inside your training loop
                    if self.use_wandb:
                        metrics = {'epoch': epoch, 'loss': self.loss}
                        wandb.log(metrics)

                    # print(self.recorder.dataframe)
                    self.loss.backward()
                    self.callbacks.backward_end(self)
                    self.opt.step()
                    self.callbacks.batch_end(self)
                # print(self.recorder.dataframe)
                
                # run evaluating at the end of every epoch
                self.evaluation([0.5])
                
                
                self.callbacks.epoch_end(self)
        except StopTrainingException as e:
            print(e)
            if self.reraise_stop_training_exceptions:
                raise e
        except KeyboardInterrupt:
            print(f"Stopped training at {self.progress.partial_epoch_progress()} epochs due to keyboard interrupt.")
            if self.reraise_keyboard_interrupt:
                raise KeyboardInterrupt
        finally:
            self.callbacks.train_end(self)


    def evaluation(self, trials, progress_bar=True):
        with torch.no_grad():
            # self.callbacks.eval_begin(self)
            for t in trials:
                self.callbacks.eval_align(self, t)
            self.callbacks.metric_plots(self)
            self.callbacks.bias_metric(self)
            self.callbacks.eval_end(self)



