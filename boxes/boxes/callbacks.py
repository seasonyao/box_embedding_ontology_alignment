from __future__ import annotations
from .box_operations import *
from learner import Callback
from dataclasses import dataclass, field
import torch
from typing import *

if TYPE_CHECKING:
    from learner import Learner, Recorder


@dataclass
class MinBoxSize(Callback):
    """Prevents boxes from getting too small during training.
    """
    min_vol: float = 1e-6
    recorder: Union[None, Recorder] = None
    name: str = "Small Boxes"
    eps: float = 1e-6 # added to the size of the boxes due to floating point precision

    def learner_post_init(self, learner: Learner):
        if self.recorder is None:
            self.recorder = learner.recorder

    def batch_end(self, l: Learner):
        with torch.no_grad():
            boxes = l.model.box_embedding.boxes
            small_boxes = detect_small_boxes(boxes, l.model.vol_func, self.min_vol)
            num_min_boxes = small_boxes.sum().detach().cpu().item()
            if self.recorder is not None:
                self.recorder.update_({self.name + f" (<{self.min_vol} before MinBoxSize)": num_min_boxes}, l.progress.partial_epoch_progress())
                self.recorder.update_({self.name + f" (<{self.min_vol - self.eps} before MinBoxSize)": detect_small_boxes(boxes, l.model.vol_func, self.min_vol - self.eps).sum()}, l.progress.partial_epoch_progress())
            if num_min_boxes > 0:
                replace_Z_by_cube_(boxes, small_boxes, self.min_vol + self.eps)
            small_boxes = detect_small_boxes(boxes, l.model.vol_func, self.min_vol)
            num_min_boxes = small_boxes.sum().detach().cpu().item()
            if self.recorder is not None:
                self.recorder.update_({self.name + f" (<{self.min_vol} after MinBoxSize)": num_min_boxes}, l.progress.partial_epoch_progress())
                self.recorder.update_({self.name + f" (<{self.min_vol - self.eps} after MinBoxSize)": detect_small_boxes(boxes, l.model.vol_func, self.min_vol - self.eps).sum()}, l.progress.partial_epoch_progress())

@dataclass
class RandomNegativeSampling(Callback):
    """
    Given a pair (u,v) generates random pairs of the form (u,x) or (y,v)
    """
    num_entities: int
    ratio: int
    
    def batch_begin(self, l: Learner):
        with torch.no_grad():            
            batch_in = l.batch_in.to("cpu") # Should be a tensor of indices, shape (batch_size, k)
            batch_size, k = batch_in.shape
            num_neg_samples = batch_in.shape[0] * self.ratio
            negative_samples = batch_in.repeat(self.ratio, 1) # shape (batch_size * ratio, k)
            negative_probs = torch.zeros(num_neg_samples).to(l.batch_out.device)
            negative_samples.scatter_(1, torch.randint(k,(num_neg_samples,1)), torch.randint(self.num_entities, (num_neg_samples,1)))
            negative_samples = negative_samples.to(l.batch_in.device)
            l.batch_in = torch.cat((l.batch_in, negative_samples), 0)
            l.batch_out = torch.cat((l.batch_out, negative_probs), 0)
    