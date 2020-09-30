from __future__ import annotations
from .exceptions import *
from typing import *
from dataclasses import dataclass, field
import torch
import wandb
from . import Callback

if TYPE_CHECKING:
    from . import Learner, Recorder

@dataclass
class WandbEpochLogger(Callback):
    recorder: Recorder
    prefix: Union[str, None] = None
    commit: bool = True
    log_epoch: bool = True

    @torch.no_grad()
    def epoch_end(self, learner: Learner):
        dict_to_log = self.recorder.last_update()
        dict_to_log = {self.prefix + k: v for k,v in dict_to_log.items()}
        if self.log_epoch:
            dict_to_log['epoch'] = dict_to_log.pop(self.prefix+'index') # rename this
        else:
            del dict_to_log[self.prefix+'index']
        wandb.log(dict_to_log, self.commit)
