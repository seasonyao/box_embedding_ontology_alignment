from .callbacks import Callback
from .learner import Recorder, Learner
from typing import Optional
from dataclasses import dataclass
import neptune

@dataclass
class NeptuneCallback(Callback):
    recorder: Recorder
    metric_prefix: Optional[str] = None

    def __post_init__(self):
        if self.metric_prefix is not None:
            self.metric_prefix = self.metric_prefix + "_"
        else:
            self.metric_prefix = ""

    def epoch_end(self, learner: Learner):
        last_record = self.recorder.dataframe.tail(1).to_dict()
        for metric_name, data in last_record.items():
            for epoch, val in data.items():
                neptune.send_metric(self.metric_prefix + metric_name, x=epoch, y=val)
