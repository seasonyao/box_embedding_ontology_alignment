from __future__ import annotations
import torch
from torch.utils.data import TensorDataset
import itertools
import numpy as np


######################################
# Data Objects
######################################

def load_numpy_num_lines(path, num = "all", dtype=np.float):
    with open(path) as f:
        if num != "all":
            f = itertools.islice(f, 0, num)
        n = np.loadtxt(f, dtype=dtype)
    return n


class Probs(TensorDataset):
    """Pairwise Probability dataset"""

    def __init__(self, ids, probs):
        super().__init__(ids, probs)
        self.ids = ids
        self.probs = probs

    def to(self, *args, **kwargs):
        self.ids = self.ids.to(*args, **kwargs)
        self.probs = self.probs.to(*args, **kwargs)
        # Now we need to make sure the self.tensors attribute points to the right versions.
        # The easiest way is to just reinitialize.
        super().__init__(self.ids, self.probs)
        return self

    @classmethod
    def load_from_tsv(cls, path: str, filename: str) -> Probs:
        ids = torch.from_numpy(np.loadtxt(f'{path}{filename}', skiprows=1, usecols=[0, 1], dtype=int))
        probs = torch.from_numpy(np.loadtxt(f'{path}{filename}', skiprows=1, usecols=[2])).float()
        return cls(ids, probs)

    @classmethod
    def load_triples_from_tsv(cls, filename:str, num="all", neg=False) -> Probs:
        ids = torch.from_numpy(load_numpy_num_lines(f"{filename}_ids.tsv", num, dtype=np.int))
        if neg:
            probs = torch.zeros(ids.shape[0])
        else:
            probs = torch.from_numpy(load_numpy_num_lines(f"{filename}_probs.tsv", num)).float()
        return cls(ids, probs)


    @classmethod
    def load_from_julia(cls, path, pos_name, neg_name, weight_name=None, num_pos = "all", num_neg = "all", ratio_neg = None):
        pos_ids = load_numpy_num_lines(f"{path}{pos_name}", num_pos, dtype=np.int64)
        num_pos = pos_ids.shape[0]
        if ratio_neg is not None:
            num_neg = num_pos * ratio_neg
        if num_neg > 0:
            neg_ids = load_numpy_num_lines(f"{path}{neg_name}", num_neg, dtype=np.int64)
            num_neg = neg_ids.shape[0]
            ids = torch.from_numpy(np.concatenate((pos_ids, neg_ids)))
        else:
            ids = torch.from_numpy(pos_ids)
        probs = np.zeros(ids.shape[0])
        if weight_name is not None:
            probs[0:num_pos] = load_numpy_num_lines(f"{path}{weight_name}", num_pos, dtype=np.float)
        else:
            probs[0:num_pos] = 1
        probs = torch.from_numpy(probs).float()
        return cls(ids, probs)
