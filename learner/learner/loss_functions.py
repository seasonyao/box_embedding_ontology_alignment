import torch
from torch import Tensor
from torch.nn import Module
from typing import *
from .learner import Learner, Recorder
from collections import OrderedDict
import wandb

FuncInput = Union[OrderedDict, dict, Collection[Union[Callable, Tuple[float, Callable]]]]

def func_list_to_dict(*func_list: FuncInput) -> Dict[str, Callable]:
    """

    :param func_list: List of functions or tuples (weight, function)
    :type func_list: Union[Collection[Union[Callable, Tuple[float, Callable]]], Dict[str, Callable]]

    :return: Ordered Dictionary of {name: function}
    :rtype: OrderedDict[str, Callable]
    """
    if type(func_list) == OrderedDict or type(func_list) == dict:
        return func_list
    func_dict = OrderedDict()
    for f in func_list:
        if type(f) is tuple:
            func_dict[f"{f[0]}*{f[1].__name__}"] = lambda *args, weight=f[0], func=f[1], **kwargs: weight * func(*args, **kwargs)
        else:
            func_dict[f.__name__] = f
    return func_dict

def unweighted_func_dict(*func_list: FuncInput) -> Dict[str, Callable]:
    if type(func_list) == OrderedDict or type(func_list) == dict:
        return func_list
    func_dict = OrderedDict()
    for f in func_list:
        if type(f) is tuple:
            f = f[1]
        func_dict[f.__name__] = f
    return func_dict

def isnan(x):
    return (x != x)

class LossPieces(Module):

    def __init__(self, *loss_funcs: FuncInput):
        """

        :param functions: List of functions or tuples (weight, function)
        :type functions: Collection[Union[Callable, Tuple[float, Callable]]]
        """
        super().__init__()
        self.unweighted_funcs = unweighted_func_dict(*loss_funcs)
        self.loss_funcs = func_list_to_dict(*loss_funcs)

    def forward(self, model_out, true_out: Tensor,
                learner: Optional[Learner] = None,
                recorder: Optional[Recorder] = None,
                weighted: bool = True,
                categories: bool = False,
                use_wandb: bool = False,
                **kwargs) -> Tensor:
        """
        Weighted sum of all loss functions. Tracks values in Recorder.
        """
        if weighted:
            loss_funcs = self.loss_funcs
        else:
            loss_funcs = self.unweighted_funcs
        
        grad_status = torch.is_grad_enabled()
        if learner is None:
            torch.set_grad_enabled(False)
        
        # change the loss function for margin loss
        try:
            if categories:
                loss_pieces = {}
                for k,l in loss_funcs.items():
                    if 'mouse' in k:
                        loss_pieces['mouse_cond_kl_loss'] = l(model_out[0], true_out[0])
                    elif 'human' in k:
                        loss_pieces['human_cond_kl_loss'] = l(model_out[1], true_out[1])
                    else:
                        loss_pieces['align_cond_kl_loss'] = l(model_out[2], true_out[2])
#                     if 'mouse' in k:
#                         loss_pieces[k] = l(model_out[0], true_out[0])
#                     elif 'human' in k:
#                         loss_pieces[k] = l(model_out[1], true_out[1])
#                     else:
#                         loss_pieces[k] = l(model_out[2], true_out[2])
                if use_wandb:
                    # Log metrics inside your training loop
                    metrics = {'mouse_loss': torch.sum(loss_pieces['mouse_cond_kl_loss']), 
                               'human_loss': torch.sum(loss_pieces['human_cond_kl_loss']), 
                               'alignmnt_loss': torch.sum(loss_pieces['align_cond_kl_loss'])}
                    wandb.log(metrics)
                
                loss = 0
                for key,val in loss_pieces.items():
                    if not isnan(val):
                        loss += val
                        

            else:
                loss_pieces = {k: l(model_out, true_out) for k, l in loss_funcs.items()}
                loss = sum(loss_pieces.values())
          

            loss_pieces['loss'] = loss
            if learner is not None:
                if recorder is not None:
                    recorder.update_(loss_pieces, learner.progress.partial_epoch_progress())
                else:
                    self.recorder.update_(loss_pieces, learner.progress.partial_epoch_progress())
        finally:
            torch.set_grad_enabled(grad_status)
        return loss
