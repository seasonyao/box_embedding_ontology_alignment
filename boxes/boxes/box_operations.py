import torch
from torch import Tensor # for type annotations
import torch.nn.functional as F
from typing import *


def intersection(A: Tensor, B: Tensor) -> Tensor:
    """
    :param A: Tensor(..., zZ, dim)
    :param B: Tensor(..., zZ, dim)
    :return: Tensor(..., zZ, dim), box embeddings for A intersect B
    """
    z = torch.max(A[...,0,:], B[...,0,:])
    Z = torch.min(A[...,1,:], B[...,1,:])
    return torch.stack((z, Z), dim=-2)

def neg_edge_adjustment(A: Tensor) -> Tensor:
    """
    :param A: Tensor(..., zZ, dim)
    
    Replace "negative" edges with their mean.
    
    (TODO: optimize this)
    """
    center_of_meet = torch.mean(A, dim=-2)
    neg_edges_mask = ((A[...,1,:] - A[...,0,:]) >= 0)
    neg_edges_mask_stack = torch.stack((neg_edges_mask, neg_edges_mask), dim=-2)
    center_of_meet_stack = torch.stack((center_of_meet, center_of_meet), dim=-2)
    return torch.where(neg_edges_mask_stack, A, center_of_meet_stack)

def join(A: Tensor, B: Tensor) -> Tensor:
    """
    :param A: Tensor(model, pair, zZ, dim)
    :param B: Tensor(model, pair, zZ, dim)
    :return: Tensor(model, pair, zZ, dim), box embeddings for the smallest box which contains A and B
    """
    z = torch.min(A[:,:,0], B[:,:,0])
    Z = torch.max(A[:,:,1], B[:,:,1])
    return torch.stack((z, Z), dim=2)


def clamp_volume(boxes: Tensor) -> Tensor:
    """
    :param boxes: Tensor(... zZ, dim)
    :return: Tensor(...) of volumes
    """
    return torch.prod((boxes[...,1,:] - boxes[...,0,:]).clamp_min(0), dim=-1)


def soft_volume(boxes: Tensor) -> Tensor:
    """
    :param sidelengths: Tensor(model, box, dim)
    :return: Tensor(model, box) of volumes
    """
    return torch.prod(F.softplus(boxes[:,:,1] - boxes[:,:,0]), dim=-1)


def log_clamp_volume(boxes: Tensor, eps:float = torch.finfo(torch.float32).tiny) -> Tensor:
    """
    :param boxes: Tensor(model, box, zZ, dim)
    :return: Tensor(model, box) of volumes
    """
    return torch.sum(torch.log((boxes[:,:,1] - boxes[:,:,0]).clamp_min(0) + eps), dim=-1)


def log_soft_volume(boxes: Tensor, eps:float = torch.finfo(torch.float32).tiny) -> Tensor:
    """
    :param sidelengths: Tensor(model, box, dim)
    :return: Tensor(model, box) of volumes
    """
    return torch.sum(torch.log(F.softplus(boxes[:,:,1] - boxes[:,:,0]) + eps), dim=-1)


def smallest_containing_box(boxes: Tensor) -> Tensor:
    """
    Returns the smallest box which contains all boxes in `boxes`.
    
    :param boxes: Box embedding of shape (model, box, zZ, dim)
    :return: Tensor of shape (model, 1, zZ, dim)
    """
    z = boxes[:,:,0]
    Z = boxes[:,:,1]
    min_z, _ = torch.min(z, dim=1, keepdim=True)
    max_Z, _ = torch.max(Z, dim=1, keepdim=True)
    return torch.stack((min_z, max_Z), dim=2)

def smallest_containing_box_outside_unit_cube(boxes: Tensor) -> Tensor:
    """
    Returns the smallest box which contains all boxes in `boxes` and the unit cube.

    :param boxes: Box embedding of shape (model, box, zZ, dim)
    :return: Tensor of shape (model, 1, zZ, dim)
    """
    z = boxes[:,:,0]
    Z = boxes[:,:,1]
    min_z, _ = torch.min(z, dim=1, keepdim=True)
    max_Z, _ = torch.max(Z, dim=1, keepdim=True)
    min_z = min_z.clamp_max(0)
    max_Z = max_Z.clamp_min(1)
    return torch.stack((min_z, max_Z), dim=2)


def detect_small_boxes(boxes: Tensor, vol_func: Callable = clamp_volume, min_vol: float = 1e-20) -> Tensor:
    """
    Returns the indices of boxes with volume smaller than eps.

    :param boxes: box parametrization as Tensor(model, box, z/Z, dim)
    :param vol_func: function taking in side lengths and returning volumes
    :param min_vol: minimum volume of boxes
    :return: masked tensor which selects boxes whose side lengths are less than min_vol
    """
    return vol_func(boxes) < min_vol


def replace_Z_by_cube(boxes: Tensor, indices: Tensor, cube_vol: float = 1e-20) -> Tensor:
    """
    Returns a new Z parameter for boxes for which those boxes[indices] are now replaced by cubes of size cube_vol

    :param boxes: box parametrization as Tensor(model, box, z/Z, dim)
    :param indices: box indices to replace by a cube
    :param cube_vol: volume of cube
    :return: tensor representing the Z parameter
    """
    return boxes[:, :, 0][indices] + cube_vol ** (1 / boxes.shape[-1])



def replace_Z_by_cube_(boxes: Tensor, indices: Tensor, cube_vol: float = 1e-20) -> Tensor:
    """
    Replaces the boxes indexed by `indices` by a cube of volume `min_vol` with the same z coordinate

    :param boxes: box parametrization as Tensor(model, box, z/Z, dim)
    :param indices: box indices to replace by a cube
    :param cube_vol: volume of cube
    :return: tensor representing the box parametrization with those boxes
    """
    boxes[:, :, 1][indices] = replace_Z_by_cube(boxes, indices, cube_vol)


def disjoint_boxes_mask(A: Tensor, B: Tensor) -> Tensor:
    """
    Returns a mask for when A and B are disjoint.
    Note: This is symmetric with respect to the arguments.
    """
    return ((B[:,:,1] <= A[:,:,0]) | (B[:,:,0] >= A[:,:,1])).any(dim=-1)


def overlapping_boxes_mask(A: Tensor, B: Tensor) -> Tensor:
    return disjoint_boxes_mask(A, B) ^ 1


def containing_boxes_mask(A: Tensor, B: Tensor) -> Tensor:
    """
    Returns a mask for when B contains A.
    Note: This is *not* symmetric with respect to it's arguments!
    """
    return ((B[:,:,1] >= A[:,:,1]) & (B[:,:,0] <= A[:,:,0])).all(dim=-1)


def needing_pull_mask(A: Tensor, B: Tensor, target_prob_B_given_A: Tensor) -> Tensor:
    return (target_prob_B_given_A != 0) & disjoint_boxes_mask(A, B)


def needing_push_mask(A: Tensor, B: Tensor, target_prob_B_given_A: Tensor) -> Tensor:
    return (target_prob_B_given_A != 1) & containing_boxes_mask(A, B)
