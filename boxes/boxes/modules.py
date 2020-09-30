from .box_operations import *
import torch
from torch import Tensor
from torch.nn import Module, Parameter
import torch.nn.functional as F


################################################
# Box Parametrization Layers
################################################
default_init_min_vol = torch.finfo(torch.float32).tiny

def initialize_boxes_in_unit_cube(shape_prefix: Tuple[int], dims:int, init_min_vol: float = default_init_min_vol,
    method = "gibbs", gibbs_iter: int = 2000, **kwargs):
    """
    Creates the Parameters used for the representation of boxes.
    Initializes boxes with a uniformly random distribution of coordinates, ensuring that each box
    contains a cube of volume larger than init_min_vol.

    :param shape_prefix: Tuple which prefixes the parameters, eg. (num_boxes, num_models) will lead to
        a Tensor of shape (num_boxes, num_models, 2, dims)
    :param dims: Dimension of boxes
    :param init_min_vol: Minimum volume for boxes which are created
    :param kwargs: Unused for now, but include this for future possible parameters.
    """
    if method == "gibbs":
        sides = torch.ones(*shape_prefix, dims)
        log_min_vol = torch.log(torch.tensor(init_min_vol))
        for i in range(gibbs_iter):
            idx = torch.randint(0, dims, shape_prefix)[...,None]
            sides.scatter_(-1, idx, 1)
            complement = torch.log(sides).sum(dim=-1)
            min = torch.exp(log_min_vol - complement)[..., None]
            new_lengths = min + torch.rand(idx.shape) * (1 - min)
            sides.scatter_(-1, idx, new_lengths)

        z = torch.rand(*shape_prefix, dims) * (1-sides)
        Z = z + sides

    else:
        rand_param = lambda min, max: min + torch.rand(*shape_prefix, dims) * (max - min)
        if init_min_vol == 0:
            per_dim_min = 0
        elif init_min_vol > 0:
            per_dim_min = torch.tensor(init_min_vol).pow(1/dims)
        else:
            raise ValueError(f"init_min_vol={init_min_vol} is an invalid option.")

        z = rand_param(0, 1-per_dim_min)
        Z = rand_param(z+per_dim_min, 1)

    return torch.stack((z,Z), dim=-2)


class BoxParam(Module):
    """
    An example class for creating a box parametrization.
    Don't inherit from this, it is just an example which contains the methods for a class to be used as a BoxParam
    layer. Refer to the docstring of the functions when implementing your own BoxParam.

    Note: to avoid naming conflicts with min/max functions, we refer to the min coordinate for a box as `z`, and the
    max coordinate as `Z`.
    """

    def __init__(self, num_models:int, num_boxes:int, dim:int, **kwargs):
        """
        Creates the Parameters used for the representation of boxes.

        :param num_models: Number of models
        :param num_boxes: Number of boxes
        :param dim: Dimension
        :param kwargs: Unused for now, but include this for future possible parameters.
        """
        # Remember to call:
        super().__init__()
        raise NotImplemented


    def forward(self, box_indices = slice(None, None, None), **kwargs) -> Tensor:
        """
        Returns a Tensor representing the boxes specified by `box_indices` in the form they should be used for training.

        :param box_indices: Slice, List, or Tensor of the box indices
        :param kwargs: Unused for now, but include this for future possible parameters.
        :return: Tensor of shape (model, id, zZ, dim).
        """
        raise NotImplemented


class Boxes(Module):
    """
    Parametrize boxes using the min coordinate and max coordinate,
    initialized to be in the unit hypercube.

    self.boxes[model, box, min/max, dim] \in [0,1]

    In this parametrization, the min and max coordinates are explicitly stored
    in separate dimensions (as shown above), which means that care must be
    taken to preserve max > min while training. (See MinBoxSize Callback.)
    """

    def __init__(self, num_models: int, num_boxes: int, dims: int,
                 init_min_vol: float = default_init_min_vol, method = "gibbs", gibbs_iter: int = 2000, **kwargs):
        super().__init__()
        self.boxes = Parameter(initialize_boxes_in_unit_cube((num_models, num_boxes), dims, init_min_vol, method, gibbs_iter, **kwargs))

    def forward(self, box_indices = slice(None, None, None), **kwargs) -> Tensor:
        """
        Returns a Tensor representing the box embeddings specified by box_indices.

        :param box_indices: Slice, List, or Tensor of the box indices
        :param kwargs: Unused for now, but include this for future possible parameters.
        :return: NamedTensor of shape (model, id, zZ, dim).
        """
        return self.boxes[:, box_indices]


class MinMaxBoxes(Module):
    """
    Parametrize boxes in \RR^d by using 2d coordinates.

    self.boxes[model, box, 2, dim] \in [0,1]

    In this parametrization, we select the z/Z coordinates simply by
    taking the min/max over axis 2, i.e.

    z, _ = torch.min(self.boxes, dim=2) # Tensor of shape (model, box, dim)
    Z, _ = torch.max(self.boxes, dim=2) # Tensor of shape (model, box, dim)

    This avoids the need to make sure the boxes don't "flip", i.e. Z becomes smaller than z.
    """

    def __init__(self, num_models: int, num_boxes: int, dim: int, init_min_vol: float = default_init_min_vol, **kwargs):
        """
        Creates the Parameters used for the representation of boxes.

        :param num_models: Number of models
        :param num_boxes: Number of boxes
        :param dim: Dimension
        :param init_min_vol: Creates boxes which a cube of this volume.
        :param kwargs: Unused for now, but include this for future possible parameters.
        """
        super().__init__()
        unit_boxes = Boxes(num_models, num_boxes, dim, init_min_vol, **kwargs)
        self.boxes = Parameter(unit_boxes.boxes.detach().clone())
        del unit_boxes

    def forward(self, box_indices = slice(None, None, None), **kwargs) -> Tensor:
        """
        Returns a Tensor representing the box embeddings specified by box_indices.

        :param box_indices: Slice, List, or Tensor of the box indices
        :param kwargs: Unused for now, but include this for future possible parameters.
        :return: NamedTensor of shape (model, id, zZ, dim).
        """
        z, _ = torch.min(self.boxes[:, box_indices], dim=2) # Tensor of shape (model, box, dim)
        Z, _ = torch.max(self.boxes[:, box_indices], dim=2) # Tensor of shape (model, box, dim)
        return torch.stack((z, Z), dim=2)


class DeltaBoxes(Module):
    """
    Parametrize boxes using the min coordinate and log of side-length.

    self.z[model, box, dim] \in \RR
    self.logdelta[model, box, dim] \in \RR

    This forces boxes to always have positive side-lengths.
    """

    def __init__(self, num_models: int, num_boxes: int, dim: int, init_min_vol: float = default_init_min_vol, **kwargs):
        super().__init__()
        unit_boxes = Boxes(num_models, num_boxes, dim, init_min_vol, **kwargs)
        self._from_UnitBoxes(unit_boxes)
        del unit_boxes

    def _from_UnitBoxes(self, unit_boxes:Boxes):
        boxes = unit_boxes.boxes.detach().clone()
        z = boxes[:,:,0]
        Z = boxes[:,:,1]
        self.z = Parameter(z)
        self.logdelta = Parameter(torch.log(Z-z))

    def forward(self, box_indices = slice(None, None, None), **kwargs) -> Tensor:
        """
        Returns a Tensor representing the box embeddings specified by box_indices.

        :param box_indices: A NamedTensor of the box indices
        :param kwargs: Unused for now, but include this for future possible parameters.
        :return: Tensor of shape (model, id, zZ, dim).
        """
        return torch.stack((self.z[:, box_indices], self.z[:, box_indices] + torch.exp(self.logdelta[:, box_indices])), dim=2)


class SigmoidBoxes(Module):
    """
    Parametrize boxes using sigmoid to make them always valid and contained within the unit cube.

    self.w[model, box, dim] in Reals
    self.W[model, box, dim] in Reals

    z = sigmoid(w)
    Z = z + sigmoid(W) * (1-z)

    This forces z in (0,1), Z in (z, 1).
    """

    def __init__(self, num_models: int, num_boxes: int, dim: int, init_min_vol: float = default_init_min_vol,  **kwargs):
        super().__init__()
        unit_boxes = Boxes(num_models, num_boxes, dim, init_min_vol, **kwargs)
        self._from_UnitBoxes(unit_boxes)
        del unit_boxes


    def _from_UnitBoxes(self, unit_boxes:Boxes):
        boxes = unit_boxes().detach().clone()
        z = boxes[:,:,0]
        Z = boxes[:,:,1]
        l = (Z-z) / (1-z)
        self.w = Parameter(torch.log(z / (1-z)))
        self.W = Parameter(torch.log(l / (1-l)))


    def forward(self, box_indices = slice(None, None, None), **kwargs) -> Tensor:
        """
        Returns a Tensor representing the box embeddings specified by box_indices.

        :param box_indices: A NamedTensor of the box indices
        :param kwargs: Unused for now, but include this for future possible parameters.
        :return: Tensor of shape (model, id, zZ, dim).
        """
        z = torch.sigmoid(self.w[:, box_indices])
        Z = z + torch.sigmoid(self.W[:, box_indices]) * (1-z)
        return torch.stack((z,Z), dim=-2)


class MinMaxSigmoidBoxes(Module):
    """
    Parametrize boxes using sigmoid to make them always valid and contained within the unit cube.

    self.boxes[model, box, 2, dim] in Reals


    In this parametrization, we first convert to the unit cube:

    unit_cube_boxes = torch.sigmoid(self.boxes)  # shape: (model, box, 2, dim)

    We now select the z/Z coordinates by taking the min/max over axis 2, i.e.

    z, _ = torch.min(unit_cube_boxes, dim=2)
    Z, _ = torch.max(unit_cube_boxes, dim=2)
    """

    def __init__(self, num_models: int, num_boxes: int, dim: int, init_min_vol: float = default_init_min_vol,  **kwargs):
        super().__init__()
        unit_boxes = Boxes(num_models, num_boxes, dim, init_min_vol, **kwargs)
        self._from_UnitBoxes(unit_boxes)
        del unit_boxes


    def _from_UnitBoxes(self, unit_boxes:Boxes):
        boxes = unit_boxes().detach().clone()
        self.boxes = Parameter(torch.log(boxes / (1-boxes)))


    def forward(self, box_indices = slice(None, None, None), **kwargs) -> Tensor:
        """
        Returns a Tensor representing the box embeddings specified by box_indices.

        :param box_indices: A NamedTensor of the box indices
        :param kwargs: Unused for now, but include this for future possible parameters.
        :return: Tensor of shape (model, id, zZ, dim).
        """
        unit_cube_boxes = torch.sigmoid(self.boxes)
        z, _ = torch.min(unit_cube_boxes, dim=2)
        Z, _ = torch.max(unit_cube_boxes, dim=2)
        return torch.stack((z,Z), dim=2)


###############################################
# Downstream Model
###############################################

class WeightedSum(Module):
    def __init__(self, num_models: int) -> None:
        super().__init__()
        self.weights = Parameter(torch.rand(num_models))

    def forward(self, box_vols: Tensor) -> Tensor:
        return (F.softmax(self.weights, dim=0).unsqueeze(0) @ box_vols).squeeze()


class LogWeightedSum(Module):
    def __init__(self, num_models: int) -> None:
        super().__init__()
        self.weights = Parameter(torch.rand(num_models))

    def forward(self, log_box_vols: Tensor) -> Tensor:
        return (torch.logsumexp(self.weights + log_box_vols, 0) - torch.logsumexp(self.weights, 0))


class BoxModel(Module):
    def __init__(self, BoxParamType: type, vol_func: Callable,
                 num_models:int, num_boxes:int, dims:int,
                 init_min_vol: float = default_init_min_vol, universe_box: Optional[Callable] = None, **kwargs):
        super().__init__()
        self.box_embedding = BoxParamType(num_models, num_boxes, dims, init_min_vol, **kwargs)
        self.vol_func = vol_func

        if universe_box is None:
            z = torch.zeros(dims)
            Z = torch.ones(dims)
            self.universe_box = lambda _: torch.stack((z,Z))[None, None]
            self.universe_vol = lambda _: self.vol_func(self.universe_box(None)).squeeze()
            self.clamp = True
        else:
            self.universe_box = universe_box
            self.universe_vol = lambda b: self.vol_func(self.universe_box(b))
            self.clamp = False

        self.weights = WeightedSum(num_models)

    def forward(self, box_indices: Tensor) -> Dict:
        # Unary
        box_embeddings_orig = self.box_embedding()
        if self.clamp:
            box_embeddings = box_embeddings_orig.clamp(0,1)
        else:
            box_embeddings = box_embeddings_orig

        universe_vol = self.universe_vol(box_embeddings)

        unary_probs = self.weights(self.vol_func(box_embeddings) / universe_vol)

        # Conditional
        A = box_embeddings[:, box_indices[:,0]]
        B = box_embeddings[:, box_indices[:,1]]
        A_int_B_vol = self.weights(self.vol_func(intersection(A, B)) / universe_vol) + torch.finfo(torch.float32).tiny
        B_vol = unary_probs[box_indices[:,1]] + torch.finfo(torch.float32).tiny
        P_A_given_B = torch.exp(torch.log(A_int_B_vol) - torch.log(B_vol))

        return {
            "unary_probs": unary_probs,
            "box_embeddings_orig": box_embeddings_orig,
            "A": A,
            "B": B,
            "P(A|B)": P_A_given_B,
        }


class BoxModelStable(Module):
    def __init__(self, BoxParamType: type, log_vol_func: Callable,
                 num_models: int, num_boxes: int, dims: int,
                 init_min_vol: float = default_init_min_vol,
                 universe_box: Optional[Callable] = None, **kwargs):
        super().__init__()
        self.box_embedding = BoxParamType(num_models, num_boxes, dims, init_min_vol, **kwargs)
        self.log_vol_func = log_vol_func

        if universe_box is None:
            z = torch.zeros(dims)
            Z = torch.ones(dims)
            self.universe_box = lambda _: torch.stack((z,Z))[None, None]
            self.log_universe_vol = lambda _: self.log_vol_func(self.universe_box(None)).squeeze()
            self.clamp = True
        else:
            self.universe_box = universe_box
            self.log_universe_vol = lambda b: self.log_vol_func(self.universe_box(b))
            self.clamp = False

        self.weights = LogWeightedSum(num_models)

    def forward(self, box_indices: Tensor) -> Dict:
        # Unary
        box_embeddings_orig = self.box_embedding()
        if self.clamp:
            box_embeddings = box_embeddings_orig.clamp(0,1)
        else:
            box_embeddings = box_embeddings_orig

        log_universe_vol = self.log_universe_vol(box_embeddings)

        log_unary_probs = self.weights(self.log_vol_func(box_embeddings) -  log_universe_vol)

        # Conditional
        A = box_embeddings[:, box_indices[:,0]]
        B = box_embeddings[:, box_indices[:,1]]
        log_A_int_B_vol = self.weights(self.log_vol_func(intersection(A, B)) - log_universe_vol)
        log_B_vol = log_unary_probs[box_indices[:,1]]
        log_P_A_given_B = log_A_int_B_vol - log_B_vol

        return {
            "log_unary_probs": log_unary_probs,
            "box_embeddings_orig": box_embeddings_orig,
            "A": A,
            "B": B,
            "log P(A|B)": log_P_A_given_B,
            "P(A|B)": torch.exp(log_P_A_given_B),
        }


class BoxModelTriples(Module):
    def __init__(self, BoxParamType: type, vol_func: Callable,
                 num_models:int, num_boxes:int, dims:int,
                 init_min_vol: float = default_init_min_vol, universe_box: Optional[Callable] = None, **kwargs):
        super().__init__()
        self.box_embedding = BoxParamType(num_models, num_boxes, dims, init_min_vol, **kwargs)
        self.vol_func = vol_func

        if universe_box is None:
            z = torch.zeros(dims)
            Z = torch.ones(dims)
            self.universe_box = lambda _: torch.stack((z,Z))[None, None]
            self.universe_vol = lambda _: self.vol_func(self.universe_box(None)).squeeze()
            self.clamp = True
        else:
            self.universe_box = universe_box
            self.universe_vol = lambda b: self.vol_func(self.universe_box(b))
            self.clamp = False

        self.weights = WeightedSum(num_models)

    def forward(self, ids):
        # Unary
        box_embeddings_orig = self.box_embedding()
        if self.clamp:
            box_embeddings = box_embeddings_orig.clamp(0,1)
        else:
            box_embeddings = box_embeddings_orig

        universe_vol = self.universe_vol(box_embeddings)

        # unary_probs = self.vol_func(box_embeddings)

        probs = torch.zeros(ids.shape[0]).to(box_embeddings_orig.device)

        unary_box_mask = ids[:,0] == ids[:,1]
        three_boxes_mask = ids[:,1] != ids[:,2]
        two_boxes_mask = (1-three_boxes_mask) * (1-unary_box_mask)

        num_unary_boxes = torch.sum(unary_box_mask)
        if num_unary_boxes > 0:
            unary_boxes = box_embeddings[:,ids[unary_box_mask,0]]
            unary_probs = self.weights(self.vol_func(unary_boxes) / universe_vol)
            probs[unary_box_mask] = unary_probs


        two_vol = torch.tensor([]).to(box_embeddings.device)
        num_two_boxes = torch.sum(two_boxes_mask)
        if num_two_boxes > 0:
            A = box_embeddings[:, ids[two_boxes_mask, 0]]
            B = box_embeddings[:, ids[two_boxes_mask, 1]]
            two_vol = self.weights(self.vol_func(intersection(A, B)) / universe_vol) + torch.finfo(torch.float32).tiny
            two_div = self.weights(self.vol_func(A) / universe_vol) + torch.finfo(torch.float32).tiny
            two_cond = torch.exp(torch.log(two_vol) - torch.log(two_div))
            probs[two_boxes_mask] = two_cond

        num_three_boxes = torch.sum(three_boxes_mask)
        if num_three_boxes > 0:
            A = box_embeddings[:, ids[three_boxes_mask, 0]]
            B = box_embeddings[:, ids[three_boxes_mask, 1]]
            C = box_embeddings[:, ids[three_boxes_mask, 2]]
            A_int_B = intersection(A, B)
            three_vol = self.weights(self.vol_func(intersection(A_int_B, C)) / universe_vol) + torch.finfo(torch.float32).tiny
            three_div = self.weights(self.vol_func(A_int_B) / universe_vol) + torch.finfo(torch.float32).tiny
            three_cond = torch.exp(torch.log(three_vol) - torch.log(three_div))
            probs[three_boxes_mask] = three_cond

        return {
            "box_embeddings": box_embeddings,
            "box_embeddings_orig": box_embeddings_orig,
            "ids": ids,
            "probs": probs,
            'weights_layer': self.weights,
            'parts': ids[:,-1],
            "unary_box_mask": unary_box_mask,
            "two_boxes_mask": two_boxes_mask,
            "three_boxes_mask": three_boxes_mask,
            "two_vol": two_vol,
        }


class BoxModelJointStable(Module):
    def __init__(self, BoxParamType: type, log_vol_func: Callable,
                 num_models: int, num_boxes: int, dims: int,
                 init_min_vol: float = default_init_min_vol,
                 universe_box: Optional[Callable] = None, **kwargs):
        super().__init__()
        self.box_embedding = BoxParamType(num_models, num_boxes, dims, init_min_vol, **kwargs)
        self.log_vol_func = log_vol_func

        if universe_box is None:
            z = torch.zeros(dims)
            Z = torch.ones(dims)
            self.universe_box = lambda _: torch.stack((z,Z))[None, None]
            self.log_universe_vol = lambda _: self.log_vol_func(self.universe_box(None)).squeeze()
            self.clamp = True
        else:
            self.universe_box = universe_box
            self.log_universe_vol = lambda b: self.log_vol_func(self.universe_box(b))
            self.clamp = False

        self.weights = LogWeightedSum(num_models)

    def forward(self, box_indices: Tensor) -> Dict:
        # Unary
        box_embeddings_orig = self.box_embedding()
        if self.clamp:
            box_embeddings = box_embeddings_orig.clamp(0,1)
        else:
            box_embeddings = box_embeddings_orig

        log_universe_vol = self.log_universe_vol(box_embeddings)

        log_unary_probs = self.weights(self.log_vol_func(box_embeddings) -  log_universe_vol)

        log_P_A = log_unary_probs[box_indices[:,0]]
        log_P_B = log_unary_probs[box_indices[:,1]]
        A = box_embeddings[:, box_indices[:,0]]
        B = box_embeddings[:, box_indices[:,1]]
        log_P_A_B = self.weights(self.log_vol_func(intersection(A, B)) - log_universe_vol)

        return {
            "log_P_A": log_P_A,
            "log_P_B": log_P_B,
            "log_P_A_B": log_P_A_B,
        }

class BoxModelJoint(Module):
    def __init__(self, BoxParamType: type, vol_func: Callable,
                 num_models:int, num_boxes:int, dims:int,
                 init_min_vol: float = default_init_min_vol, universe_box: Optional[Callable] = None, **kwargs):
        super().__init__()
        self.box_embedding = BoxParamType(num_models, num_boxes, dims, init_min_vol, **kwargs)
        self.vol_func = vol_func

        if universe_box is None:
            z = torch.zeros(dims)
            Z = torch.ones(dims)
            self.universe_box = lambda _: torch.stack((z,Z))[None, None]
            self.universe_vol = lambda _: self.vol_func(self.universe_box(None)).squeeze()
            self.clamp = True
        else:
            self.universe_box = universe_box
            self.universe_vol = lambda b: self.vol_func(self.universe_box(b))
            self.clamp = False

        self.weights = WeightedSum(num_models)

    def forward(self, box_indices: Tensor) -> Dict:
        # Unary
        box_embeddings_orig = self.box_embedding()
        if self.clamp:
            box_embeddings = box_embeddings_orig.clamp(0,1)
        else:
            box_embeddings = box_embeddings_orig

        universe_vol = self.universe_vol(box_embeddings)

        unary_probs = self.weights(self.vol_func(box_embeddings) / universe_vol)

        P_A = unary_probs[box_indices[:,0]]
        P_B = unary_probs[box_indices[:,1]]
        A = box_embeddings[:, box_indices[:,0]]
        B = box_embeddings[:, box_indices[:,1]]
        P_A_B = self.weights(self.vol_func(intersection(A, B)) / universe_vol)

        return {
            "P_A": P_A,
            "P_B": P_B,
            "P_A_B": P_A_B,
            "A": A,
            "B": B,
        }

class BoxModelJointTriple(Module):
    def __init__(self, BoxParamType: type, vol_func: Callable,
                 num_models:int, num_boxes:int, dims:int,
                 init_min_vol: float = default_init_min_vol, universe_box: Optional[Callable] = None, **kwargs):
        super().__init__()
        self.box_embedding = BoxParamType(num_models, num_boxes, dims, init_min_vol, **kwargs)
        self.vol_func = vol_func

        if universe_box is None:
            z = torch.zeros(dims)
            Z = torch.ones(dims)
            self.universe_box = lambda _: torch.stack((z,Z))[None, None]
            self.universe_vol = lambda _: self.vol_func(self.universe_box(None)).squeeze()
            self.clamp = True
        else:
            self.universe_box = universe_box
            self.universe_vol = lambda b: self.vol_func(self.universe_box(b))
            self.clamp = False

        self.weights = WeightedSum(num_models)

    def forward(self, box_indices: Tensor) -> Dict:
        # Unary
        box_embeddings_orig = self.box_embedding()
        if self.clamp:
            box_embeddings = box_embeddings_orig.clamp(0,1)
        else:
            box_embeddings = box_embeddings_orig

        universe_vol = self.universe_vol(box_embeddings)

        unary_probs = self.weights(self.vol_func(box_embeddings) / universe_vol)

        P_A = unary_probs[box_indices[:,0]]
        P_B = unary_probs[box_indices[:,1]]
        P_C = unary_probs[box_indices[:,2]]
        A = box_embeddings[:, box_indices[:,0]]
        B = box_embeddings[:, box_indices[:,1]]
        C = box_embeddings[:, box_indices[:,2]]
        P_A_B = self.weights(self.vol_func(intersection(A, B)) / universe_vol)
        P_A_C = self.weights(self.vol_func(intersection(A, C)) / universe_vol)
        P_B_C = self.weights(self.vol_func(intersection(B, C)) / universe_vol)

        P_A_B_C = self.weights(self.vol_func(intersection(intersection(A,B), C)) / universe_vol)

        return {
            "P_A": P_A,
            "P_B": P_B,
            "P_C": P_C,
            "P_A_B": P_A_B,
            "P_A_C": P_A_C,
            "P_B_C": P_B_C,
            "P_A_B_C": P_A_B_C,
            "A": A,
            "B": B,
            "C": C,
        }
