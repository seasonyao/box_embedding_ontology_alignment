from dataclasses import dataclass


class LearnerException(Exception):
    """Base class for exceptions in this module."""
    pass


class StopTrainingException(LearnerException):
    """Base class for exceptions which should stop training in this module."""
    pass


@dataclass
class MaxLoss(StopTrainingException):
    """Max Loss Exceeded"""
    loss: float
    max_loss: float

    def __str__(self):
        return f"Max Loss Exceeded: {self.loss} > {self.max_loss}"


@dataclass
class EarlyStopping(StopTrainingException):
    """Max Value Exceeded"""
    condition: str

    def __str__(self):
        return f"EarlyStopping: {self.condition}"


@dataclass
class NaNsException(StopTrainingException):
    location: str

    def __str__(self):
        return f"NaNs: NaNs in {self.location}"