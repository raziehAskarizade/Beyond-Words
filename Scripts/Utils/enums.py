from enum import Enum


class Optimizer(Enum):
    GD = 1,
    SGD = 2,
    RMS_PROP = 3
    ADAM = 4,
    ADA_GRAD = 5,
    ADA_MAX = 6


class LossType(Enum):
    CROSS_ENTROPY = 1,
    BCE = 2,
    MSE = 3
