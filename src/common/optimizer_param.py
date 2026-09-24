# coding: utf-8

from dataclasses import dataclass


@dataclass(slots=True)
class OptimizerParam:
    learning_rate: float = 1.0e-2


@dataclass(slots=True)
class SGDParam(OptimizerParam):
    ...


@dataclass(slots=True)
class MomentumParam(OptimizerParam):
    momentum: float = 0.9


@dataclass(slots=True)
class NesterovParam(OptimizerParam):
    momentum: float = 0.9


@dataclass(slots=True)
class AdaGradParam(OptimizerParam):
    ...


@dataclass(slots=True)
class RMSpropParam(OptimizerParam):
    decay_rate: float = 0.99


@dataclass(slots=True)
class AdamParam(OptimizerParam):
    beta1: float = 0.9
    beta2: float = 0.999
