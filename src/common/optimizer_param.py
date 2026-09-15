# coding: utf-8

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OptimizerParam:
    learning_rate: float = 1.0e-2


@dataclass(frozen=True, slots=True)
class SGDParam(OptimizerParam):
    ...


@dataclass(frozen=True, slots=True)
class MomentumParam(OptimizerParam):
    momentum: float = 0.9


@dataclass(frozen=True, slots=True)
class NesterovParam(OptimizerParam):
    momentum: float = 0.9


@dataclass(frozen=True, slots=True)
class AdaGradParam(OptimizerParam):
    ...


@dataclass(frozen=True, slots=True)
class RMSpropParam(OptimizerParam):
    decay_rate: float = 0.99


@dataclass(frozen=True, slots=True)
class AdamParam(OptimizerParam):
    beta1: float = 0.9
    beta2: float = 0.999
