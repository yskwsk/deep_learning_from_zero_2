# coding: utf-8

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from common.np import np, NDArray
from common.optimizer_param import (
    OptimizerParam,
    SGDParam,
    MomentumParam,
    NesterovParam,
    AdaGradParam,
    RMSpropParam,
    AdamParam
)


P = TypeVar("P", bound=OptimizerParam)


class Optimizer(ABC, Generic[P]):
    def __init__(self, param: P) -> None:
        self.param = param

    @abstractmethod
    def update(self, params: list[NDArray], grads: list[NDArray]) -> None:
        ...


class SGD(Optimizer[SGDParam]):
    '''
    確率的勾配降下法（Stochastic Gradient Descent）
    '''
    def __init__(self, param: SGDParam = SGDParam(learning_rate=0.1)) -> None:
        self.learning_rate = param.learning_rate

    def update(self, params: list[NDArray], grads: list[NDArray]) -> None:
        for i in range(len(params)):
            params[i] -= self.learning_rate * grads[i]


class Momentum(Optimizer[MomentumParam]):
    def __init__(
        self,
        param: MomentumParam = MomentumParam(learning_rate=0.01, momentum=0.9)
    ) -> None:
        self.learning_rate = param.learning_rate
        self.momentum = param.momentum
        self.v: list[NDArray] | None = None

    def update(self, params: list[NDArray], grads: list[NDArray]) -> None:
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))

        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.learning_rate * grads[i]
            params[i] += self.v[i]


class Nesterov(Optimizer[NesterovParam]):
    '''
    Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)
    '''
    def __init__(
        self,
        param: NesterovParam = NesterovParam(learning_rate=0.01, momentum=0.9)
    ) -> None:
        self.learning_rate = param.learning_rate
        self.momentum = param.momentum
        self.v: list[NDArray] | None = None

    def update(self, params: list[NDArray], grads: list[NDArray]) -> None:
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))

        for i in range(len(params)):
            self.v[i] *= self.momentum
            self.v[i] -= self.learning_rate * grads[i]
            params[i] += self.momentum * self.momentum * self.v[i]
            params[i] -= (1 + self.momentum) * self.learning_rate * grads[i]


class AdaGrad(Optimizer[AdaGradParam]):
    def __init__(self, param: AdaGradParam = AdaGradParam(learning_rate=0.01)) -> None:
        self.learning_rate = param.learning_rate
        self.h: list[NDArray] | None = None

    def update(self, params: list[NDArray], grads: list[NDArray]) -> None:
        if self.h is None:
            self.h = []
            for param in params:
                self.h.append(np.zeros_like(param))

        for i in range(len(params)):
            self.h[i] += grads[i] * grads[i]
            params[i] -= self.learning_rate * grads[i] / (np.sqrt(self.h[i]) + 1e-7)


class RMSprop(Optimizer[RMSpropParam]):
    def __init__(
        self,
        param: RMSpropParam = RMSpropParam(learning_rate=0.01, decay_rate=0.99)
    ) -> None:
        self.learning_rate = param.learning_rate
        self.decay_rate = param.decay_rate
        self.h: list[NDArray] | None = None

    def update(self, params: list[NDArray], grads: list[NDArray]) -> None:
        if self.h is None:
            self.h = []
            for param in params:
                self.h.append(np.zeros_like(param))

        for i in range(len(params)):
            self.h[i] *= self.decay_rate
            self.h[i] += (1 - self.decay_rate) * grads[i] * grads[i]
            params[i] -= self.learning_rate * grads[i] / (np.sqrt(self.h[i]) + 1e-7)


class Adam(Optimizer[AdamParam]):
    '''
    Adam (http://arxiv.org/abs/1412.6980v8)
    '''
    def __init__(
        self,
        param: AdamParam = AdamParam(learning_rate=0.001, beta1=0.9, beta2=0.999)
    ) -> None:
        self.learning_rate = param.learning_rate
        self.beta1 = param.beta1
        self.beta2 = param.beta2
        self.iter = 0
        self.m: list[NDArray] | None = None
        self.v: list[NDArray] | None = None

    def update(self, params: list[NDArray], grads: list[NDArray]) -> None:
        if self.m is None or self.v is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        self.iter += 1
        lr_t = self.learning_rate * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)
