# coding: utf-8

import numpy
import time
import matplotlib.pyplot as plt

# import numpy as np
from common.base_model import BaseModel
from common.np import np, NDArray
from common.optimizer import Optimizer
from common.util import clip_grads


def remove_duplicate(
    params: list[NDArray],
    grads: list[NDArray]
) -> tuple[list[NDArray], list[NDArray]]:
    '''
    パラメータ配列中の重複する重みをひとつに集約し、
    その重みに対応する勾配を加算する
    '''
    # copy list
    params, grads = params[:], grads[:]

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 重みを共有する場合
                if params[i] is params[j]:
                    # 勾配の加算
                    grads[i] += grads[j]
                    find_flg = True
                    # 重複した重みと勾配を取り除く
                    params.pop(j)
                    grads.pop(j)
                # 転置行列として重みを共有する場合（weight tying）
                elif (
                    params[i].ndim == 2 and
                    params[j].ndim == 2 and
                    params[i].T.shape == params[j].shape and
                    np.all(params[i].T == params[j])
                ):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                if find_flg:
                    break
            if find_flg:
                break
        if not find_flg:
            # 重複する重みが見つからなくなった場合、whileループから抜ける
            break

    return params, grads


class Trainer:
    def __init__(
        self,
        model: BaseModel,
        optimizer: Optimizer
    ) -> None:
        self.model: BaseModel = model
        self.optimizer: Optimizer = optimizer
        self.loss_list: list[float] = []
        self.eval_interval: int | None = None
        self.current_epoch: int = 0

    def fit(
        self,
        x: NDArray,
        t: NDArray,
        max_epoch: int = 10,
        batch_size: int = 32,
        max_grad: float | None = None,
        eval_interval: int = 20
    ):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0.0
        loss_count = 0.0

        start_time = time.time()

        for epoch in range(max_epoch):
            # データのシャッフル
            idx = numpy.random.permutation(numpy.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]

                # 勾配を求め、パラメータを更新
                loss = model.forward(batch_x, batch_t)
                model.backward()
                # 共有された重みを1つに集約
                params, grads = remove_duplicate(model.params, model.grads)
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += float(loss)
                loss_count += 1

                # 評価
                if (eval_interval is not None) and ((iters + 1) % eval_interval == 0):
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print((
                        '| epoch %d |  iter %d / %d | time %d[s] | loss %.2f'
                        % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss)
                        )
                    )
                    self.loss_list.append(float(avg_loss))
                    total_loss = 0
                    loss_count = 0

            self.current_epoch += 1

    def plot(self, ylim: tuple[float, float] | None = None):
        x = numpy.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('loss')
        plt.show()


# class RnnlmTrainer:
#     def __init__(self, model, optimizer):
#         self.model = model
#         self.optimizer = optimizer
#         self.time_idx = None
#         self.ppl_list = None
#         self.eval_interval = None
#         self.current_epoch = 0

#     def get_batch(self, x, t, batch_size, time_size):
#         batch_x = np.empty((batch_size, time_size), dtype='i')
#         batch_t = np.empty((batch_size, time_size), dtype='i')

#         data_size = len(x)
#         jump = data_size // batch_size
#         offsets = [i * jump for i in range(batch_size)]  # バッチの各サンプルの読み込み開始位置

#         for time in range(time_size):
#             for i, offset in enumerate(offsets):
#                 batch_x[i, time] = x[(offset + self.time_idx) % data_size]
#                 batch_t[i, time] = t[(offset + self.time_idx) % data_size]
#             self.time_idx += 1
#         return batch_x, batch_t

#     def fit(self, xs, ts, max_epoch=10, batch_size=20, time_size=35,
#             max_grad=None, eval_interval=20):
#         data_size = len(xs)
#         max_iters = data_size // (batch_size * time_size)
#         self.time_idx = 0
#         self.ppl_list = []
#         self.eval_interval = eval_interval
#         model, optimizer = self.model, self.optimizer
#         total_loss = 0
#         loss_count = 0

#         start_time = time.time()
#         for epoch in range(max_epoch):
#             for iters in range(max_iters):
#                 batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

#                 # 勾配を求め、パラメータを更新
#                 loss = model.forward(batch_x, batch_t)
#                 model.backward()
#                 params, grads = remove_duplicate(model.params, model.grads)  # 共有された重みを1つに集約
#                 if max_grad is not None:
#                     clip_grads(grads, max_grad)
#                 optimizer.update(params, grads)
#                 total_loss += loss
#                 loss_count += 1

#                 # パープレキシティの評価
#                 if (eval_interval is not None) and (iters % eval_interval) == 0:
#                     ppl = np.exp(total_loss / loss_count)
#                     elapsed_time = time.time() - start_time
#                     print('| epoch %d |  iter %d / %d | time %d[s] | perplexity %.2f'
#                           % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, ppl))
#                     self.ppl_list.append(float(ppl))
#                     total_loss, loss_count = 0, 0

#             self.current_epoch += 1

#     def plot(self, ylim=None):
#         x = numpy.arange(len(self.ppl_list))
#         if ylim is not None:
#             plt.ylim(*ylim)
#         plt.plot(x, self.ppl_list, label='train')
#         plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
#         plt.ylabel('perplexity')
#         plt.show()
