# coding: utf-8

from dataset import ptb
from common.util import eval_perplexity

from rnnlm import Rnnlm
from better_rnnlm import BetterRnnlm


if __name__ == '__main__':
    model = Rnnlm()
    # model = BetterRnnlm()

    # 学習済みのパラメータの読み込み
    model.load_params()

    corpus, _, _ = ptb.load_data('test')

    model.reset_state()
    ppl_test = eval_perplexity(model, corpus)
    print('test perplexity: ', ppl_test)
