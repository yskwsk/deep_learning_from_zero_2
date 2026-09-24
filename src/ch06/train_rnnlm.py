# coding: utf-8

from common.np import np
from common.optimizer import SGD
from common.optimizer_param import SGDParam
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity
from dataset import ptb
from rnnlm import Rnnlm


np.random.seed(1234)


# ハイパーパラメータの設定
batch_size = 20
wordvec_size = 100
# RNNの隠れ状態ベクトルの要素数
hidden_size = 100
# RNNを展開するサイズ
time_size = 35
lr = 20.0
max_epoch = 4
max_grad = 0.25

# 学習データの読み込み
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_test, word_to_id_test, id_to_word_test = ptb.load_data('test')
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

# モデルの生成
model = Rnnlm(
    vocab_size=vocab_size,
    wordvec_size=wordvec_size,
    hidden_size=hidden_size
)
optimizer = SGD(SGDParam(learning_rate=lr))
trainer = RnnlmTrainer(
    model=model,
    optimizer=optimizer
)

# 勾配クリッピングを適用して学習
trainer.fit(
    xs=xs,
    ts=ts,
    max_epoch=max_epoch,
    batch_size=batch_size,
    time_size=time_size,
    max_grad=max_grad,
    eval_interval=20
)
trainer.plot(ylim=(0, 500))

# テストデータで評価
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print('test perplexity: ', ppl_test)

# パラメータの保存
model.save_params()
