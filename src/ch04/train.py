# coding: utf-8

import sys
sys.path.append('..')

import pickle
from typing import Any

from common import config
config.GPU = True

from cbow import CBOW
from common.np import np
from common.optimizer import Adam
from common.util import create_contexts_target, to_cpu, to_gpu
from common.trainer import Trainer
from dataset import ptb


# ハイパーパラメータの設定
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 100

# データの読み込み
corpus, word_to_id, id_to_word = ptb.load_data("train")
vocab_size = len(word_to_id)
contexts, target = create_contexts_target(
    corpus=corpus,
    window_size=window_size
)
if config.GPU:
    contexts, target = to_gpu(contexts), to_gpu(target)

# モデルの生成
model = CBOW(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    window_size=window_size,
    corpus=corpus
)
optimizer = Adam()
trainer = Trainer(
    model=model,
    optimizer=optimizer
)

# 学習開始
trainer.fit(
    x=contexts,
    t=target,
    max_epoch=max_epoch,
    batch_size=batch_size,
)
trainer.plot()

# 後ほど利用できるように必要なデータを保存
word_vecs = model.word_vecs
if config.GPU:
    word_vecs = to_cpu(word_vecs)
params: dict[str, Any] = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = "cbow_params.pkl"
# pkl_file = "skipgram_params.pkl"
with open(pkl_file, 'wb') as file:
    pickle.dump(params, file, -1)
