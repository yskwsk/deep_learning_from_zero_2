# coding: utf-8

import numpy as np

from common.util import (
    create_co_occurence_matrix,
    most_similar,
    positive_pointwise_mutual_information,
    positive_pointwise_mutual_information_fast,
)
from dataset import ptb


window_size = 2
wordvec_size = 100
use_ppmi_fast = True

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print('counting  co-occurrence ...')
co_matrix = create_co_occurence_matrix(
    corpus=corpus,
    vocab_size=vocab_size,
    window_size=window_size
)
print('calculating PPMI ...')
if use_ppmi_fast:
    ppmi = positive_pointwise_mutual_information_fast(
        co_occuerence_matrix=co_matrix,
        verbose=True
    )
else:
    ppmi = positive_pointwise_mutual_information(
        co_occuerence_matrix=co_matrix,
        verbose=True
    )

print('calculating SVD ...')
try:
    # truncated SVD (fast!)
    from sklearn.utils.extmath import randomized_svd  # type: ignore[import-untyped]
    U, S, V = randomized_svd(
        M=ppmi,
        n_components=wordvec_size,
        n_iter=5,
        random_state=None
    )
except ImportError:
    # SVD (slow)
    U, S, V = np.linalg.svd(a=ppmi)

word_vecs = U[:, :wordvec_size]

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(
        query=query,
        word_to_id=word_to_id,
        id_to_word=id_to_word,
        word_matrix=word_vecs,
        top=5
    )
