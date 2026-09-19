# coding: utf-8

import numpy as np

from common.util import (
    create_co_occurence_matrix,
    positive_pointwise_mutual_information,
    preprocess
)


text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text=text)
vocab_size = len(word_to_id)
co_matrix = create_co_occurence_matrix(corpus=corpus, vocab_size=vocab_size)
ppmi = positive_pointwise_mutual_information(co_occuerence_matrix=co_matrix)

np.set_printoptions(precision=3)
print("co-occurence matrix")
print(co_matrix)
print("-"*50)
print("PPMI")
print(ppmi)
