# coding: utf-8

from common.util import (
    cos_similarity,
    create_co_occurence_matrix,
    preprocess
)


text: str = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text=text)
vocab_size = len(word_to_id)
C = create_co_occurence_matrix(corpus=corpus, vocab_size=vocab_size)

# "you"の単語ベクトル
c0 = C[word_to_id["you"]]
# "i"の単語ベクトル
c1 = C[word_to_id["i"]]

sim = cos_similarity(c0, c1)
print(sim)
