# coding: utf-8

from common.util import (
    create_co_occurence_matrix,
    most_similar,
    preprocess
)


text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text=text)
vocab_size = len(word_to_id)
C = create_co_occurence_matrix(corpus=corpus, vocab_size=vocab_size)

most_similar(
    query="you",
    word_to_id=word_to_id,
    id_to_word=id_to_word,
    word_matrix=C,
    top=5
)
