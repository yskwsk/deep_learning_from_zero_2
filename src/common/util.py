# coding: utf-8

from common.np import np, NDArray
# import numpy as np


def preprocess(text: str) -> tuple[np.ndarray, dict[str, int], dict[int, str]]:
    text = text.lower()
    text = text.replace(".", " .")
    words = text.split(" ")

    word_to_id: dict[str, int] = {}
    id_to_word: dict[int, str] = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


def cos_similarity(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)


def most_similar(
    query: str,
    word_to_id: dict[str, int],
    id_to_word: dict[int, str],
    word_matrix: np.ndarray,
    top: int = 5
) -> None:
    # クエリを取り出す
    if query not in word_to_id:
        print(f"{query} is not found")
        return

    print(f"\n[query] {query}")
    query_id: int = word_to_id[query]
    query_vec: np.ndarray = word_matrix[query_id]

    # コサイン類似度の算出
    vocab_size: int = len(id_to_word)
    similarity: np.ndarray = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # コサイン類似度の結果から、その値を高い順に出力
    count: int = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(f" {id_to_word[i]}: {similarity[i]}")
        count += 1
        if count >= top:
            return


def convert_one_hot(corpus: np.ndarray, vocab_size: int) -> np.ndarray:
    """one-hot表現への変換
    Args:
        corpus: 単語IDのリスト (1次元あるいは2次元のNumPy配列)
        vocab_size: 語彙数
    Returns:
        one-hot: 単語IDのリストをone-hot表現に変換した結果（2次元あるいは3次元のNumPy配列）
    """
    N = corpus.shape[0]
    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1
    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1
    else:
        raise ValueError("corpus.ndim must be 1 or 2.")
    return one_hot


def create_co_occurence_matrix(
    corpus: np.ndarray,
    vocab_size: int,
    window_size: int = 1
) -> np.ndarray:
    """共起行列の作成"""
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def positive_pointwise_mutual_information(
    co_occuerence_matrix: np.ndarray,
    verbose: bool = False,
    eps: float = 1.0e-8
) -> np.ndarray:
    '''PPMI（正の相互情報量）の計算'''
    ppmi = np.zeros_like(co_occuerence_matrix, dtype=np.float32)
    N = np.sum(co_occuerence_matrix)
    S = np.sum(co_occuerence_matrix, axis=0)
    total = co_occuerence_matrix.shape[0] * co_occuerence_matrix.shape[1]
    cnt = 0

    for i in range(co_occuerence_matrix.shape[0]):
        for j in range(co_occuerence_matrix.shape[1]):
            pmi = np.log2(co_occuerence_matrix[i, j] * N / (S[i] * S[j]) + eps)
            ppmi[i, j] = max(0.0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total // 100) == 0:
                    print(f"{100*cnt/total}% done")

    return ppmi


def positive_pointwise_mutual_information_fast(
    co_occuerence_matrix: np.ndarray,
    verbose: bool = False,
    eps: float = 1.0e-8
) -> np.ndarray:
    '''PPMI（正の相互情報量）の計算の高速版'''
    N = np.sum(co_occuerence_matrix, dtype=np.float32)
    S = np.sum(co_occuerence_matrix, axis=0)
    Sij = np.outer(S, S)
    ppmi = np.log2(co_occuerence_matrix * N / (Sij) + eps).astype(np.float32)
    ppmi = np.where(ppmi > 0.0, ppmi, 0.0)
    return ppmi


def create_contexts_target(
    corpus: np.ndarray,
    window_size: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    target = corpus[window_size:-window_size]
    contexts: list[list[int]] = []

    begin = window_size
    end = len(corpus) - window_size

    for idx in range(begin, end):
        cs: list[int] = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


def to_cpu(x):
    import numpy
    if isinstance(x, numpy.ndarray):
        return x
    return np.asnumpy(x)


def to_gpu(x):
    import cupy  # type: ignore[import-not-found]
    if isinstance(x, cupy.ndarray):
        return x
    return cupy.asarray(x)


def clip_grads(grads: list[NDArray], max_norm: float) -> None:
    total_norm = 0.0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1.0:
        for grad in grads:
            grad *= rate


def analogy(
    word1: str,
    word2: str,
    word3: str,
    word_to_id: dict[str, int],
    id_to_word: dict[int, str],
    word_matrix: np.ndarray,
    top: int = 5,
    answer: str | None = None
):
    for word in (word1, word2, word3):
        if word not in word_to_id:
            print(f"{word} is not found")
            return

    print(f"\n[analogy] {word1}:{word2} = {word3}:?")
    word1_vec = word_matrix[word_to_id[word1]]
    word2_vec = word_matrix[word_to_id[word2]]
    word3_vec = word_matrix[word_to_id[word3]]
    query_vec = word2_vec - word1_vec + word3_vec
    query_vec = normalize(query_vec)

    similarity = np.dot(word_matrix, query_vec)

    if answer is not None:
        ans_id = word_to_id[answer]
        ans = np.dot(word_matrix[ans_id], query_vec)
        print(f"==>{answer}:{ans}")

    count = 0
    for i in (-1 * similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        if id_to_word[i] in (word1, word2, word3):
            continue
        print(f" {id_to_word[i]}: {similarity[i]}")

        count += 1
        if count >= top:
            return


def normalize(x: np.ndarray):
    if x.ndim == 1:
        s = np.sqrt((x * x).sum())
        x /= s
    elif x.ndim == 2:
        s = np.sqrt((x * x).sum(axis=1))
        x /= s.reshape((s.shape[0], 1))
    else:
        raise ValueError("ndim of input array must be 1 or 2.")
    return x
