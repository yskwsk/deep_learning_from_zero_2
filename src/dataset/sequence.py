# coding: utf-8

import numpy
import os

id_to_char: dict[int, str] = {}
char_to_id: dict[str, int] = {}


def _update_vocab(txt: str) -> None:
    chars = list(txt)

    for i, char in enumerate(chars):
        if char not in char_to_id:
            tmp_id = len(char_to_id)
            char_to_id[char] = tmp_id
            id_to_char[tmp_id] = char


def load_data(
    file_name: str = "addition.txt",
    seed: int | None = 1984
) -> tuple[tuple[numpy.ndarray, numpy.ndarray], tuple[numpy.ndarray, numpy.ndarray]] | None:
    file_path = os.path.dirname(os.path.abspath(__file__)) + "/" + file_name

    if not os.path.exists(file_path):
        print(f"No file: {file_name}")
        return None

    questions: list[str] = []
    answers: list[str] = []

    for line in open(file_path, mode="r"):
        idx = line.find("_")
        questions.append(line[:idx])
        answers.append(line[idx:-1])

    # create vocab dict
    for i in range(len(questions)):
        _update_vocab(questions[i])
        _update_vocab(answers[i])

    # create numpy array
    x = numpy.zeros((len(questions), len(questions[0])), dtype=int)
    t = numpy.zeros((len(questions), len(answers[0])), dtype=int)

    for i, sentence in enumerate(questions):
        x[i] = [char_to_id[c] for c in list(sentence)]
    for i, sentence in enumerate(answers):
        t[i] = [char_to_id[c] for c in list(sentence)]

    # shuffle
    indices = numpy.arange(len(x))
    if seed is not None:
        numpy.random.seed(seed)
    numpy.random.shuffle(indices)
    x = x[indices]
    t = t[indices]

    # 10% for validation set
    split_at = len(x) - len(x) // 10
    (x_train, x_test) = x[:split_at], x[split_at:]
    (t_train, t_test) = t[:split_at], t[split_at:]

    return (x_train, t_train), (x_test, t_test)


def get_vocab() -> tuple[dict[str, int], dict[int, str]]:
    return char_to_id, id_to_char
