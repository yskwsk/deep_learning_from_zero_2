# coding: utf-8

import numpy as np
import os
import pickle
try:
    import urllib.request
    import urllib.error
except ImportError:
    raise ImportError("Use Python3!")
from pathlib import Path
from typing import Final, Literal, get_args


URL_BASE: Final[str] = "https://raw.githubusercontent.com/tomsercu/lstm/master/data/"

DATA_TYPE = Literal[
    "train",
    "test",
    "valid"
]

KEY_FILES: Final[dict[DATA_TYPE, str]] = {
    "train": "ptb.train.txt",
    "test": "ptb.test.txt",
    "valid": "ptb.valid.txt",
}

SAVE_FILES: Final[dict[DATA_TYPE, str]] = {
    "train": "ptb.train.npy",
    "test": "ptb.test.npy",
    "valid": "ptb.valid.npy",
}

VOCAB_FILE: Final[str] = "ptb.vocab.pkl"

DATASET_DIR_PATH: Final[Path] = Path(os.path.dirname(os.path.abspath(__file__)))


def _download(filename: str) -> None:
    file_path = DATASET_DIR_PATH / filename
    if file_path.exists():
        return

    print(f"Downloading {filename} ...")

    try:
        urllib.request.urlretrieve(URL_BASE + filename, file_path)
    except urllib.error.URLError:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[assignment]
        urllib.request.urlretrieve(URL_BASE + filename, file_path)

    print("Done")


def load_vocab() -> tuple[dict[str, int], dict[int, str]]:
    vocab_path = DATASET_DIR_PATH / VOCAB_FILE

    if vocab_path.exists():
        with open(vocab_path, mode="rb") as file:
            _word_to_id, _id_to_word = pickle.load(file)
        return _word_to_id, _id_to_word

    word_to_id: dict[str, int] = {}
    id_to_word: dict[int, str] = {}
    data_type: DATA_TYPE = "train"
    file_name = KEY_FILES[data_type]
    file_path = DATASET_DIR_PATH / file_name

    _download(file_name)

    words = open(file_path).read().replace("\n", "<eos>").strip().split()

    for i, word in enumerate(words):
        if word not in word_to_id:
            tmp_id = len(word_to_id)
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word

    with open(vocab_path, mode="wb") as file:
        pickle.dump((word_to_id, id_to_word), file)

    return word_to_id, id_to_word


def load_data(data_type: DATA_TYPE = "train") -> tuple[np.ndarray, dict[str, int], dict[int, str]]:
    save_path = DATASET_DIR_PATH / SAVE_FILES[data_type]

    word_to_id, id_to_word = load_vocab()

    if save_path.exists():
        corpus = np.load(save_path)
        return corpus, word_to_id, id_to_word

    file_name = KEY_FILES[data_type]
    file_path = DATASET_DIR_PATH / file_name
    _download(file_name)

    words = open(file_path).read().replace("\n", "<eos>").strip().split()
    corpus = np.array([word_to_id[w] for w in words])

    np.save(save_path, corpus)
    return corpus, word_to_id, id_to_word


if __name__ == "__main__":
    for data_type in get_args(DATA_TYPE):
        load_data(data_type)
