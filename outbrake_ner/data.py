"""Dataset loading, tokenisation and ``tf.data`` pipeline construction."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .preprocessing import preprocess_sentence

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Reading the CoNLL-style training file
# --------------------------------------------------------------------------- #
def read_conll_pairs(path: str | Path) -> List[Tuple[str, str]]:
    """Read ``<token> <tag>`` lines from *path*.

    Mirrors the notebook: a line is used only when it splits into exactly two
    whitespace separated fields; blank (sentence separator) lines are skipped.
    """
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8-sig") as handle:  # utf-8-sig drops a leading BOM
        for line in handle:
            fields = line.split()
            if len(fields) == 2:
                pairs.append((fields[0], fields[1]))
    if not pairs:
        raise ValueError(f"No '<token> <tag>' pairs found in {path}")
    logger.info("Read %d token/tag pairs from %s", len(pairs), path)
    return pairs


def build_source_target(pairs: Iterable[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    """Apply the notebook's preprocessing to every word and tag."""
    source, target = [], []
    for word, tag in pairs:
        source.append(preprocess_sentence(word))
        target.append(preprocess_sentence(tag))
    return source, target


# --------------------------------------------------------------------------- #
# Tokeniser (drop-in replacement for tf.keras.preprocessing.text.Tokenizer,
# which no longer exists in Keras 3).  Index 0 is reserved for padding.
# --------------------------------------------------------------------------- #
class Tokenizer:
    """Whitespace tokeniser with a frequency-ordered vocabulary."""

    def __init__(self) -> None:
        self.word_index: Dict[str, int] = {}
        self.index_word: Dict[int, str] = {}

    # -- fitting ------------------------------------------------------------
    def fit_on_texts(self, texts: Iterable[str]) -> "Tokenizer":
        counts: Counter = Counter()
        for text in texts:
            counts.update(text.split())
        # most frequent first; ties keep first-seen order (same as Keras)
        ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        self.word_index = {word: idx for idx, (word, _) in enumerate(ordered, start=1)}
        self.index_word = {idx: word for word, idx in self.word_index.items()}
        return self

    # -- encoding -----------------------------------------------------------
    def texts_to_sequences(self, texts: Iterable[str]) -> List[List[int]]:
        # unknown words are dropped, exactly like the Keras tokenizer
        return [[self.word_index[w] for w in t.split() if w in self.word_index] for t in texts]

    @property
    def vocab_size(self) -> int:
        """Vocabulary size *including* the padding index 0."""
        return len(self.word_index) + 1

    # -- (de)serialisation ---------------------------------------------------
    def to_dict(self) -> Dict[str, int]:
        return dict(self.word_index)

    @classmethod
    def from_dict(cls, word_index: Dict[str, int]) -> "Tokenizer":
        tok = cls()
        tok.word_index = {w: int(i) for w, i in word_index.items()}
        tok.index_word = {i: w for w, i in tok.word_index.items()}
        return tok


def pad_sequences(sequences: Sequence[Sequence[int]], maxlen: int | None = None) -> np.ndarray:
    """Post-pad integer sequences with zeros into a ``(n, maxlen)`` int32 array."""
    if maxlen is None:
        maxlen = max((len(s) for s in sequences), default=0)
    out = np.zeros((len(sequences), maxlen), dtype=np.int32)
    for i, seq in enumerate(sequences):
        seq = list(seq)[:maxlen]
        out[i, : len(seq)] = seq
    return out


def tokenize(texts: Sequence[str]) -> Tuple[np.ndarray, Tokenizer]:
    tokenizer = Tokenizer().fit_on_texts(texts)
    tensor = pad_sequences(tokenizer.texts_to_sequences(texts))
    return tensor, tokenizer


# --------------------------------------------------------------------------- #
# End-to-end preparation
# --------------------------------------------------------------------------- #
@dataclass
class PreparedData:
    input_tensor_train: np.ndarray
    target_tensor_train: np.ndarray
    input_tensor_val: np.ndarray
    target_tensor_val: np.ndarray
    inp_tokenizer: Tokenizer
    targ_tokenizer: Tokenizer

    @property
    def max_length_inp(self) -> int:
        return int(self.input_tensor_train.shape[1])

    @property
    def max_length_targ(self) -> int:
        return int(self.target_tensor_train.shape[1])


def prepare_data(data_path: str | Path, validation_split: float = 0.2, seed: int | None = None) -> PreparedData:
    """Read, preprocess, tokenise and split the training file."""
    pairs = read_conll_pairs(data_path)
    source, target = build_source_target(pairs)

    input_tensor, inp_tokenizer = tokenize(source)
    target_tensor, targ_tokenizer = tokenize(target)

    if validation_split > 0:
        x_train, x_val, y_train, y_val = train_test_split(
            input_tensor, target_tensor, test_size=validation_split, random_state=seed
        )
    else:
        x_train, y_train = input_tensor, target_tensor
        x_val = np.zeros((0, input_tensor.shape[1]), dtype=input_tensor.dtype)
        y_val = np.zeros((0, target_tensor.shape[1]), dtype=target_tensor.dtype)

    logger.info(
        "Train examples: %d | held-out examples: %d | input vocab: %d | target vocab: %d | "
        "max input len: %d | max target len: %d",
        len(x_train), len(x_val), inp_tokenizer.vocab_size, targ_tokenizer.vocab_size,
        input_tensor.shape[1], target_tensor.shape[1],
    )
    return PreparedData(x_train, y_train, x_val, y_val, inp_tokenizer, targ_tokenizer)


def make_dataset(inputs: np.ndarray, targets: np.ndarray, batch_size: int, seed: int | None = None) -> tf.data.Dataset:
    """Shuffled, batched ``tf.data.Dataset`` (incomplete final batch dropped)."""
    buffer_size = len(inputs)
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(buffer_size, seed=seed)
    return dataset.batch(batch_size, drop_remainder=True)
