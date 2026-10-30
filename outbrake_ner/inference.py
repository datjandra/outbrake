"""Inference helpers: greedy decoding of a single word into its tag and
grouping of BIO-tagged tokens into entities.

Mirrors the ``evaluate`` / ``resolve`` functions and the entity-grouping
logic of the original notebook.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import tensorflow as tf

from .config import TrainingConfig
from .data import Tokenizer, pad_sequences
from .model import Decoder, Encoder
from .preprocessing import END_TOKEN, START_TOKEN, preprocess_sentence

logger = logging.getLogger(__name__)

FRIENDLY_LABEL = {
    "case": "Case: ",
    "date": "Date: ",
    "disease": "Disease: ",
    "loc": "Location: ",
}

Triple = Tuple[str, str, str]  # (word, bio-prefix, entity-type)


class WordTagger:
    """Translate one word into its tag string with the trained seq2seq model."""

    def __init__(
        self,
        config: TrainingConfig,
        encoder: Encoder,
        decoder: Decoder,
        inp_tokenizer: Tokenizer,
        targ_tokenizer: Tokenizer,
    ) -> None:
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.inp_tokenizer = inp_tokenizer
        self.targ_tokenizer = targ_tokenizer
        self.max_length_inp = int(config.extra["max_length_inp"])
        self.max_length_targ = int(config.extra["max_length_targ"])
        self.start_id = targ_tokenizer.word_index[START_TOKEN]
        self._cache: Dict[str, str] = {}

    # ------------------------------------------------------------ decoding
    def evaluate(self, word: str) -> str:
        """Greedy-decode the tag sequence for *word* (includes start/end tokens)."""
        sentence = preprocess_sentence(word)
        ids = [self.inp_tokenizer.word_index.get(tok, 0) for tok in sentence.split(" ")]
        inputs = tf.convert_to_tensor(pad_sequences([ids], maxlen=self.max_length_inp))

        hidden = self.encoder.initialize_hidden_state(1)
        enc_out, enc_hidden = self.encoder(inputs, hidden, training=False)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.start_id], 0)

        result = [START_TOKEN]
        for _ in range(self.max_length_targ):
            predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_out, training=False)
            predicted_id = int(tf.argmax(predictions[0]).numpy())
            token = self.targ_tokenizer.index_word.get(predicted_id, "")
            result.append(token)
            if token == END_TOKEN:
                break
            dec_input = tf.expand_dims([predicted_id], 0)
        return " ".join(result)

    def resolve(self, word: str) -> str:
        """Tag for *word* without start/end tokens, e.g. ``'b disease'`` or ``'o'``."""
        if word not in self._cache:
            tag = self.evaluate(word)
            tag = tag.replace(START_TOKEN, "").replace(END_TOKEN, "").strip()
            self._cache[word] = tag
        return self._cache[word]

    def tag_tokens(self, tokens: Sequence[str]) -> List[Tuple[str, str]]:
        """Return ``(token, tag)`` pairs for every token not tagged ``o``."""
        word_tags = []
        for token in tokens:
            tag = self.resolve(token)
            if tag != "o":
                word_tags.append((token, tag))
        return word_tags


# ---------------------------------------------------------------- grouping
def group_entities(word_tags: Sequence[Tuple[str, str]]) -> Dict[str, List[Tuple[Triple, ...]]]:
    """Group BIO-tagged tokens into entities, keyed by entity type.

    A ``b`` tag starts a new entity; an ``i`` tag extends the most recent
    entity of the same type (or starts one if there is none).  Duplicate
    entities are removed while preserving order.
    """
    entities: Dict[str, List[List[Triple]]] = {}
    for word, tag in word_tags:
        parts = tag.split()
        if len(parts) != 2:  # malformed prediction (e.g. just 'b') - ignore
            logger.debug("Skipping malformed tag %r for token %r", tag, word)
            continue
        section, entity = parts
        triple: Triple = (word, section, entity)

        entity_list = entities.setdefault(entity, [])
        if section == "b":
            entity_list.append([triple])
        elif section == "i":
            if entity_list:
                entity_list[-1].append(triple)
            else:
                entity_list.append([triple])

    unique: Dict[str, List[Tuple[Triple, ...]]] = {}
    for entity, items in entities.items():
        unique[entity] = list(dict.fromkeys(tuple(item) for item in items))
    return unique


def format_entities(entities: Dict[str, List[Tuple[Triple, ...]]]) -> List[str]:
    """Human-readable lines such as ``'Disease: Ebola virus'``."""
    lines = []
    for entity, items in entities.items():
        label = FRIENDLY_LABEL.get(entity, f"{entity.capitalize()}: ")
        for item in items:
            lines.append(label + " ".join(t[0] for t in item))
    return lines
