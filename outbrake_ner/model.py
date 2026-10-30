"""Encoder / attention / decoder architecture (as in the original notebook)."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf


class Encoder(tf.keras.Model):
    """Embedding -> GRU encoder returning the full sequence and final state."""

    def __init__(self, vocab_size: int, embedding_dim: int, enc_units: int, batch_sz: int, dropout: float = 0.2):
        super().__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, hidden, training=False):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        output = self.dropout(output, training=training)
        return output, state

    def initialize_hidden_state(self, batch_size: int | None = None) -> tf.Tensor:
        return tf.zeros((batch_size or self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    """Additive attention (Bahdanau et al., 2015)."""

    def __init__(self, units: int):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query: (batch, hidden) -> (batch, 1, hidden) so it broadcasts over time
        hidden_with_time_axis = tf.expand_dims(query, 1)
        # score: (batch, max_length, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        # context: (batch, hidden)
        context_vector = tf.reduce_sum(attention_weights * values, axis=1)
        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    """Attention-based GRU decoder producing one output token per call."""

    def __init__(self, vocab_size: int, embedding_dim: int, dec_units: int, batch_sz: int, dropout: float = 0.2):
        super().__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(dec_units)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, hidden, enc_output, training=False):
        # enc_output: (batch, max_length, hidden)
        context_vector, attention_weights = self.attention(hidden, enc_output)
        context_vector = self.dropout(context_vector, training=training)

        # x: (batch, 1) -> (batch, 1, embedding_dim)
        x = self.embedding(x)
        # (batch, 1, embedding_dim + hidden)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)
        # (batch, hidden)
        output = tf.reshape(output, (-1, output.shape[2]))
        # (batch, vocab)
        x = self.fc(output)
        return x, state, attention_weights


def build_models(
    vocab_inp_size: int,
    vocab_tar_size: int,
    embedding_dim: int,
    units: int,
    batch_size: int,
    dropout: float = 0.2,
) -> Tuple[Encoder, Decoder]:
    """Instantiate an (encoder, decoder) pair from the hyper-parameters."""
    encoder = Encoder(vocab_inp_size, embedding_dim, units, batch_size, dropout)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, batch_size, dropout)
    return encoder, decoder


def build_variables(encoder: Encoder, decoder: Decoder, max_length_inp: int, batch_size: int = 1) -> None:
    """Run a dummy forward pass so that all weights are created.

    Subclassed Keras models only create their variables on first call; this is
    required before ``load_weights`` and is also useful before saving.
    """
    dummy_inp = tf.zeros((batch_size, max_length_inp), dtype=tf.int32)
    hidden = encoder.initialize_hidden_state(batch_size)
    enc_output, enc_hidden = encoder(dummy_inp, hidden, training=False)
    dec_input = tf.zeros((batch_size, 1), dtype=tf.int32)
    decoder(dec_input, enc_hidden, enc_output, training=False)
