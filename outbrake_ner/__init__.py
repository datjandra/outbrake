"""Modular training code for the outbrake NER seq2seq model.

The package is split into:

* ``config``        - training hyper-parameters and paths
* ``preprocessing`` - text normalisation (identical to the original notebook)
* ``data``          - reading ``train.txt``, tokenising and building ``tf.data`` pipelines
* ``model``         - Encoder / BahdanauAttention / Decoder Keras models
* ``training``      - the custom training loop with teacher forcing and early stopping
* ``persistence``   - saving (and re-loading) the trained artifacts to local disk
"""

__all__ = ["config", "preprocessing", "data", "model", "training", "persistence"]
