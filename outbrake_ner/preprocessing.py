"""Text normalisation helpers (kept identical to the original notebook).

Every source word and every tag is normalised with :func:`preprocess_sentence`,
which lower-cases, strips accents, replaces any character outside
``0-9 a-z ? . ! , ¿`` by a space and wraps the result in ``<start>``/``<end>``
tokens.  As a consequence a tag such as ``B-DISEASE`` becomes
``<start> b disease <end>`` - the decoder learns to emit the BIO prefix and
the entity type as two separate tokens.
"""

from __future__ import annotations

import re
import unicodedata

START_TOKEN = "<start>"
END_TOKEN = "<end>"

_NON_ALLOWED = re.compile(r"[^0-9a-zA-Z?.!,¿]+")


def unicode_to_ascii(s: str) -> str:
    """Remove diacritics: ``'é' -> 'e'``."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def preprocess_sentence(w: str) -> str:
    """Normalise a word / tag and wrap it with start and end tokens."""
    w = unicode_to_ascii(w.lower().strip())
    w = _NON_ALLOWED.sub(" ", w)
    w = w.rstrip().strip()
    return f"{START_TOKEN} {w} {END_TOKEN}"
