"""Save / load the trained model and everything needed to use it again."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import tensorflow as tf

from .config import TrainingConfig
from .data import Tokenizer
from .model import Decoder, Encoder, build_models, build_variables

logger = logging.getLogger(__name__)

CONFIG_FILE = "config.json"
TOKENIZERS_FILE = "tokenizers.json"
ENCODER_WEIGHTS = "encoder.weights.h5"
DECODER_WEIGHTS = "decoder.weights.h5"


def save_artifacts(
    output_dir: str | Path,
    config: TrainingConfig,
    encoder: Encoder,
    decoder: Decoder,
    inp_tokenizer: Tokenizer,
    targ_tokenizer: Tokenizer,
    max_length_inp: int,
    max_length_targ: int,
) -> Path:
    """Write weights, tokenisers and config to *output_dir* and return it."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config.extra.update(
        {
            "max_length_inp": int(max_length_inp),
            "max_length_targ": int(max_length_targ),
            "vocab_inp_size": inp_tokenizer.vocab_size,
            "vocab_tar_size": targ_tokenizer.vocab_size,
        }
    )
    config.save(output_dir / CONFIG_FILE)

    (output_dir / TOKENIZERS_FILE).write_text(
        json.dumps({"input": inp_tokenizer.to_dict(), "target": targ_tokenizer.to_dict()}, indent=2)
    )

    encoder.save_weights(str(output_dir / ENCODER_WEIGHTS))
    decoder.save_weights(str(output_dir / DECODER_WEIGHTS))
    logger.info("Saved model artifacts to %s", output_dir.resolve())
    return output_dir


def save_checkpoint(checkpoint_dir: str | Path, encoder: Encoder, decoder: Decoder, epoch: int) -> Path:
    """Save intermediate encoder/decoder weights for a given epoch."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    encoder.save_weights(str(checkpoint_dir / f"encoder_epoch{epoch:03d}.weights.h5"))
    decoder.save_weights(str(checkpoint_dir / f"decoder_epoch{epoch:03d}.weights.h5"))
    return checkpoint_dir


def load_artifacts(artifact_dir: str | Path) -> Tuple[TrainingConfig, Encoder, Decoder, Tokenizer, Tokenizer]:
    """Rebuild encoder/decoder from a directory written by :func:`save_artifacts`."""
    artifact_dir = Path(artifact_dir)
    config = TrainingConfig.load(artifact_dir / CONFIG_FILE)

    tok_payload = json.loads((artifact_dir / TOKENIZERS_FILE).read_text())
    inp_tokenizer = Tokenizer.from_dict(tok_payload["input"])
    targ_tokenizer = Tokenizer.from_dict(tok_payload["target"])

    encoder, decoder = build_models(
        vocab_inp_size=config.extra["vocab_inp_size"],
        vocab_tar_size=config.extra["vocab_tar_size"],
        embedding_dim=config.embedding_dim,
        units=config.units,
        batch_size=config.batch_size,
        dropout=config.dropout,
    )
    build_variables(encoder, decoder, config.extra["max_length_inp"])
    encoder.load_weights(str(artifact_dir / ENCODER_WEIGHTS))
    decoder.load_weights(str(artifact_dir / DECODER_WEIGHTS))
    return config, encoder, decoder, inp_tokenizer, targ_tokenizer
