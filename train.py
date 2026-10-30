#!/usr/bin/env python
"""Train the outbrake NER seq2seq model and save it to local disk.

Example
-------
    python train.py --data-path train.txt --output-dir artifacts

All hyper-parameters default to the values used in the original notebook and
can be overridden from the command line (``python train.py --help``).
"""

from __future__ import annotations

import argparse
import logging
import random

import numpy as np
import tensorflow as tf

from outbrake_ner.config import TrainingConfig
from outbrake_ner.data import make_dataset, prepare_data
from outbrake_ner.model import build_models
from outbrake_ner.persistence import save_artifacts, save_checkpoint
from outbrake_ner.training import Seq2SeqTrainer, start_token_id


def parse_args() -> TrainingConfig:
    defaults = TrainingConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", default=defaults.data_path, help="CoNLL style '<token> <tag>' file")
    p.add_argument("--output-dir", default=defaults.output_dir, help="where the trained model is written")
    p.add_argument("--validation-split", type=float, default=defaults.validation_split,
                   help="fraction of examples held out (not trained on). 0 trains on everything")
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--embedding-dim", type=int, default=defaults.embedding_dim)
    p.add_argument("--units", type=int, default=defaults.units, help="GRU / attention units")
    p.add_argument("--dropout", type=float, default=defaults.dropout)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--epochs", type=int, default=defaults.epochs)
    p.add_argument("--patience", type=int, default=defaults.patience, help="early stopping patience (epochs)")
    p.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    p.add_argument("--log-every", type=int, default=defaults.log_every, help="log batch loss every N batches")
    p.add_argument("--checkpoint-every", type=int, default=defaults.checkpoint_every,
                   help="save intermediate weights every N epochs (0 = only save the final model)")
    args = p.parse_args()
    return TrainingConfig(**{k: v for k, v in vars(args).items()})


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = parse_args()
    set_seed(config.seed)

    # 1. data ---------------------------------------------------------------
    data = prepare_data(config.data_path, config.validation_split, config.seed)
    dataset = make_dataset(data.input_tensor_train, data.target_tensor_train, config.batch_size, config.seed)
    steps_per_epoch = len(data.input_tensor_train) // config.batch_size
    if steps_per_epoch == 0:
        raise ValueError("Not enough training examples for a single batch; lower --batch-size")

    # 2. model --------------------------------------------------------------
    encoder, decoder = build_models(
        vocab_inp_size=data.inp_tokenizer.vocab_size,
        vocab_tar_size=data.targ_tokenizer.vocab_size,
        embedding_dim=config.embedding_dim,
        units=config.units,
        batch_size=config.batch_size,
        dropout=config.dropout,
    )

    # 3. train --------------------------------------------------------------
    trainer = Seq2SeqTrainer(encoder, decoder, start_token_id(data.targ_tokenizer.word_index), config)

    def on_epoch_end(epoch: int, _total_loss: float) -> None:
        if config.checkpoint_every and epoch % config.checkpoint_every == 0:
            save_checkpoint(f"{config.output_dir}/checkpoints", encoder, decoder, epoch)

    trainer.fit(dataset, steps_per_epoch, on_epoch_end=on_epoch_end)

    # 4. save ---------------------------------------------------------------
    save_artifacts(
        config.output_dir,
        config,
        encoder,
        decoder,
        data.inp_tokenizer,
        data.targ_tokenizer,
        max_length_inp=data.max_length_inp,
        max_length_targ=data.max_length_targ,
    )


if __name__ == "__main__":
    main()
