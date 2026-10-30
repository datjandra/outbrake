"""Custom training loop with teacher forcing and early stopping."""

from __future__ import annotations

import logging
import sys
import time
from typing import Callable, Optional

import tensorflow as tf

from .config import TrainingConfig
from .model import Decoder, Encoder
from .preprocessing import START_TOKEN

logger = logging.getLogger(__name__)


class Seq2SeqTrainer:
    """Wires encoder, decoder, optimiser and loss into a training loop."""

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        start_token_id: int,
        config: TrainingConfig,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.start_token_id = start_token_id
        self.config = config
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
        self._train_step = tf.function(self._train_step_impl)

    # ------------------------------------------------------------------ loss
    def loss_function(self, real: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
        """Masked sparse cross-entropy: padded positions (id 0) are ignored."""
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        return tf.reduce_mean(loss_ * mask)

    # ------------------------------------------------------------------ step
    def _train_step_impl(self, inp: tf.Tensor, targ: tf.Tensor, enc_hidden: tf.Tensor) -> tf.Tensor:
        loss = 0.0
        batch_size = self.config.batch_size

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp, enc_hidden, training=True)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([self.start_token_id] * batch_size, 1)

            # Teacher forcing: feed the ground-truth token as the next input
            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output, training=True)
                loss += self.loss_function(targ[:, t], predictions)
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = loss / int(targ.shape[1])
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    # ------------------------------------------------------------------ loop
    def fit(
        self,
        dataset: tf.data.Dataset,
        steps_per_epoch: int,
        on_epoch_end: Optional[Callable[[int, float], None]] = None,
    ) -> dict:
        """Train for ``config.epochs`` epochs or until early stopping triggers.

        Returns a small history dict with the per-epoch total loss.
        """
        cfg = self.config
        history = {"total_loss": []}

        best_total_loss = sys.maxsize
        stopping_step = 0
        training_start = time.time()

        logger.info(
            "Starting training of at most %d epochs or if no improvement after %d epochs",
            cfg.epochs, cfg.patience,
        )

        for epoch in range(cfg.epochs):
            start = time.time()
            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0.0

            for batch, (inp, targ) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = self._train_step(inp, targ, enc_hidden)
                total_loss += float(batch_loss)
                if cfg.log_every and batch % cfg.log_every == 0:
                    logger.info("Epoch %d Batch %d Loss %.4f", epoch + 1, batch, float(batch_loss))

            history["total_loss"].append(total_loss)
            logger.info("Epoch %d Total Loss %.4f", epoch + 1, total_loss)
            logger.info("Time taken for this epoch %.4f sec", time.time() - start)

            if on_epoch_end is not None:
                on_epoch_end(epoch + 1, total_loss)

            # early stopping on the training loss, as in the notebook
            if total_loss < best_total_loss:
                stopping_step = 0
                best_total_loss = total_loss
            else:
                stopping_step += 1
            if stopping_step >= cfg.patience:
                logger.info("Early stopping: no improvement for %d epochs", cfg.patience)
                break

        logger.info("Training finished in %.1f sec (best total loss %.4f)", time.time() - training_start, best_total_loss)
        history["best_total_loss"] = best_total_loss
        return history


def start_token_id(word_index: dict) -> int:
    return word_index[START_TOKEN]
