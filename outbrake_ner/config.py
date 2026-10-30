"""Hyper-parameters and paths used by the training pipeline.

The defaults reproduce the values hard-coded in the original notebook.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class TrainingConfig:
    # ---- data -------------------------------------------------------------
    data_path: str = "train.txt"
    output_dir: str = "artifacts"
    validation_split: float = 0.2  # 80/20 split, as in the notebook (held-out part is not trained on)
    seed: int = 42

    # ---- model ------------------------------------------------------------
    embedding_dim: int = 1024
    units: int = 1024
    dropout: float = 0.2

    # ---- optimisation -----------------------------------------------------
    batch_size: int = 8
    epochs: int = 32
    patience: int = 16  # early stopping: epochs without improvement in total loss
    learning_rate: float = 1e-3  # tf.keras.optimizers.Adam() default

    # ---- bookkeeping --------------------------------------------------------
    log_every: int = 8  # print batch loss every N batches
    checkpoint_every: int = 0  # save intermediate weights every N epochs (0 = disabled)

    # populated automatically after the data has been processed
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ utils
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrainingConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
